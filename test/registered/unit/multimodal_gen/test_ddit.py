import importlib.util
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.ddit.config import (
    build_execution_plan,
    parse_switch_plan,
    resolve_vae_ranks,
)
from sglang.multimodal_gen.runtime.ddit.logging import LifecycleCsvLogger
from sglang.multimodal_gen.runtime.ddit.scheduler import (
    DDiTRequestState,
    DDiTSchedulerConfig,
    HungryFirstScheduler,
    RequestPhase,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


def _load_client_module():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    path = os.path.join(
        root, "examples", "multimodal_gen", "ddit_mixed_workload_client.py"
    )
    spec = importlib.util.spec_from_file_location("ddit_mixed_workload_client", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_mock_simulator_module():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    path = os.path.join(
        root, "examples", "multimodal_gen", "ddit_mock_simulator.py"
    )
    spec = importlib.util.spec_from_file_location("ddit_mock_simulator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_mock_profile(tmp_dir):
    path = os.path.join(tmp_dir, "profile.json")
    payload = {
        "models": {
            "z-image": {
                "opt_gpus_num": {"720p": 2, "2k": 2},
                "opt_vae_k": {"720p": 1, "2k": 1},
                "dit_step_times": {
                    "720p": {"1": 10.0, "2": 1.0, "8": 1.0},
                    "2k": {"1": 12.0, "2": 2.0, "8": 2.0},
                },
                "vae_times": {
                    "720p": {"1": 0.1, "2": 0.2, "8": 0.2},
                    "2k": {"1": 0.1, "2": 0.2, "8": 0.2},
                },
                "text_encoder_times": {"720p": 0.1, "2k": 0.1},
                "dit_step_num": 2,
            }
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        import json

        json.dump(payload, f)
    return path


class TestDDiTConfig(CustomTestCase):
    def test_parse_switch_plan_count_transitions(self):
        events = parse_switch_plan("15:1->2;30:2->4;45:4->8")
        self.assertEqual([event.after_step for event in events], [15, 30, 45])
        self.assertEqual(events[0].ranks, (0, 1))
        self.assertEqual(events[2].ranks, tuple(range(8)))

    def test_build_execution_plan_validates_local_rank_scope(self):
        server_args = SimpleNamespace(
            ddit_allowed_gpu_counts="1,2,4,8",
            ddit_initial_gpus=1,
            ddit_initial_ranks=None,
            ddit_switch_plan="15:0,1",
        )
        batch = SimpleNamespace(extra={})
        plan = build_execution_plan(server_args, batch, world_size=8)
        self.assertEqual(plan.initial_ranks, (0,))
        self.assertEqual(plan.switch_after(15).ranks, (0, 1))

    def test_resolve_vae_ranks_supports_k_and_explicit_ranks(self):
        server_args = SimpleNamespace(ddit_allowed_gpu_counts="1,2,4,8", ddit_vae_gpus=2)
        batch = SimpleNamespace(extra={})
        self.assertEqual(
            resolve_vae_ranks(
                server_args, batch, world_size=8, final_dit_ranks=(0, 1, 2, 3)
            ),
            (0, 1),
        )
        batch = SimpleNamespace(extra={"ddit_vae_ranks": "2,3"})
        self.assertEqual(
            resolve_vae_ranks(server_args, batch, world_size=8, final_dit_ranks=(0, 1)),
            (2, 3),
        )


class TestHungryFirstScheduler(CustomTestCase):
    def test_144p_release_promotes_720p_then_vae_keeps_k(self):
        scheduler = HungryFirstScheduler(
            DDiTSchedulerConfig(
                local_ranks=tuple(range(8)),
                opt_gpus_num={"144p": 1, "720p": 8},
                dit_step_times={
                    "144p": {1: 1.0, 2: 1.1, 4: 1.2, 8: 1.3},
                    "720p": {1: 80.0, 2: 40.0, 4: 20.0, 8: 10.0},
                },
                allowed_gpu_counts=(1, 2, 4, 8),
            )
        )
        scheduler.add_request(DDiTRequestState("small", "144p", total_steps=50))
        scheduler.add_request(
            DDiTRequestState("large", "720p", total_steps=50, vae_k=2)
        )
        decisions = scheduler.schedule()
        self.assertEqual(decisions[0]["new_ranks"], (0,))
        self.assertEqual(decisions[1]["new_ranks"], (1, 2, 3, 4))

        scheduler.update_cur_step("large", 10)
        scheduler.complete_vae("small")
        decisions = scheduler.schedule()
        self.assertEqual(decisions[0]["request_id"], "large")
        self.assertEqual(decisions[0]["new_ranks"], tuple(range(8)))

        vae_ranks = scheduler.complete_dit("large", vae_k=2)
        self.assertEqual(vae_ranks, (0, 1))
        self.assertEqual(scheduler.requests["large"].phase, RequestPhase.VAE)
        scheduler.complete_vae("large")
        self.assertTrue(all(owner is None for owner in scheduler.gpu_owner.values()))


class TestMixedWorkloadClient(CustomTestCase):
    def test_counts_round_and_last_bucket_fills_total(self):
        client = _load_client_module()
        self.assertEqual(client.counts_from_ratios(10, [0.5, 0.3, 0.2]), [5, 3, 2])
        self.assertEqual(sum(client.counts_from_ratios(7, [0.34, 0.33, 0.33])), 7)

    def test_build_workload_is_shuffled_and_reproducible(self):
        client = _load_client_module()
        workload_a = client.build_workload(
            num_requests=6,
            resolutions=["144p", "720p"],
            ratios=[0.5, 0.5],
            seed=7,
            prompt="test",
            size_map={"144p": "256x144", "720p": "1280x720"},
        )
        workload_b = client.build_workload(
            num_requests=6,
            resolutions=["144p", "720p"],
            ratios=[0.5, 0.5],
            seed=7,
            prompt="test",
            size_map={"144p": "256x144", "720p": "1280x720"},
        )
        self.assertEqual([item.resolution for item in workload_a], [item.resolution for item in workload_b])
        self.assertEqual(len(workload_a), 6)

    def test_build_workload_can_inject_ddit_vae_k(self):
        client = _load_client_module()
        workload = client.build_workload(
            num_requests=2,
            resolutions=["144p", "720p"],
            ratios=[0.5, 0.5],
            seed=7,
            prompt="test",
            size_map={"144p": "256x144", "720p": "1280x720"},
            ddit_vae_k_resolver=client.build_vae_k_resolver('{"144p":1,"720p":8}'),
        )
        by_resolution = {
            item.resolution: item.payload["ddit_vae_k"] for item in workload
        }

        self.assertEqual(by_resolution, {"144p": 1, "720p": 8})

    def test_send_workload_burst_does_not_wait_for_each_response(self):
        client = _load_client_module()
        workload = [
            client.WorkloadRequest(
                f"req_{idx}", "144p", {"request_id": f"req_{idx}"}
            )
            for idx in range(3)
        ]

        class FakeResponse:
            status_code = 200
            text = "{}"

            def json(self):
                return {}

        def fake_post(_endpoint, *, json, timeout):
            del json, timeout
            time.sleep(0.1)
            return FakeResponse()

        start = time.perf_counter()
        responses = client.send_workload(
            server_url="http://127.0.0.1:30000",
            workload=workload,
            rate=None,
            timeout=10,
            max_inflight=3,
            post_fn=fake_post,
        )

        self.assertLess(time.perf_counter() - start, 0.25)
        self.assertEqual(len(responses), 3)


class TestDDiTMockSimulator(CustomTestCase):
    def test_profile_loader_normalizes_model_and_reads_timings(self):
        simulator = _load_mock_simulator_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = _write_mock_profile(tmp_dir)
            profile = simulator.DDiTProfile.load(profile_path, "Z_Image")

        self.assertEqual(profile.model_id, "z-image")
        self.assertEqual(profile.opt_gpu_count("720p", (1, 2)), 2)
        self.assertEqual(profile.opt_vae_count("720p", (1, 2)), 1)
        self.assertEqual(profile.per_step_time("2k", 2), 2.0)
        self.assertAlmostEqual(profile.unit_slo("720p", gpu_count=8), 2.3)

    def test_hungry_scale_up_can_happen_before_last_remaining_step(self):
        simulator = _load_mock_simulator_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = _write_mock_profile(tmp_dir)
            config = simulator.SimulationConfig(
                profile_path=profile_path,
                policy="hungry_first",
                num_nodes=1,
                gpus_per_node=2,
                allowed_gpu_counts=(1, 2),
                resolutions=("720p",),
                ratios=(1.0,),
                num_requests=1,
                rate="burst",
                out_dir=tmp_dir,
            )
            sim = simulator.DDiTMockSimulator(config)
            req = sim.requests[0]
            req.node_id = 0
            req.phase = "dit"
            req.ranks = (0,)
            req.cur_step = 1
            req.total_steps = 2
            req.last_scheduled_step = 0
            sim.nodes[0].owners[0] = req.request_id

            changed = sim._schedule_scale_up(1.0)

        self.assertTrue(changed)
        self.assertEqual(req.ranks, (0, 1))

    def test_small_sweep_case_completes_all_requests(self):
        simulator = _load_mock_simulator_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = _write_mock_profile(tmp_dir)
            config = simulator.SimulationConfig(
                profile_path=profile_path,
                policy="wsjf_scale_up",
                num_nodes=2,
                gpus_per_node=2,
                allowed_gpu_counts=(1, 2),
                resolutions=("720p", "2k"),
                ratios=(0.5, 0.5),
                num_requests=4,
                rate="1.0",
                out_dir=tmp_dir,
            )
            result = simulator.run_simulation(config, write=True)

            self.assertEqual(result["summary"]["completed_count"], 4)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ddit_lifecycle.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ddit_rank_switch.jsonl")))


class TestLifecycleCsvLogger(CustomTestCase):
    def test_appends_lifespan_percentile_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "ddit_lifecycle.csv")
            logger = LifecycleCsvLogger(path)
            for request_id, add_time, vae_end_time in (
                ("r1", 10.0, 11.0),
                ("r2", 10.0, 15.0),
                ("r3", 10.0, 20.0),
            ):
                logger.record(
                    request_id=request_id,
                    resolution="720p",
                    event="add",
                    timestamp=add_time,
                )
                logger.record(
                    request_id=request_id,
                    resolution="720p",
                    event="vae_end",
                    timestamp=vae_end_time,
                )

            with open(path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            self.assertEqual(
                lines[-3:],
                ["p50,5.000000", "p90,10.000000", "p99,10.000000"],
            )

            reloaded = LifecycleCsvLogger(path)
            self.assertNotIn("p50", reloaded._rows)


if __name__ == "__main__":
    unittest.main(verbosity=3)
