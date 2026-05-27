import importlib.util
import os
import tempfile
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
    spec.loader.exec_module(module)
    return module


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
