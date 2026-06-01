import unittest

from sglang.shiftserve.client import build_request_specs
from sglang.shiftserve.config import DeploymentConfig, TrafficConfig
from sglang.shiftserve.launcher import LaunchCommandBuilder, LaunchDefaults, PortAllocator
from sglang.shiftserve.scheduler import (
    HysteresisFlipMonitor,
    InstanceRuntimeState,
    RequestEstimate,
    SchedulerMode,
    ShiftServeScheduler,
    StageCostProfile,
    StageKind,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")


class TestShiftServeConfig(CustomTestCase):
    def test_deployment_config_validates_ports_and_aliases(self):
        deployment = DeploymentConfig.from_dict(
            {
                "nodes": [{"node_id": "a"}],
                "instances": [
                    {
                        "id": "llm0",
                        "kind": "llm",
                        "node_id": "a",
                        "ports": {"http": 30000},
                    },
                    {
                        "id": "ditvae0",
                        "kind": "ddit_worker",
                        "node_id": "a",
                        "ports": {"work": 31000, "control": 31001},
                    },
                ],
                "bins": {"short": 512, "long": 2048},
            }
        )
        self.assertEqual(deployment.instances["llm0"].kind, "pe")
        self.assertEqual(deployment.instances["ditvae0"].kind, "dit_vae")
        PortAllocator.validate(deployment)

    def test_port_collision_raises(self):
        deployment = DeploymentConfig.from_dict(
            {
                "nodes": [{"node_id": "a"}],
                "instances": [
                    {"id": "pe0", "kind": "pe", "node_id": "a", "ports": {"http": 1}},
                    {
                        "id": "te0",
                        "kind": "te",
                        "node_id": "a",
                        "ports": {"work": 1, "control": 2},
                    },
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, "Port collision"):
            PortAllocator.validate(deployment)


class TestShiftServeScheduler(CustomTestCase):
    def test_round_robin_does_not_use_weighted_costs(self):
        instances = [
            InstanceRuntimeState("pe0", StageKind.PE, "a"),
            InstanceRuntimeState("pe1", StageKind.PE, "a"),
        ]
        scheduler = ShiftServeScheduler(
            mode=SchedulerMode.ROUND_ROBIN,
            cost_profile=StageCostProfile(ttft_ms=999999, tpot_ms=999999),
        )
        request = RequestEstimate("r0", StageKind.PE, bin_tokens=2048)
        self.assertEqual(scheduler.select(instances, request).instance_id, "pe0")
        self.assertEqual(scheduler.select(instances, request).instance_id, "pe1")

    def test_weighted_selects_lower_remaining_work(self):
        instances = [
            InstanceRuntimeState("dit0", StageKind.DIT, "a"),
            InstanceRuntimeState("dit1", StageKind.DIT, "a"),
        ]
        instances[0].queue.extend(["old0", "old1"])
        scheduler = ShiftServeScheduler(
            mode=SchedulerMode.WEIGHTED,
            cost_profile=StageCostProfile(dit_steps=10, dit_per_step_ms=5),
        )
        result = scheduler.select(
            instances,
            RequestEstimate("r0", StageKind.DIT, remaining_steps=10),
        )
        self.assertEqual(result.instance_id, "dit1")
        self.assertGreater(result.estimated_work_ms, 0)

    def test_hysteresis_flip_monitor_uses_margin(self):
        monitor = HysteresisFlipMonitor(
            short_bin=512,
            long_bin=2048,
            window_size=2,
            margin_enabled=True,
            margin_ratio=0.1,
        )
        self.assertIsNone(monitor.record_completion(1400))
        self.assertEqual(monitor.record_completion(2048), "short_to_long")
        self.assertIsNone(monitor.record_completion(900))
        self.assertEqual(monitor.record_completion(512), "long_to_short")

    def test_same_node_fallback_records_reason(self):
        instances = [InstanceRuntimeState("dit0", StageKind.DIT, "b")]
        scheduler = ShiftServeScheduler(mode=SchedulerMode.ROUND_ROBIN)
        result = scheduler.select(
            instances,
            RequestEstimate("r0", StageKind.DIT, target_node_id="a"),
        )
        self.assertEqual(result.instance_id, "dit0")
        self.assertEqual(result.fallback_reason, "same_node_or_pipeline_group_unavailable")


class TestShiftServeLauncherAndTraffic(CustomTestCase):
    def test_launch_defaults_include_required_flags(self):
        deployment = DeploymentConfig.from_dict(
            {
                "nodes": [{"node_id": "a", "host": "127.0.0.1"}],
                "instances": [
                    {
                        "id": "pe0",
                        "kind": "pe",
                        "node_id": "a",
                        "ranks": 1,
                        "ports": {"http": 30000},
                    },
                    {
                        "id": "dit0",
                        "kind": "dit",
                        "node_id": "a",
                        "gpu_ids": [0],
                        "ports": {"work": 31000, "control": 31001},
                    },
                ],
            }
        )
        builder = LaunchCommandBuilder(LaunchDefaults(rank0_broadcast=True))
        commands = builder.build_launch_plan(
            deployment,
            pe_model_path="PE",
            diffusion_model_path="WAN",
            server_addr="tcp://127.0.0.1:5555",
            weighted_schedule=True,
        )
        pe = commands["pe0"]
        dit = commands["dit0"]
        self.assertIn("--disable-piecewise-cuda-graph", pe)
        self.assertIn("--cuda-graph-max-bs", pe)
        self.assertIn("4096", pe)
        self.assertIn("--disagg-transfer-pool-size", dit)
        self.assertIn("--disagg-transfer-calibration-mode", dit)
        self.assertIn("fixed", dit)
        self.assertIn(str(2 * 1024 * 1024 * 1024), dit)
        self.assertIn("--pin-cpu-memory", dit)
        self.assertIn("false", dit)
        self.assertIn("rank0-broadcast", dit)

    def test_traffic_plan_generates_short_and_long_requests(self):
        traffic = TrafficConfig.from_dict(
            {
                "duration_min": 2,
                "default_rate_per_min": 1,
                "intervals": [
                    {"start_min": 0, "end_min": 1, "rate_per_min": 2, "bin": "short"},
                    {"start_min": 1, "end_min": 2, "rate_per_min": 1, "bin": "long"},
                ],
            }
        )
        specs = build_request_specs(traffic)
        self.assertEqual([spec.output_tokens for spec in specs], [512, 512, 2048])


if __name__ == "__main__":
    unittest.main(verbosity=3)
