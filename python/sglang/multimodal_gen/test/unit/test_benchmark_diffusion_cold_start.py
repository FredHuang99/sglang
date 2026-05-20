# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pathlib
import unittest


def _load_benchmark_module():
    repo_root = pathlib.Path(__file__).resolve().parents[5]
    path = repo_root / "scripts" / "benchmark_diffusion_cold_start.py"
    spec = importlib.util.spec_from_file_location("benchmark_diffusion_cold_start", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestDiffusionColdStartBenchmarkSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_benchmark_module()

    def test_split_setups_rejects_unknown(self):
        with self.assertRaisesRegex(ValueError, "Unknown setups"):
            self.benchmark.split_setups("baseline,missing")

    def test_summary_handles_baseline_without_broadcast_fields(self):
        result = {
            "setup": "baseline",
            "run_id": "run0",
            "launch_wall_s": 12.3,
            "ready": True,
            "start_epoch_s": 1000.0,
        }
        records = [
            {
                "component": "transformer",
                "sp_rank": "0",
                "weight_load:read_safetensors_ms": 1000.0,
                "weight_load:cpu_materialize_ms": 500.0,
                "weight_load:total_bytes": 1024**3,
            },
            {
                "component": "transformer",
                "sp_rank": "1",
                "weight_load:read_safetensors_ms": 2000.0,
                "weight_load:total_bytes": 1024**3,
            },
        ]

        module_records = [
            {
                "component": "text_encoder",
                "sp_rank": "0",
                "source": "native",
                "fallback": True,
                "duration_ms": 3000.0,
                "started_at_s": 1001.0,
                "status": "success",
            }
        ]

        summary = self.benchmark.summarize_run(result, records, module_records)

        self.assertEqual(summary["ready"], 1.0)
        self.assertEqual(summary["launch_wall_s"], 12.3)
        self.assertEqual(summary["transformer_rank0_read_s"], 1.0)
        self.assertEqual(summary["transformer_nonrank_avg_read_s"], 2.0)
        self.assertEqual(summary["transformer_nonrank_max_total_gib"], 1.0)
        self.assertEqual(summary["text_encoder_rank0_module_fallback_count"], 1.0)
        self.assertEqual(summary["text_encoder_rank0_module_native_source_count"], 1.0)
        self.assertEqual(summary["wall_unexplained_s"], 8.3)

    def test_aggregate_by_setup_emits_distribution_columns(self):
        columns, rows = self.benchmark.aggregate_by_setup(
            [
                ("pageable", {"launch_wall_s": 10.0, "ready": 1.0}),
                ("pageable", {"launch_wall_s": 12.0, "ready": 1.0}),
            ]
        )

        self.assertIn("launch_wall_s_avg", columns)
        self.assertIn("launch_wall_s_p50", columns)
        self.assertIn("launch_wall_s_min", columns)
        self.assertIn("launch_wall_s_max", columns)
        self.assertIn("launch_wall_s_std", columns)
        self.assertEqual(rows[0]["setup"], "pageable")
        self.assertEqual(rows[0]["launch_wall_s_avg"], 11.0)
        self.assertEqual(rows[0]["launch_wall_s_p50"], 11.0)
        self.assertEqual(rows[0]["launch_wall_s_min"], 10.0)
        self.assertEqual(rows[0]["launch_wall_s_max"], 12.0)
        self.assertEqual(rows[0]["launch_wall_s_std"], 1.0)

    def test_compat_preset_adds_old_zimage_flags(self):
        class Args:
            python = "python3"
            model_path = "/data/Z_Image"
            model_id = "Z-Image"
            profile_output_dir = "/data/profile"
            num_gpus = 8
            sp_degree = 8
            ulysses_degree = 2
            ring_degree = 4
            attention_backend = "fa"
            host = "127.0.0.1"
            compat_profile_preset = "zimage-launch-breakdown"
            extra_launch_arg = []
            extra_launch_args = ""
            _current_ports = {
                "port": 30001,
                "scheduler_port": 30002,
                "master_port": 30003,
            }

        command = self.benchmark.build_command(Args, "baseline", "run0")

        self.assertIn("--port", command)
        self.assertIn("30001", command)
        self.assertIn("--text-encoder-cpu-offload", command)
        idx = command.index("--text-encoder-cpu-offload")
        self.assertEqual(command[idx + 1], "false")
        self.assertIn("--pin-cpu-memory", command)

    def test_text_encoder_fallback_source_info(self):
        sources, fallback_count = self.benchmark.collect_text_encoder_source_info(
            [
                {
                    "component": "text_encoder",
                    "source": "sgl-diffusion",
                    "fallback": False,
                },
                {
                    "component": "text_encoder",
                    "source": "native",
                    "fallback": True,
                },
                {
                    "component": "vae",
                    "source": "native",
                    "fallback": True,
                },
            ]
        )

        self.assertEqual(sources, ["native", "sgl-diffusion"])
        self.assertEqual(fallback_count, 1)


if __name__ == "__main__":
    unittest.main()
