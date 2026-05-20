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

    def test_reference_aligned_command_uses_old_zimage_template(self):
        class Args:
            python = "python3"
            model_path = "/data/Z_Image"
            model_id = "Z-Image"
            profile_output_dir = "/data/profile"
            num_gpus = 4
            sp_degree = 4
            ulysses_degree = 2
            ring_degree = 2
            attention_backend = None
            host = "127.0.0.1"
            launch_entrypoint = "module"
            diagnostic_module_profile = False
            compat_profile_preset = "none"
            extra_launch_arg = []
            extra_launch_args = ""
            _current_ports = {
                "port": 30001,
                "scheduler_port": 30002,
                "master_port": 30003,
            }
            _current_run_profile_dir = pathlib.Path("/tmp/run0_launch_benchmark")

        command = self.benchmark.build_command(Args, "baseline", "run0")

        self.assertEqual(
            command[:4],
            [
                "python3",
                "-m",
                "sglang.multimodal_gen.runtime.entrypoints.cli.main",
                "serve",
            ],
        )
        self.assertIn("--port", command)
        self.assertIn("30001", command)
        self.assertNotIn("--attention-backend", command)
        self.assertIn("--warmup", command)
        self.assertEqual(command[command.index("--warmup") + 1], "false")
        self.assertIn("--text-encoder-cpu-offload", command)
        idx = command.index("--text-encoder-cpu-offload")
        self.assertEqual(command[idx + 1], "false")
        self.assertIn("--dit-layerwise-offload", command)
        self.assertIn("--tp-size", command)
        self.assertEqual(command[command.index("--tp-size") + 1], "1")
        self.assertIn("--pin-cpu-memory", command)

    def test_weight_load_setups_only_add_allowed_diffusion_weight_args(self):
        class Args:
            python = "python3"
            model_path = "/data/Z_Image"
            model_id = "Z-Image"
            profile_output_dir = "/data/profile"
            num_gpus = 4
            sp_degree = 4
            ulysses_degree = 2
            ring_degree = 2
            attention_backend = None
            host = "127.0.0.1"
            launch_entrypoint = "module"
            diagnostic_module_profile = False
            compat_profile_preset = "none"
            extra_launch_arg = []
            extra_launch_args = ""
            _current_ports = {
                "port": 30001,
                "scheduler_port": 30002,
                "master_port": 30003,
            }
            _current_run_profile_dir = pathlib.Path("/tmp/run0_launch_benchmark")

        baseline = self.benchmark.build_command(Args, "baseline", "run0")
        pageable = self.benchmark.build_command(Args, "pageable", "run0")

        self.assertEqual(
            self.benchmark.normalized_server_arg_map(baseline),
            self.benchmark.normalized_server_arg_map(pageable),
        )
        self.assertIn("--diffusion-weight-load-mode", pageable)

    def test_reference_diff_detects_attention_backend_drift(self):
        reference = [
            "python3",
            "-m",
            "sglang.multimodal_gen.runtime.entrypoints.cli.main",
            "serve",
            "--model-path",
            "/old/Z_Image",
            "--host",
            "127.0.0.1",
            "--port",
            "1",
            "--scheduler-port",
            "2",
            "--master-port",
            "3",
            "--num-gpus",
            "4",
            "--warmup",
            "false",
            "--output-path",
            "/old/outputs",
            "--input-save-path",
            "/old/uploads",
            "--log-level",
            "info",
            "--dit-cpu-offload",
            "false",
            "--dit-layerwise-offload",
            "false",
            "--text-encoder-cpu-offload",
            "false",
            "--image-encoder-cpu-offload",
            "false",
            "--vae-cpu-offload",
            "false",
            "--pin-cpu-memory",
            "false",
            "--model-id",
            "Z-Image",
            "--tp-size",
            "1",
            "--sp-degree",
            "4",
            "--ulysses-degree",
            "2",
            "--ring-degree",
            "2",
        ]
        current = reference + ["--profile-enabled", "--attention-backend", "fa"]

        diff = self.benchmark.diff_reference_command(reference, current)

        self.assertEqual(diff, ["--attention-backend: expected=None actual=['fa']"])

    def test_validate_server_args_detects_offload_violation(self):
        errors = self.benchmark.validate_server_args(
            {
                "dit_cpu_offload": False,
                "dit_layerwise_offload": False,
                "text_encoder_cpu_offload": True,
                "image_encoder_cpu_offload": False,
                "vae_cpu_offload": False,
                "pin_cpu_memory": False,
                "use_fsdp_inference": False,
                "tp_size": 1,
            }
        )

        self.assertEqual(
            errors,
            ["text_encoder_cpu_offload: expected=False actual=True"],
        )

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
