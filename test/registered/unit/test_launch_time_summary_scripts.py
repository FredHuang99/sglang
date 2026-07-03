import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def repo_root():
    return ROOT


def load_module_from_path(module_name, module_path):
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scripts_dir = ROOT / "scripts"

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except Exception:
    register_cpu_ci = load_module_from_path(
        "ci_register_for_launch_time_summary_test",
        ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py",
    ).register_cpu_ci
    CustomTestCase = unittest.TestCase


register_cpu_ci(est_time=2, suite="stage-a-cpu-only")

if "requests" not in sys.modules:
    fake_requests = types.ModuleType("requests")

    def _fake_get(*args, **kwargs):
        raise RuntimeError("requests.get is not available in this unit test")

    fake_requests.get = _fake_get
    sys.modules["requests"] = fake_requests


def load_script_module(module_name, relative_path):
    return load_module_from_path(module_name, repo_root() / relative_path)


pe7b = load_script_module(
    "profile_pe7b_launch_time_matrix_for_test",
    Path("scripts") / "profile_pe7b_launch_time_matrix.py",
)
pe7b_exec = load_script_module(
    "profile_pe7b_execution_time_matrix_for_test",
    Path("scripts") / "profile_pe7b_execution_time_matrix.py",
)
diffusion = load_script_module(
    "profile_diffusion_launch_time_summary_for_test",
    Path("scripts") / "profile_diffusion_launch_time_summary.py",
)
breakdown = load_script_module(
    "profile_server_launch_breakdown_for_summary_test",
    Path("scripts") / "profile_server_launch_breakdown.py",
)
stage_breakdown = load_script_module(
    "profile_stage_breakdown_for_summary_test",
    Path("python")
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "profile_stage_breakdown.py",
)


EXPECTED_DIFFUSION_MODEL_IDS = {
    "wan2.2-ti2v-5b": "Wan2.2-TI2V-5B-Diffusers",
    "wan2.1-t2v-1.3b": "Wan2.1-T2V-1.3B-Diffusers",
    "z-image": "Z-Image",
}


def end_event(task, elapsed_s, component=None, extra=None):
    event = {
        "phase": "end",
        "family": "sglang-diffusion",
        "task": task,
        "elapsed_s": elapsed_s,
        "status": "ok",
    }
    if component is not None:
        event["component"] = component
    if extra is not None:
        event["extra"] = extra
    return event


class TestLaunchTimeSummaryScripts(CustomTestCase):
    def test_pe7b_unconstrained_and_constrained_commands(self):
        preset = pe7b.get_preset(None)
        unconstrained = pe7b.PE7B_SETUPS["unconstrained"]
        constrained = pe7b.PE7B_SETUPS["constrained_4096"]

        unconstrained_cmd = pe7b.launch_time.build_promptenhancer_command(
            preset=preset,
            tp_size=2,
            host="127.0.0.1",
            port=30000,
            llm_setup=unconstrained,
        )
        constrained_cmd = pe7b.launch_time.build_promptenhancer_command(
            preset=preset,
            tp_size=2,
            host="127.0.0.1",
            port=30000,
            llm_setup=constrained,
        )

        self.assertNotIn("--max-running-requests", unconstrained_cmd)
        self.assertNotIn("--max-total-tokens", unconstrained_cmd)
        self.assertNotIn("--chunked-prefill-size", unconstrained_cmd)
        self.assertIn("--context-length", unconstrained_cmd)
        self.assertIn("--mem-fraction-static", unconstrained_cmd)
        self.assertEqual(
            unconstrained_cmd[unconstrained_cmd.index("--served-model-name") + 1],
            "Hunyuan_PromptEnhancer_7B",
        )

        self.assertEqual(
            constrained_cmd[constrained_cmd.index("--max-running-requests") + 1],
            "1",
        )
        self.assertEqual(
            constrained_cmd[constrained_cmd.index("--max-total-tokens") + 1],
            "4096",
        )
        self.assertEqual(
            constrained_cmd[constrained_cmd.index("--chunked-prefill-size") + 1],
            "4096",
        )
        self.assertEqual(
            constrained_cmd[constrained_cmd.index("--cuda-graph-max-bs") + 1],
            "1",
        )
        self.assertEqual(
            constrained_cmd[constrained_cmd.index("--served-model-name") + 1],
            "Hunyuan_PromptEnhancer_7B",
        )

    def test_pe7b_csv_row_extracts_launch_tasks_and_params(self):
        preset = pe7b.get_preset(None)
        setup = pe7b.PE7B_SETUPS["constrained_4096"]
        record = {
            "status": "completed",
            "launch_time_s": 30.0,
            "tasks": {
                "load_weight": {"observed": True, "elapsed_s_max": 10.25},
                "cuda_graph_capture": {"observed": True, "elapsed_s_max": 3.5},
            },
        }
        row = pe7b.build_csv_row(record, preset=preset, setup=setup, gpu_num=4)
        self.assertEqual(row["row_name"], "tp4_constrained_4096")
        self.assertEqual(row["e2e_launch_time_s"], "30.000000")
        self.assertEqual(row["weight_load_time_s"], "10.250000")
        self.assertEqual(row["cuda_graph_capture_time_s"], "3.500000")
        self.assertEqual(row["max_total_tokens"], "4096")

    def test_pe7b_execution_server_and_bench_commands_are_constrained(self):
        preset = pe7b_exec.get_preset(None)
        server_cmd = pe7b_exec.build_server_command(
            preset=preset,
            tp_size=4,
            host="127.0.0.1",
            port=30000,
        )
        self.assertEqual(
            server_cmd[server_cmd.index("--served-model-name") + 1],
            "Hunyuan_PromptEnhancer_7B",
        )
        self.assertEqual(
            server_cmd[server_cmd.index("--max-running-requests") + 1], "1"
        )
        self.assertEqual(
            server_cmd[server_cmd.index("--max-total-tokens") + 1], "4096"
        )
        self.assertEqual(
            server_cmd[server_cmd.index("--chunked-prefill-size") + 1], "4096"
        )
        self.assertEqual(
            server_cmd[server_cmd.index("--cuda-graph-max-bs") + 1], "1"
        )

        bench_cmd = pe7b_exec.build_bench_command(
            preset=preset,
            host="127.0.0.1",
            port=30000,
            input_len=128,
            output_len=384,
            output_file=repo_root() / "bench.jsonl",
            run_tag="unit",
        )
        self.assertEqual(bench_cmd[bench_cmd.index("--model") + 1], preset.model_path)
        self.assertEqual(
            bench_cmd[bench_cmd.index("--served-model-name") + 1], preset.model_id
        )
        self.assertEqual(
            bench_cmd[bench_cmd.index("--tokenizer") + 1], preset.model_path
        )
        self.assertEqual(bench_cmd[bench_cmd.index("--num-prompts") + 1], "1")
        self.assertEqual(bench_cmd[bench_cmd.index("--max-concurrency") + 1], "1")
        self.assertEqual(bench_cmd[bench_cmd.index("--warmup-requests") + 1], "0")

    def test_pe7b_execution_default_io_matrix_matches_reprefill_cases(self):
        cases = pe7b_exec.default_io_cases()
        self.assertEqual(len(cases), 119)
        self.assertEqual(cases[0], (128, 384))
        self.assertEqual(cases[-1], (2048, 128))
        self.assertEqual(
            pe7b_exec.DEFAULT_OUTPUT_LENS[1],
            [256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 1920],
        )
        self.assertIn((256, 1920), cases)
        self.assertIn((1792, 384), cases)
        self.assertNotIn((256, 128), cases)
        self.assertNotIn((1792, 128), cases)

    def test_pe7b_execution_resolves_custom_and_legacy_io_cases(self):
        default_args = pe7b_exec.parse_args(["--gpu-nums", "1"])
        self.assertEqual(default_args.io_cases, pe7b_exec.default_io_cases())

        matrix_args = pe7b_exec.parse_args(
            ["--io-matrix", "128:384,512", "256:256"]
        )
        self.assertEqual(
            matrix_args.io_cases,
            [(128, 384), (128, 512), (256, 256)],
        )

        flat_args = pe7b_exec.parse_args(
            ["--input-lens", "128", "256", "--output-lens", "384", "512"]
        )
        self.assertEqual(
            flat_args.io_cases,
            [(128, 384), (128, 512), (256, 384), (256, 512)],
        )

        with self.assertRaises(SystemExit):
            pe7b_exec.parse_args(["--io-matrix", "128:384", "--input-lens", "128"])

    def test_pe7b_execution_averages_only_measured_runs(self):
        runs = [
            pe7b_exec.BenchRun(
                run_index=1,
                is_warmup=True,
                status="ok",
                reason="",
                returncode=0,
                elapsed_s=1.0,
                ttft_ms=1000.0,
                tpot_ms=1000.0,
                bench_jsonl_path="warmup1.jsonl",
                client_log_path="warmup1.log",
                command=[],
            ),
            pe7b_exec.BenchRun(
                run_index=2,
                is_warmup=True,
                status="ok",
                reason="",
                returncode=0,
                elapsed_s=1.0,
                ttft_ms=1000.0,
                tpot_ms=1000.0,
                bench_jsonl_path="warmup2.jsonl",
                client_log_path="warmup2.log",
                command=[],
            ),
            pe7b_exec.BenchRun(
                run_index=3,
                is_warmup=False,
                status="ok",
                reason="",
                returncode=0,
                elapsed_s=1.0,
                ttft_ms=10.0,
                tpot_ms=1.0,
                bench_jsonl_path="measure3.jsonl",
                client_log_path="measure3.log",
                command=[],
            ),
            pe7b_exec.BenchRun(
                run_index=4,
                is_warmup=False,
                status="ok",
                reason="",
                returncode=0,
                elapsed_s=1.0,
                ttft_ms=20.0,
                tpot_ms=2.0,
                bench_jsonl_path="measure4.jsonl",
                client_log_path="measure4.log",
                command=[],
            ),
            pe7b_exec.BenchRun(
                run_index=5,
                is_warmup=False,
                status="ok",
                reason="",
                returncode=0,
                elapsed_s=1.0,
                ttft_ms=30.0,
                tpot_ms=3.0,
                bench_jsonl_path="measure5.jsonl",
                client_log_path="measure5.log",
                command=[],
            ),
        ]
        result = pe7b_exec.aggregate_case(
            tp_size=2,
            input_len=128,
            output_len=384,
            case_dir=repo_root() / "case",
            server_log_path=repo_root() / "server.log",
            server_command=[],
            run_results=runs,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.to_csv_row()["row_name"], "tp2_in128_out384")
        self.assertEqual(result.to_csv_row()["ttft_ms"], "20.000000")
        self.assertEqual(result.to_csv_row()["tpot_ms"], "2.000000")

    def test_pe7b_execution_parses_bench_jsonl(self):
        class FakeJsonlPath:
            def exists(self):
                return True

            def read_text(self, encoding):
                return json.dumps({"mean_ttft_ms": 12.5, "mean_tpot_ms": 3.25}) + "\n"

        ttft, tpot, reason = pe7b_exec.parse_bench_jsonl(FakeJsonlPath())
        self.assertEqual(ttft, 12.5)
        self.assertEqual(tpot, 3.25)
        self.assertEqual(reason, "")

    def test_diffusion_presets_use_registry_short_model_ids(self):
        self.assertEqual(
            diffusion.launch_time.DIFFUSION_REGISTRY_MODEL_IDS,
            EXPECTED_DIFFUSION_MODEL_IDS,
        )
        for model_key, expected_model_id in EXPECTED_DIFFUSION_MODEL_IDS.items():
            with self.subTest(model_key=model_key):
                self.assertEqual(
                    diffusion.launch_time.PRESETS[model_key].model_id,
                    expected_model_id,
                )
                self.assertEqual(
                    diffusion.get_preset(model_key, None).model_id,
                    expected_model_id,
                )
                self.assertEqual(
                    stage_breakdown.MODEL_PRESETS[model_key].model_id,
                    expected_model_id,
                )

    def test_diffusion_reduced_parallelism_policy_is_tp1_sp_gpu(self):
        expected = {
            ("wan2.2-ti2v-5b", 8): (1, 8, 8, 1),
            ("wan2.1-t2v-1.3b", 8): (1, 8, 4, 2),
            ("wan2.1-t2v-1.3b", 4): (1, 4, 4, 1),
            ("z-image", 8): (1, 8, 2, 4),
            ("z-image", 1): (1, 1, 1, 1),
        }
        for (model, gpu_num), values in expected.items():
            with self.subTest(model=model, gpu_num=gpu_num):
                run_config = diffusion.resolve_diffusion_run_config(model, gpu_num)
                diffusion.validate_run_config(run_config, gpu_num)
                self.assertEqual(
                    (
                        run_config.tp_size,
                        run_config.sp_degree,
                        run_config.ulysses_degree,
                        run_config.ring_degree,
                    ),
                    values,
                )

    def test_stage_breakdown_reduced_parallelism_policy(self):
        expected = {
            ("wan2.2-ti2v-5b", 8): (8, 8, 1),
            ("wan2.1-t2v-1.3b", 8): (8, 4, 2),
            ("wan2.1-t2v-1.3b", 4): (4, 4, 1),
            ("z-image", 8): (8, 2, 4),
            ("z-image", 1): (1, 1, 1),
        }
        for (model, gpu_num), values in expected.items():
            with self.subTest(model=model, gpu_num=gpu_num):
                parallel = stage_breakdown.resolve_parallelism(
                    model_name=model,
                    gpu_num=gpu_num,
                    ulysses_degree=None,
                    ring_degree=None,
                )
                self.assertTrue(parallel.valid)
                self.assertEqual(
                    (
                        parallel.sp_degree,
                        parallel.ulysses_degree,
                        parallel.ring_degree,
                    ),
                    values,
                )

    def test_diffusion_command_forces_tp1_sp_gpu_and_disables_offload_fsdp(self):
        preset = diffusion.get_preset("z-image", None)
        run_config = diffusion.resolve_diffusion_run_config("z-image", 4)
        command = diffusion.build_command(
            preset=preset,
            run_config=run_config,
            host="127.0.0.1",
            case_dir=repo_root() / "_dry_run_case_for_test",
        )

        self.assertEqual(diffusion.command_value(command, "--model-id"), "Z-Image")
        self.assertEqual(diffusion.command_value(command, "--tp-size"), "1")
        self.assertEqual(diffusion.command_value(command, "--sp-degree"), "4")
        self.assertEqual(diffusion.command_value(command, "--warmup"), "false")
        for flag in (
            "--dit-cpu-offload",
            "--dit-layerwise-offload",
            "--text-encoder-cpu-offload",
            "--image-encoder-cpu-offload",
            "--vae-cpu-offload",
            "--pin-cpu-memory",
            "--use-fsdp-inference",
        ):
            self.assertEqual(diffusion.command_value(command, flag), "false")

    def test_stage_breakdown_command_and_csv_use_role_duration_columns(self):
        preset = stage_breakdown.MODEL_PRESETS["z-image"]
        parallel = stage_breakdown.resolve_parallelism(
            model_name="z-image",
            gpu_num=4,
            ulysses_degree=None,
            ring_degree=None,
        )
        command = stage_breakdown.build_generate_command(
            preset=preset,
            model_path=preset.hf_path,
            prompt=preset.default_prompt,
            image_paths=None,
            gpu_num=4,
            width=1024,
            height=1024,
            parallel=parallel,
            case_dir=repo_root() / "_stage_case_for_test",
            base_gpu_id=None,
        )
        self.assertEqual(command[command.index("--model-id") + 1], "Z-Image")
        self.assertEqual(command[command.index("--tp-size") + 1], "1")
        self.assertEqual(command[command.index("--sp-degree") + 1], "4")
        self.assertEqual(command[command.index("--ulysses-degree") + 1], "2")
        self.assertEqual(command[command.index("--ring-degree") + 1], "2")
        self.assertEqual(command[command.index("--use-fsdp-inference") + 1], "false")

        result = stage_breakdown.CaseResult(
            row_name="4_144p",
            model="z-image",
            model_path=preset.hf_path,
            gpu_num=4,
            resolution="144p",
            width=256,
            height=144,
            ulysses_degree=2,
            ring_degree=2,
            status="ok",
            returncode=0,
            elapsed_s=1.0,
            durations_s={"text_encoder": 1.0, "denoiser": 2.0, "decoder": 3.0},
            perf_path="perf.json",
            case_dir="case",
            error_tail="",
            command=[],
        )
        self.assertEqual(
            result.to_csv_row(),
            {
                "row_name": "4_144p",
                "text_encoder_duration_s": "1.000000",
                "denoiser_duration_s": "2.000000",
                "decoder_duration_s": "3.000000",
            },
        )

    def test_stage_breakdown_perf_parser_uses_denoise_fallback(self):
        durations = stage_breakdown.extract_stage_durations(
            {
                "steps": [
                    {"name": "TextEncodingStage", "duration_ms": 1000},
                    {"name": "DecodingStage", "duration_ms": 3000},
                ],
                "denoise_steps_ms": [500, {"duration_ms": 1500}],
            }
        )
        self.assertEqual(durations["text_encoder"], 1.0)
        self.assertEqual(durations["denoiser"], 2.0)
        self.assertEqual(durations["decoder"], 3.0)

    def test_diffusion_role_csv_row_aggregates_role_events(self):
        events = [
            end_event("component_load", 0.5, "text_encoder"),
            end_event("component_load", 0.1, "tokenizer"),
            end_event("component_load", 2.0, "transformer"),
            end_event("component_load", 1.5, "transformer_2"),
            end_event("component_load", 0.3, "vae"),
            end_event("component_load", 0.2, "scheduler"),
            end_event("component_cpu_materialization", 0.4, "text_encoder"),
            end_event("component_cpu_materialization", 0.05, "tokenizer"),
            end_event("component_cpu_materialization", 1.2, "transformer"),
            end_event("component_cpu_materialization", 0.8, "transformer_2"),
            end_event("component_cpu_materialization", 0.25, "vae"),
            end_event(
                "component_load_stats",
                0.0,
                "text_encoder",
                {"loaded_weight_file_size_gb": 1.0},
            ),
            end_event(
                "component_load_stats",
                0.0,
                "transformer",
                {"loaded_weight_file_size_gb": 2.0},
            ),
            end_event(
                "component_load_stats",
                0.0,
                "transformer_2",
                {"loaded_weight_file_size_gb": 3.0},
            ),
            end_event(
                "component_load_stats",
                0.0,
                "vae",
                {"loaded_weight_file_size_gb": 0.5},
            ),
            end_event("role_launch_total", 0.6, "encoder"),
            end_event("role_launch_total", 3.5, "denoiser"),
            end_event("role_launch_total", 0.3, "decoder"),
            end_event("role_launch_total", 0.2, "shared"),
        ]
        record = {
            "status": "completed",
            "launch_time_s": 9.0,
            "tasks": breakdown.summarize_launch_tasks(
                events, family="sglang-diffusion"
            ),
        }
        run_config = diffusion.resolve_diffusion_run_config("z-image", 4)
        row = diffusion.build_csv_row(
            record, model_key="z-image", run_config=run_config
        )
        detail = diffusion.build_detail_row(
            record, model_key="z-image", run_config=run_config
        )

        self.assertEqual(row["text_encoder_e2e_launch_time_s"], "0.600000")
        self.assertEqual(row["denoiser_e2e_launch_time_s"], "3.500000")
        self.assertEqual(row["decoder_e2e_launch_time_s"], "0.300000")
        self.assertEqual(row["text_encoder_weight_load_time_s"], "0.600000")
        self.assertEqual(row["denoiser_weight_load_time_s"], "3.500000")
        self.assertEqual(row["decoder_weight_load_time_s"], "0.300000")
        self.assertEqual(row["text_encoder_cpu_materialization_time_s"], "0.450000")
        self.assertEqual(row["denoiser_cpu_materialization_time_s"], "2.000000")
        self.assertEqual(row["decoder_cpu_materialization_time_s"], "0.250000")
        self.assertEqual(row["denoiser_weight_size_gb"], "5.000000")
        self.assertEqual(detail["shared_component_launch_time_s"], "0.200000")


if __name__ == "__main__":
    unittest.main(verbosity=3)
