import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def repo_root():
    return Path(__file__).resolve().parents[3]


def load_profile_module():
    root = repo_root()
    module_path = (
        root
        / "python"
        / "sglang"
        / "multimodal_gen"
        / "benchmarks"
        / "profile_stage_breakdown.py"
    )
    return load_module("profile_stage_breakdown_for_test", module_path)


def load_register_cpu_ci():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "python"
        / "sglang"
        / "test"
        / "ci"
        / "ci_register.py"
    )
    return load_module("ci_register_for_profile_stage_breakdown_test", module_path).register_cpu_ci


register_cpu_ci = load_register_cpu_ci()
register_cpu_ci(est_time=2, suite="stage-a-cpu-only")
psb = load_profile_module()


def make_case_result(row_name, durations_s, status="ok"):
    return psb.CaseResult(
        row_name=row_name,
        model="z-image",
        model_path="Tongyi-MAI/Z-Image",
        gpu_num=1,
        resolution="144p",
        width=256,
        height=144,
        ulysses_degree=1,
        ring_degree=1,
        status=status,
        returncode=0,
        elapsed_s=1.0,
        durations_s=durations_s,
        perf_path=f"/tmp/{row_name}.json",
        case_dir=f"/tmp/{row_name}",
        error_tail="",
        command=["python", "-m", "sglang"],
    )


class TestDiffusionProfileStageBreakdown(unittest.TestCase):
    def test_default_repeat_args(self):
        args = psb.parse_args(["--model", "z-image"])
        self.assertEqual(args.num_runs, 5)
        self.assertEqual(args.num_warmup_runs, 2)

    def test_resolution_mapping(self):
        self.assertEqual(psb.resolve_resolution("wan2.2-ti2v-5b", "720p"), (1280, 704))
        self.assertEqual(psb.resolve_resolution("wan2.2-ti2v-5b", "144p"), (256, 128))
        self.assertEqual(psb.resolve_resolution("wan2.1-t2v-1.3b", "360p"), (640, 352))
        self.assertEqual(psb.resolve_resolution("z-image", "240p"), (432, 240))

    def test_parallelism_auto_and_head_conflict_fallback(self):
        parallel = psb.resolve_parallelism(
            gpu_num=8,
            attention_heads=40,
            ulysses_degree=None,
            ring_degree=None,
        )
        self.assertTrue(parallel.valid)
        self.assertEqual(parallel.ulysses_degree, 8)
        self.assertEqual(parallel.ring_degree, 1)

        parallel = psb.resolve_parallelism(
            gpu_num=8,
            attention_heads=30,
            ulysses_degree=None,
            ring_degree=None,
        )
        self.assertTrue(parallel.valid)
        self.assertEqual(parallel.ulysses_degree, 2)
        self.assertEqual(parallel.ring_degree, 4)
        self.assertTrue(parallel.is_fallback)

    def test_parallelism_user_values_are_validated_by_product(self):
        parallel = psb.resolve_parallelism(
            gpu_num=4,
            attention_heads=30,
            ulysses_degree=3,
            ring_degree=1,
        )
        self.assertFalse(parallel.valid)
        self.assertIn("sp_degree", parallel.error)

        parallel = psb.resolve_parallelism(
            gpu_num=4,
            attention_heads=30,
            ulysses_degree=2,
            ring_degree=None,
        )
        self.assertTrue(parallel.valid)
        self.assertEqual(parallel.ulysses_degree, 2)
        self.assertEqual(parallel.ring_degree, 2)

    def test_wan22_default_image_path(self):
        preset = psb.MODEL_PRESETS["wan2.2-ti2v-5b"]
        self.assertEqual(
            psb.resolve_image_paths(preset, None),
            [
                str(
                    psb.repo_root_from_file()
                    / psb.WAN22_DEFAULT_IMAGE_RELATIVE_PATH
                )
            ],
        )
        self.assertEqual(
            psb.resolve_image_paths(preset, ["C:\\custom\\image.png"]),
            ["C:\\custom\\image.png"],
        )
        self.assertIsNone(psb.resolve_image_paths(psb.MODEL_PRESETS["z-image"], None))

    def test_extract_stage_durations(self):
        report = {
            "steps": [
                {"name": "TextEncodingStage", "duration_ms": 1200.0},
                {"name": "DenoisingStage", "duration_ms": 3456.0},
                {"name": "DecodingStage", "duration_ms": 789.0},
            ]
        }
        durations = psb.extract_stage_durations(report)
        self.assertEqual(durations["text_encoder"], 1.2)
        self.assertEqual(durations["denoising"], 3.456)
        self.assertEqual(durations["decoder"], 0.789)

    def test_extract_stage_durations_denoise_step_fallback(self):
        report = {
            "steps": [
                {"name": "TextEncodingStage", "duration_ms": 100.0},
                {"name": "DecodingStage", "duration_ms": 300.0},
            ],
            "denoise_steps_ms": [
                {"step": 0, "duration_ms": 10.0},
                {"step": 1, "duration_ms": 20.0},
            ],
        }
        durations = psb.extract_stage_durations(report)
        self.assertEqual(durations["text_encoder"], 0.1)
        self.assertEqual(durations["denoising"], 0.03)
        self.assertEqual(durations["decoder"], 0.3)

    def test_average_measured_durations_excludes_warmup(self):
        run_results = [
            make_case_result(
                "1_144p",
                {"text_encoder": 100.0, "denoising": 100.0, "decoder": 100.0},
            ),
            make_case_result(
                "1_144p",
                {"text_encoder": 200.0, "denoising": 200.0, "decoder": 200.0},
            ),
            make_case_result(
                "1_144p",
                {"text_encoder": 1.0, "denoising": 10.0, "decoder": 100.0},
            ),
            make_case_result(
                "1_144p",
                {"text_encoder": 2.0, "denoising": 20.0, "decoder": 200.0},
            ),
            make_case_result(
                "1_144p",
                {"text_encoder": 3.0, "denoising": 30.0, "decoder": 300.0},
            ),
        ]
        durations = psb.average_measured_durations(
            run_results,
            num_warmup_runs=2,
        )
        self.assertEqual(durations["text_encoder"], 2.0)
        self.assertEqual(durations["denoising"], 20.0)
        self.assertEqual(durations["decoder"], 200.0)

    def test_write_csv_failure_row(self):
        with tempfile.TemporaryDirectory(dir=repo_root()) as temp_dir:
            output_dir = Path(temp_dir)
            result = psb.make_failure_result(
                row_name="8_2k",
                preset=psb.MODEL_PRESETS["z-image"],
                model_path="Tongyi-MAI/Z-Image",
                gpu_num=8,
                resolution="2k",
                width=2048,
                height=1152,
                parallel=psb.fallback_parallelism(8),
                status="failed",
                case_dir=output_dir / "case",
                error_tail="boom",
            )
            csv_path = output_dir / "stage.csv"
            details_csv_path = output_dir / "details.csv"
            details_json_path = output_dir / "details.json"
            psb.write_csv_outputs(
                [result],
                csv_path=csv_path,
                details_csv_path=details_csv_path,
                details_json_path=details_json_path,
            )
            self.assertIn("8_2k,nan,nan,nan", csv_path.read_text(encoding="utf-8"))
            self.assertIn('"status": "failed"', details_json_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=3)
