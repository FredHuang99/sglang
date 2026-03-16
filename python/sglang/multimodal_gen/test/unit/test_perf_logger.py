# SPDX-License-Identifier: Apache-2.0
"""Unit tests for diffusion performance logging helpers."""

import os
import unittest
from unittest import mock

from sglang.multimodal_gen.runtime.utils.perf_logger import (
    PerformanceLogger,
    RequestTimings,
    get_sync_stage_profiling_mode,
    should_sync_stage_profiling,
)


class TestStageProfilingSyncMode(unittest.TestCase):
    """Test environment parsing for stage profiling synchronization."""

    def tearDown(self) -> None:
        get_sync_stage_profiling_mode.cache_clear()

    def test_sync_mode_defaults_to_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            get_sync_stage_profiling_mode.cache_clear()
            self.assertEqual(get_sync_stage_profiling_mode(), "off")

    def test_sync_mode_supports_all(self):
        with mock.patch.dict(
            os.environ,
            {"SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "all"},
            clear=True,
        ):
            get_sync_stage_profiling_mode.cache_clear()
            self.assertEqual(get_sync_stage_profiling_mode(), "all")

    def test_sync_mode_supports_backward_compatible_one(self):
        with mock.patch.dict(
            os.environ,
            {"SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "1"},
            clear=True,
        ):
            get_sync_stage_profiling_mode.cache_clear()
            self.assertEqual(get_sync_stage_profiling_mode(), "denoising")

    def test_should_sync_stage_profiling_for_all(self):
        with mock.patch(
            "sglang.multimodal_gen.runtime.utils.perf_logger.torch.cuda.is_available",
            return_value=True,
        ):
            with mock.patch.dict(
                os.environ,
                {"SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "all"},
                clear=True,
            ):
                get_sync_stage_profiling_mode.cache_clear()
                self.assertTrue(should_sync_stage_profiling("TextEncodingStage"))
                self.assertTrue(should_sync_stage_profiling("denoising_step_0"))

    def test_should_sync_stage_profiling_for_denoising_only(self):
        with mock.patch(
            "sglang.multimodal_gen.runtime.utils.perf_logger.torch.cuda.is_available",
            return_value=True,
        ):
            with mock.patch.dict(
                os.environ,
                {"SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "denoising"},
                clear=True,
            ):
                get_sync_stage_profiling_mode.cache_clear()
                self.assertTrue(should_sync_stage_profiling("denoising_step_3"))
                self.assertFalse(should_sync_stage_profiling("TextEncodingStage"))


class TestBenchmarkReport(unittest.TestCase):
    """Test JSON and text benchmark report helpers."""

    def make_timings(self) -> RequestTimings:
        timings = RequestTimings(request_id="req-1")
        timings.stages = {
            "TextEncodingStage": 12.5,
            "DenoisingStage": 34.0,
        }
        timings.steps = [4.0, 5.0, 6.0]
        timings.total_duration_ms = 80.0
        return timings

    def test_build_benchmark_report_contains_stage_alias(self):
        timings = self.make_timings()
        with mock.patch(
            "sglang.multimodal_gen.runtime.utils.perf_logger.get_git_commit_hash",
            return_value="deadbeef",
        ):
            report = PerformanceLogger.build_benchmark_report(
                timings=timings,
                meta={"model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
                tag="unit_test",
            )

        self.assertEqual(report["request_id"], "req-1")
        self.assertEqual(report["tag"], "unit_test")
        self.assertEqual(report["total_duration_ms"], 80.0)
        self.assertEqual(report["steps"], report["stages_ms"])
        self.assertEqual(report["stages_ms"][0]["name"], "TextEncodingStage")
        self.assertEqual(report["denoise_steps_ms"][1]["duration_ms"], 5.0)

    def test_format_benchmark_summary_includes_stages_and_denoise_stats(self):
        timings = self.make_timings()
        summary = PerformanceLogger.format_benchmark_summary(
            timings=timings,
            meta={
                "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "prompt": "A calm lake with mountains in the background",
            },
        )

        self.assertIn("Performance summary:", summary)
        self.assertIn("model: Wan-AI/Wan2.2-TI2V-5B-Diffusers", summary)
        self.assertIn("TextEncodingStage: 12.50 ms", summary)
        self.assertIn("DenoisingStage: 34.00 ms", summary)
        self.assertIn("count=3, avg=5.00 ms, min=4.00 ms, max=6.00 ms", summary)


if __name__ == "__main__":
    unittest.main()
