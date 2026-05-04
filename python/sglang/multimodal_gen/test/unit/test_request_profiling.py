# SPDX-License-Identifier: Apache-2.0

import asyncio
import csv
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.benchmarks.wan_ti2v_profile import (
    BenchmarkRequest,
    apply_request_overrides,
    build_parser,
    build_default_request_spec,
    build_submission_schedule,
    build_warmup_requests,
    detect_profile_preset,
    disable_cfg_for_request,
    _row_from_task_result,
    _submit_and_poll_video_request,
    summarize_profile_run,
)
from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    RequestTracker,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics
from sglang.multimodal_gen.runtime.utils.request_profiling import (
    CsvProfileWriter,
    RequestCsvProfiler,
    aggregate_logical_stage_durations,
    flatten_request_metrics,
    resolve_profile_dir,
)


class _ProfiledNoopStage(PipelineStage):
    def __init__(self):
        self.server_args = SimpleNamespace(comfyui_mode=False)

    def forward(self, batch, server_args):
        time.sleep(0.001)
        return batch


class TestRequestProfilingUtils(unittest.TestCase):
    def test_csv_writer_adds_new_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "profile.csv")
            writer = CsvProfileWriter(file_path)
            writer.write_row({"request_id": "r1", "encoder_ms": 1.0})
            writer.write_row({"request_id": "r2", "decoder_ms": 2.0})

            with open(file_path, "r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))

            self.assertEqual(len(rows), 2)
            self.assertIn("encoder_ms", rows[0])
            self.assertIn("decoder_ms", rows[0])
            self.assertEqual(rows[0]["encoder_ms"], "1.0")
            self.assertEqual(rows[1]["decoder_ms"], "2.0")

    def test_request_profiler_finalized_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "profile.csv")
            profiler = RequestCsvProfiler(file_path, finalized_cache_size=2)

            profiler.finalize("r1", status="completed")
            profiler.finalize("r2", status="completed")
            profiler.finalize("r3", status="completed")

            self.assertEqual(list(profiler._finalized_request_ids), ["r2", "r3"])

    def test_logical_stage_aggregation(self):
        metrics = RequestMetrics(request_id="req-1")
        metrics.total_duration_ms = 100.0
        metrics.stages = {
            "text_encoder": 20.0,
            "ImageVAEEncodingStage": 5.0,
            "denoising": 50.0,
            "vae_decode": 10.0,
        }
        logical = aggregate_logical_stage_durations(metrics)
        self.assertEqual(logical["denoiser"], 50.0)
        self.assertEqual(logical["decoder"], 10.0)
        self.assertEqual(logical["encoder"], 25.0)

    def test_logical_stage_aggregation_omits_unobserved_stages(self):
        metrics = RequestMetrics(request_id="req-empty")

        logical = aggregate_logical_stage_durations(metrics)
        row = flatten_request_metrics(metrics)

        self.assertEqual(logical, {})
        self.assertNotIn("logical_encoder_duration_ms", row)
        self.assertNotIn("logical_denoiser_duration_ms", row)
        self.assertNotIn("logical_decoder_duration_ms", row)

    def test_profile_enabled_records_stage_timing_without_perf_dump_path(self):
        metrics = RequestMetrics(request_id="req-profile")
        batch = SimpleNamespace(
            is_warmup=False,
            perf_dump_path=None,
            metrics=metrics,
        )
        server_args = SimpleNamespace(profile_enabled=True, comfyui_mode=False)

        _ProfiledNoopStage()(batch, server_args)

        self.assertIn("_ProfiledNoopStage", metrics.stages)
        self.assertGreater(metrics.stages["_ProfiledNoopStage"], 0.0)
        logical = aggregate_logical_stage_durations(metrics)
        self.assertGreater(logical["encoder"], 0.0)

    def test_flatten_request_metrics_exposes_queue_and_unattributed_time(self):
        metrics = RequestMetrics(request_id="req-queue")
        metrics.arrival_time_s = 10.0
        metrics.start_time_s = 12.5
        metrics.finish_time_s = 15.0
        metrics.total_duration_ms = 120.0
        metrics.stages = {
            "InputValidationStage": 5.0,
            "DenoisingStage": 80.0,
            "DecodingStage": 20.0,
        }

        row = flatten_request_metrics(metrics)

        self.assertEqual(row["queue_duration_ms"], 2500.0)
        self.assertEqual(row["e2e_duration_ms"], 5000.0)
        self.assertEqual(row["logical_encoder_duration_ms"], 5.0)
        self.assertEqual(row["logical_denoiser_duration_ms"], 80.0)
        self.assertEqual(row["logical_decoder_duration_ms"], 20.0)
        self.assertEqual(row["unattributed_duration_ms"], 15.0)

    def test_request_tracker_can_preserve_external_arrival_time(self):
        tracker = RequestTracker()
        record = tracker.submit("req-disagg", submit_time_s=123.5)

        self.assertEqual(record.submit_time_s, 123.5)
        self.assertEqual(record.last_transition_time_s, 123.5)
        self.assertEqual(record.state_timestamps[RequestState.PENDING.value], 123.5)


class _FakeAsyncResponse:
    def __init__(self, payload=None, *, status=200, enter_error=None):
        self._payload = payload or {}
        self.status = status
        self._enter_error = enter_error

    async def __aenter__(self):
        if self._enter_error is not None:
            raise self._enter_error
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self):
        return self._payload


class _FakeVideoSession:
    def __init__(self, *, post_payload, get_results):
        self._post_payload = post_payload
        self._get_results = list(get_results)
        self.get_count = 0

    def post(self, *_args, **_kwargs):
        return _FakeAsyncResponse(self._post_payload)

    def get(self, *_args, **_kwargs):
        self.get_count += 1
        result = self._get_results.pop(0)
        if isinstance(result, BaseException):
            return _FakeAsyncResponse(enter_error=result)
        return _FakeAsyncResponse(result)


async def _no_sleep(_delay):
    return None


class TestWanTi2vBenchmarkHelpers(unittest.TestCase):
    def test_detect_profile_preset_auto_uses_model_name(self):
        self.assertEqual(
            detect_profile_preset("auto", model_override="Tongyi-MAI/Z-Image-Turbo"),
            "z_image",
        )
        self.assertEqual(
            detect_profile_preset(
                "auto", request_model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            ),
            "wan2_1_t2v_1_3b",
        )
        self.assertEqual(detect_profile_preset("auto"), "wan2_2_ti2v_5b")

    def test_build_default_request_spec_for_zimage(self):
        request_spec = build_default_request_spec("z_image")
        self.assertEqual(request_spec.endpoint_kind, "image")
        self.assertEqual(request_spec.payload["model"], "Z-Image")
        self.assertEqual(request_spec.payload["num_inference_steps"], 50)
        self.assertEqual(request_spec.payload["guidance_scale"], 5.0)
        self.assertEqual(request_spec.payload["response_format"], "url")
        self.assertNotIn("size", request_spec.payload)

    def test_build_default_request_spec_for_zimage_turbo_override(self):
        request_spec = build_default_request_spec(
            "z_image",
            model_override="Z-Image-Turbo",
        )
        self.assertEqual(request_spec.endpoint_kind, "image")
        self.assertEqual(request_spec.payload["num_inference_steps"], 9)
        self.assertEqual(request_spec.payload["guidance_scale"], 0.0)

    def test_apply_request_overrides_ignores_video_only_fields_for_image(self):
        payload = apply_request_overrides(
            {"model": "Tongyi-MAI/Z-Image-Turbo", "prompt": "base"},
            endpoint_kind="image",
            input_reference="/tmp/example.png",
            fps=16,
            num_frames=81,
            size="1024x1024",
        )
        self.assertEqual(payload["size"], "1024x1024")
        self.assertNotIn("input_reference", payload)
        self.assertNotIn("fps", payload)
        self.assertNotIn("num_frames", payload)

    def test_disable_cfg_for_video_request_clamps_guidance_and_clears_negative_prompt(self):
        payload = disable_cfg_for_request(
            {
                "model": "Wan2.2-TI2V-5B-Diffusers",
                "guidance_scale": 5.0,
                "guidance_scale_2": 3.5,
                "negative_prompt": "bad anatomy",
                "cfg_normalization": 1.0,
            },
            endpoint_kind="video",
        )
        self.assertEqual(payload["guidance_scale"], 1.0)
        self.assertEqual(payload["guidance_scale_2"], 1.0)
        self.assertIsNone(payload["negative_prompt"])
        self.assertEqual(payload["cfg_normalization"], 0.0)

    def test_disable_cfg_for_image_request_preserves_already_disabled_scales(self):
        payload = disable_cfg_for_request(
            {
                "model": "Tongyi-MAI/Z-Image-Turbo",
                "guidance_scale": 0.0,
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
            },
            endpoint_kind="image",
        )
        self.assertEqual(payload["guidance_scale"], 0.0)
        self.assertEqual(payload["true_cfg_scale"], 1.0)
        self.assertIsNone(payload["negative_prompt"])

    def test_build_submission_schedule(self):
        self.assertEqual(
            build_submission_schedule(
                num_requests=3,
                traffic_mode="burst",
                requests_per_minute=24,
                start_time_s=10.0,
            ),
            [10.0, 10.0, 10.0],
        )
        self.assertEqual(
            build_submission_schedule(
                num_requests=3,
                traffic_mode="normal",
                requests_per_minute=30,
                start_time_s=10.0,
            ),
            [10.0, 12.0, 14.0],
        )

    def test_build_warmup_requests_reuses_payload_without_client_schedule(self):
        payload = {"model": "Wan2.2-TI2V-5B-Diffusers", "size": "1280x704"}
        requests = build_warmup_requests(
            payload=payload,
            num_warmup_requests=3,
            start_time_s=12.5,
        )
        self.assertEqual([request.index for request in requests], [-1, -2, -3])
        self.assertEqual(
            [request.scheduled_submit_time_s for request in requests],
            [12.5, 12.5, 12.5],
        )
        self.assertEqual(requests[0].payload, payload)
        self.assertIsNot(requests[0].payload, payload)

    def test_parser_defaults_to_three_warmup_requests(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--deployment-mode", "monolithic", "--traffic-mode", "burst"]
        )
        self.assertEqual(args.num_warmup_requests, 3)

    def test_video_poll_retries_transient_connection_reset(self):
        request = BenchmarkRequest(
            index=0,
            scheduled_submit_time_s=0.0,
            payload={"prompt": "hello"},
        )
        session = _FakeVideoSession(
            post_payload={"id": "video-1", "status": "queued"},
            get_results=[
                OSError("connection reset"),
                {"id": "video-1", "status": "completed", "file_path": "out.mp4"},
            ],
        )

        with mock.patch(
            "sglang.multimodal_gen.benchmarks.wan_ti2v_profile.asyncio.sleep",
            new=_no_sleep,
        ):
            row = asyncio.run(
                _submit_and_poll_video_request(
                    session=session,
                    base_url="http://127.0.0.1:30010",
                    request=request,
                    poll_interval_s=0.01,
                )
            )

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["request_id"], "video-1")
        self.assertEqual(session.get_count, 2)

    def test_video_poll_exhausted_retry_returns_client_error_row(self):
        request = BenchmarkRequest(
            index=1,
            scheduled_submit_time_s=0.0,
            payload={"prompt": "hello"},
        )
        session = _FakeVideoSession(
            post_payload={"id": "video-2", "status": "queued"},
            get_results=[OSError("connection reset")] * 6,
        )

        with mock.patch(
            "sglang.multimodal_gen.benchmarks.wan_ti2v_profile.asyncio.sleep",
            new=_no_sleep,
        ):
            row = asyncio.run(
                _submit_and_poll_video_request(
                    session=session,
                    base_url="http://127.0.0.1:30010",
                    request=request,
                    poll_interval_s=0.01,
                )
            )

        self.assertEqual(row["status"], "client_error")
        self.assertEqual(row["request_id"], "video-2")
        self.assertIn("OSError", row["error"])
        self.assertEqual(session.get_count, 6)

    def test_task_exception_is_converted_to_client_error_row(self):
        request = BenchmarkRequest(
            index=2,
            scheduled_submit_time_s=12.5,
            payload={"prompt": "hello"},
        )

        row = _row_from_task_result(request, RuntimeError("boom"))

        self.assertEqual(row["status"], "client_error")
        self.assertEqual(row["request_id"], "")
        self.assertIn("RuntimeError: boom", row["error"])

    def test_monolithic_summary_finds_runtime_csv_in_sibling_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "run-1"
            final_dir = resolve_profile_dir(tmpdir, run_id, "monolithic", "burst")
            runtime_dir = resolve_profile_dir(tmpdir, run_id, "monolithic")

            runtime_writer = CsvProfileWriter(
                os.path.join(runtime_dir, "monolithic_server.csv")
            )
            runtime_writer.write_row(
                {
                    "request_id": "req-1",
                    "arrival_time_s": 1.0,
                    "finish_time_s": 5.0,
                    "logical_encoder_duration_ms": 100.0,
                    "logical_denoiser_duration_ms": 200.0,
                    "logical_decoder_duration_ms": 300.0,
                }
            )

            summary_lines = summarize_profile_run(
                output_dir=final_dir,
                run_id=run_id,
                deployment_mode="monolithic",
                traffic_mode="burst",
                client_rows=[{"request_id": "req-1", "status": "completed"}],
            )

            self.assertIn("e2e_latency_ms.P50=4000.000", summary_lines)
            self.assertIn("encoder_duration_ms.P50=100.000", summary_lines)
            self.assertTrue(
                os.path.exists(os.path.join(final_dir, "monolithic_server.csv"))
            )

    def test_disaggregation_summary_uses_role_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "run-2"
            final_dir = resolve_profile_dir(tmpdir, run_id, "disaggregation", "normal")
            runtime_dir = resolve_profile_dir(tmpdir, run_id, "disaggregation")

            CsvProfileWriter(os.path.join(runtime_dir, "server.csv")).write_row(
                {
                    "request_id": "req-2",
                    "request_arrival_time_s": 2.0,
                    "finish_time_s": 6.0,
                }
            )
            CsvProfileWriter(os.path.join(runtime_dir, "encoder.csv")).write_row(
                {"request_id": "req-2", "compute_duration_ms": 50.0}
            )
            CsvProfileWriter(os.path.join(runtime_dir, "denoiser.csv")).write_row(
                {"request_id": "req-2", "compute_duration_ms": 150.0}
            )
            CsvProfileWriter(os.path.join(runtime_dir, "decoder.csv")).write_row(
                {"request_id": "req-2", "compute_duration_ms": 250.0}
            )

            summary_lines = summarize_profile_run(
                output_dir=final_dir,
                run_id=run_id,
                deployment_mode="disaggregation",
                traffic_mode="normal",
                client_rows=[{"request_id": "req-2", "status": "completed"}],
            )

            self.assertIn("e2e_latency_ms.P50=4000.000", summary_lines)
            self.assertIn("denoiser_duration_ms.P50=150.000", summary_lines)
            self.assertTrue(os.path.exists(os.path.join(final_dir, "server.csv")))


if __name__ == "__main__":
    unittest.main()
