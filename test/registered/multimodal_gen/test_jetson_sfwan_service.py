"""CPU-only derived-property tests for the minimal SFWan service."""

import ast
import asyncio
import builtins
import copy
import contextvars
import hashlib
import importlib.util
import inspect
import json
import math
import random
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang.multimodal_gen.experimental.jetson_sfwan.client import (
    _build_parser,
    _make_dummy_latents,
    _post_at_offset,
    _profile_measurement_view,
    _run_dit_profile_iteration,
    _run_profile_iteration,
    build_interarrival_delays,
    run_profile_vae,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.engine import (
    JobRecord,
    SingleWorkerEngine,
    VaeJobRecord,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.model import (
    DIT_COMPONENT_NAMES,
    DecodedChunk,
    LatentChunk,
    ModelLoadConfig,
    SfWanDitModel,
    SfWanMonolithicModel,
    SfWanVaeModel,
    _ComponentSet,
    _component_runtime_contract,
    _denormalize_vae_latents_fastvideo,
    _fastvideo_t5_postprocess,
    _map_fastvideo_dmd_timesteps,
    _pred_noise_to_pred_video_fastvideo,
    _resolve_offload_settings,
    _timed_cuda_call,
    _uses_trt_vae,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.protocol import (
    DitProfileRequest,
    GenerationRequest,
    JobState,
    LatentJobSpec,
    MAX_SAFETENSORS_HEADER_BYTES,
    SafetensorsLatentPayload,
    SharedMemoryChunkReady,
    decoded_frames_for_chunk,
    deserialize_latent_tensor,
    latent_frame_count,
    parse_latent_safetensors_payload,
    serialize_latent_tensor,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.server import (
    ServerConfig,
    SfWanRuntime,
    _parse_args,
    create_app,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.transport import (
    SharedMemoryChunkSender,
    StagedDeviceChunk,
    _SharedPendingChunk,
    _SharedTransferState,
    _copy_payload_data_to_pinned,
    build_shared_memory_descriptor,
    _shared_memory_header,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_build import (
    _attach_timing_cache,
    _audit_tensorrt_tactics,
    _build_engine_bytes,
    _candidate_timing_cache_bytes,
    _audit_feature_cache_engine_io,
    _capture_target_conv_call_shapes,
    _commit_timing_cache,
    _load_validated_fp16_onnx,
    _mark_stage_audit,
    _make_export_wrappers,
    _normalize_export_cache_tensors,
    _onnx_default_opset_version,
    _parse_args as _parse_trt_build_args,
    _portable_conv3d_layout_for_export,
    _portable_nearest_upsample_for_export,
    _probe_suite_supports_prequantized_weight_fallback,
    _record_stage,
    _stage_is_current,
    _validate_onnx_fp16_io_contract,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_fusion import (
    FUSION_ANALYSIS_SCHEMA_VERSION,
    FUSION_VARIANT,
    _cache_update_shape,
    _constant_array,
    _current_shape_from_prepad,
    _find_cache_output,
    _ordered_causal_concat_inputs,
    _record_static_shape,
    _require_only_consumers,
    _require_removable_subgraph,
    _unpad_shape,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_fusion_build import (
    _analysis_signature as _fusion_analysis_signature,
    _load_analysis_contracts,
    _selected_call_sites as _fusion_selected_call_sites,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_perf_compare import (
    compare_profile_summaries,
    render_markdown as render_trt_perf_markdown,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_qdq import (
    EXPECTED_CALL_SITES,
    EXPECTED_CONV_SIGNATURES,
    EXPECTED_LOGICAL_CONVS,
    QDQ_OPSET,
    QDQ_SCHEMA_VERSION,
    QDQ_TOPOLOGY,
    WEIGHT_ENCODING_FP32_QDQ,
    WEIGHT_ENCODING_PREQUANTIZED_INT8_DQ,
    audit_qdq_model,
    rewrite_onnx_with_int8_qdq,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_profile import (
    PROFILE_CATEGORIES,
    SUPPORTED_TRT_LAYER_PROFILE_SCHEMA_VERSIONS,
    TRT_LAYER_PROFILE_SCHEMA_VERSION,
    TrtLayerProfileCapture,
    aggregate_trt_layer_profile_iterations,
    build_physical_layer_catalog,
    catalog_sha256,
    make_compact_layer_profile_metrics,
    probe_trt_layer_profile_api,
    write_trt_layer_profile_artifact,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_profile_build import (
    _parse_args as _parse_trt_profile_build_args,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_runtime import (
    TRT_VAE_CACHE_BANK_BYTES,
    TRT_VAE_CACHE_TOTAL_ELEMENTS,
    TRT_VAE_LATENT_SHAPE,
    TensorRTVaeRuntime,
    _configure_context_nvtx,
    validate_trt_vae_manifest,
)
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.distributed.local_single_process import (
    LocalSingleProcessGroupCoordinator,
)
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.loader import fsdp_load
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanT2VCrossAttention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _build_safetensors_document(header: dict, data: bytes) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    return len(header_bytes).to_bytes(8, "little") + header_bytes + data


async def _wait_until_terminal(*records: JobRecord) -> None:
    async def _wait() -> None:
        while any(
            record.state not in {JobState.COMPLETED, JobState.FAILED}
            for record in records
        ):
            await asyncio.sleep(0)

    await asyncio.wait_for(_wait(), timeout=2.0)


class TestSfWanFrameMath(CustomTestCase):
    """Protect the derived 4x temporal VAE and 3-latent chunk alignment."""

    def test_81_and_93_frame_boundaries(self):
        request_81 = GenerationRequest(prompt="test", num_frames=81)
        self.assertEqual(request_81.resolved_num_frames, 81)
        self.assertEqual(latent_frame_count(81), 21)
        self.assertEqual(request_81.total_chunks, 7)
        self.assertEqual(request_81.latent_chunk_shape, (1, 16, 3, 60, 104))
        self.assertEqual(
            sum(
                decoded_frames_for_chunk(index)
                for index in range(request_81.total_chunks)
            ),
            81,
        )

        request_93 = GenerationRequest(prompt="test", num_frames=93)
        self.assertEqual(request_93.resolved_num_frames, 93)
        self.assertEqual(latent_frame_count(93), 24)
        self.assertEqual(request_93.total_chunks, 8)
        self.assertEqual(
            sum(
                decoded_frames_for_chunk(index)
                for index in range(request_93.total_chunks)
            ),
            93,
        )
        self.assertTrue(request_93.warnings())

    def test_strict_duration_and_invalid_alignment(self):
        exact = GenerationRequest(
            prompt="test",
            duration_seconds=81 / 16,
            fps=16,
        )
        self.assertEqual(exact.resolved_num_frames, 81)
        with self.assertRaises(ValidationError):
            GenerationRequest(prompt="test", duration_seconds=5.0, fps=16)
        with self.assertRaises(ValidationError):
            GenerationRequest(prompt="test", num_frames=82)
        with self.assertRaises(ValidationError):
            GenerationRequest(
                prompt="test",
                num_frames=81,
                duration_seconds=81 / 16,
            )
        with self.assertRaises(ValidationError):
            GenerationRequest(prompt="test", height=478)
        with self.assertRaises(ValueError):
            GenerationRequest(
                prompt="test",
                height=496,
                width=832,
            ).validate_server_limits(480 * 832)


class TestSfWanFastVideoNumericalContract(CustomTestCase):
    """Pin the dtype, timestep, RNG, and low-level call contract."""

    def test_cross_attention_caches_text_kv_across_dit_forwards(self):
        class _Projection:
            def __init__(self, offset):
                self.offset = offset
                self.calls = 0

            def __call__(self, value):
                self.calls += 1
                return value + self.offset, None

        class _CoreAttention:
            def __init__(self):
                self.calls = []

            def __call__(self, q, k, v):
                self.calls.append(
                    (q.detach().clone(), k.detach().clone(), v.detach().clone())
                )
                return q

        to_q = _Projection(1)
        to_k = _Projection(2)
        to_v = _Projection(3)
        to_out = _Projection(0)
        core_attention = _CoreAttention()
        attention = SimpleNamespace(
            to_q=to_q,
            to_k=to_k,
            to_v=to_v,
            to_out=to_out,
            norm_q=lambda value: value,
            norm_k=lambda value: value,
            tp_rmsnorm=False,
            local_num_heads=2,
            head_dim=2,
            attn=core_attention,
        )
        cache = CrossAttentionKVCache(
            k=torch.zeros((1, 3, 2, 2)),
            v=torch.zeros((1, 3, 2, 2)),
        )
        first_context = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        second_context = first_context + 100

        first_output = WanT2VCrossAttention.forward(
            attention,
            torch.zeros((1, 2, 4)),
            first_context,
            None,
            crossattn_cache=cache,
        )
        cached_k = cache.k.clone()
        cached_v = cache.v.clone()

        self.assertTrue(cache.is_init)
        self.assertEqual(tuple(cache.k.shape), (1, 3, 2, 2))
        self.assertEqual(tuple(cache.v.shape), (1, 3, 2, 2))
        self.assertEqual((to_q.calls, to_k.calls, to_v.calls), (1, 1, 1))
        self.assertEqual(tuple(first_output.shape), (1, 2, 4))

        # The remaining three DMD calls and clean-KV forward reuse text K/V.
        for _ in range(4):
            WanT2VCrossAttention.forward(
                attention,
                torch.ones((1, 2, 4)),
                second_context,
                None,
                crossattn_cache=cache,
            )

        self.assertEqual((to_q.calls, to_k.calls, to_v.calls), (5, 1, 1))
        torch.testing.assert_close(cache.k, cached_k)
        torch.testing.assert_close(cache.v, cached_v)
        torch.testing.assert_close(core_attention.calls[-1][1], cached_k)
        torch.testing.assert_close(core_attention.calls[-1][2], cached_v)

        cache.reset()
        WanT2VCrossAttention.forward(
            attention,
            torch.ones((1, 2, 4)),
            second_context,
            None,
            crossattn_cache=cache,
        )

        self.assertEqual((to_q.calls, to_k.calls, to_v.calls), (6, 2, 2))
        self.assertFalse(torch.equal(cache.k, cached_k))
        self.assertFalse(torch.equal(cache.v, cached_v))

        WanT2VCrossAttention.forward(
            attention,
            torch.ones((1, 2, 4)),
            first_context,
            None,
        )
        WanT2VCrossAttention.forward(
            attention,
            torch.ones((1, 2, 4)),
            first_context,
            None,
        )
        self.assertEqual((to_q.calls, to_k.calls, to_v.calls), (8, 4, 4))

    def test_positive_prompt_encoding_uses_exact_fastvideo_inputs(self):
        observed = {}

        class _Tokenized(dict):
            def to(self, device):
                observed["token_device"] = str(device)
                return self

        class _Tokenizer:
            def __call__(self, prompts, **kwargs):
                observed["prompts"] = prompts
                observed["tokenizer_kwargs"] = kwargs
                return _Tokenized(
                    input_ids=torch.tensor([[7, 8, 0]], dtype=torch.int64),
                    attention_mask=torch.tensor(
                        [[1, 1, 0]],
                        dtype=torch.int64,
                    ),
                )

        class _TextEncoder:
            def __call__(self, **kwargs):
                observed["encoder_kwargs"] = kwargs
                return SimpleNamespace(
                    last_hidden_state=torch.ones(
                        (1, 3, 4096),
                        dtype=torch.float32,
                    )
                )

        model = object.__new__(SfWanDitModel)
        model.device = torch.device("cpu")
        model.tokenizer = _Tokenizer()
        model.text_encoder = _TextEncoder()
        model._load_config = SimpleNamespace(
            enable_profile=True,
            enable_nvtx=False,
        )
        forward_context_path = (
            "sglang.multimodal_gen.runtime.managers.forward_context.set_forward_context"
        )
        with mock.patch(
            forward_context_path,
            return_value=nullcontext(),
        ):
            prompt_embeds, _elapsed = model._encode_prompt("positive only")

        self.assertEqual(observed["prompts"], ["positive only"])
        self.assertEqual(
            observed["tokenizer_kwargs"],
            {
                "truncation": True,
                "padding": True,
                "max_length": 512,
                "add_special_tokens": True,
                "return_attention_mask": True,
                "return_tensors": "pt",
            },
        )
        self.assertEqual(
            observed["encoder_kwargs"]["input_ids"].dtype,
            torch.int64,
        )
        self.assertEqual(
            observed["encoder_kwargs"]["attention_mask"].dtype,
            torch.int64,
        )
        self.assertFalse(observed["encoder_kwargs"]["output_hidden_states"])
        self.assertEqual(tuple(prompt_embeds.shape), (1, 512, 4096))
        self.assertEqual(prompt_embeds.dtype, torch.float32)
        source = inspect.getsource(SfWanDitModel)
        self.assertNotIn("negative_prompt", source)
        self.assertNotIn("guidance_scale", source)

    def test_t5_postprocess_is_fp32_and_pads_to_512(self):
        hidden = torch.arange(
            3 * 4096,
            dtype=torch.float32,
        ).reshape(1, 3, 4096)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)
        result = _fastvideo_t5_postprocess(hidden, mask)
        self.assertEqual(tuple(result.shape), (1, 512, 4096))
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result[:, :2], hidden[:, :2])
        self.assertEqual(torch.count_nonzero(result[:, 2:]).item(), 0)

    def test_scheduler_mapping_does_not_require_runtime_set_timesteps(self):
        base_sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]
        shifted_sigmas = 5 * base_sigmas / (1 + 4 * base_sigmas)
        mapped = _map_fastvideo_dmd_timesteps(
            scheduler_timesteps=shifted_sigmas * 1000,
            raw_dmd_steps=[1000, 750, 500, 250],
            num_train_timesteps=1000,
        )
        torch.testing.assert_close(
            mapped,
            torch.tensor([1000.0, 937.5, 833.3333, 625.0]),
            rtol=1e-5,
            atol=1e-4,
        )
        source = inspect.getsource(SfWanDitModel)
        self.assertNotIn(".set_timesteps(", source)
        self.assertNotIn("pipelines_core", source)

    def test_initial_noise_is_full_sequence_fp32_and_seeded(self):
        model = object.__new__(SfWanDitModel)
        model.device = torch.device("cpu")
        request = GenerationRequest(
            prompt="test",
            height=16,
            width=16,
            num_frames=9,
        )
        first = model._prepare_initial_latents(
            request,
            generator=torch.Generator(device="cpu").manual_seed(1024),
            dtype=torch.float32,
        )
        second = model._prepare_initial_latents(
            request,
            generator=torch.Generator(device="cpu").manual_seed(1024),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(first.shape), (1, 16, 3, 2, 2))
        self.assertEqual(first.dtype, torch.float32)
        torch.testing.assert_close(first, second)

    def test_dmd_calls_four_forwards_then_clean_kv(self):
        class _Scheduler:
            def __init__(self):
                self.timesteps = torch.tensor(
                    [1000.0, 937.5, 833.3333, 625.0],
                    dtype=torch.float32,
                )
                self.sigmas = torch.tensor(
                    [1.0, 0.9375, 0.8333333, 0.625],
                    dtype=torch.float32,
                )
                self.add_noise_calls = []

            def add_noise(self, original_samples, noise, timestep):
                self.add_noise_calls.append(
                    {
                        "original_dtype": original_samples.dtype,
                        "noise_dtype": noise.dtype,
                        "timestep_dtype": timestep.dtype,
                    }
                )
                return noise

            def step(self, *_args, **_kwargs):
                raise AssertionError("causal DMD must not call scheduler.step")

        class _Transformer:
            def __init__(self):
                self.calls = []

            def __call__(
                self,
                latent,
                prompt_embeds,
                timestep,
                **kwargs,
            ):
                self.calls.append(
                    {
                        "latent_dtype": latent.dtype,
                        "prompt_dtype": prompt_embeds.dtype,
                        "timestep_dtype": timestep.dtype,
                        "timestep_shape": tuple(timestep.shape),
                        "current_start": kwargs["current_start"],
                        "start_frame": kwargs["start_frame"],
                    }
                )
                return torch.zeros_like(latent, dtype=torch.bfloat16)

        model = object.__new__(SfWanDitModel)
        model.device = torch.device("cpu")
        model.target_dtype = torch.bfloat16
        model.scheduler = _Scheduler()
        model.transformer = _Transformer()
        model.pipeline_config = SimpleNamespace(context_noise=0)
        model._load_config = SimpleNamespace(
            enable_profile=True,
            enable_nvtx=False,
        )
        model._kv_cache = [object()]
        model._crossattn_cache = [object()]
        chunk = torch.randn((1, 16, 3, 2, 2), dtype=torch.float32)
        clean, metrics = model._denoise_chunk(
            chunk=chunk,
            prompt_embeds=torch.zeros((1, 512, 4096), dtype=torch.float32),
            timesteps=model.scheduler.timesteps,
            generator=torch.Generator(device="cpu").manual_seed(1024),
            current_start_tokens=0,
            start_frame=0,
        )

        self.assertEqual(len(model.transformer.calls), 5)
        self.assertTrue(
            all(
                call["latent_dtype"] == torch.bfloat16
                for call in model.transformer.calls
            )
        )
        self.assertTrue(
            all(
                call["prompt_dtype"] == torch.float32
                for call in model.transformer.calls
            )
        )
        self.assertTrue(
            all(call["timestep_shape"] == (1, 1) for call in model.transformer.calls)
        )
        self.assertTrue(
            all(
                call["timestep_dtype"] == torch.float32
                for call in model.transformer.calls[:4]
            )
        )
        self.assertEqual(
            model.transformer.calls[-1]["timestep_dtype"],
            torch.int64,
        )
        self.assertEqual(len(model.scheduler.add_noise_calls), 3)
        self.assertEqual(len(metrics["denoise_steps"]), 4)
        self.assertTrue(
            all(
                step["pred_to_video_cuda_ms"] is not None
                for step in metrics["denoise_steps"]
            )
        )
        self.assertTrue(
            all(
                step["renoise_cuda_ms"] is not None
                for step in metrics["denoise_steps"][:3]
            )
        )
        self.assertIsNone(metrics["denoise_steps"][3]["renoise_cuda_ms"])
        self.assertTrue(
            all(
                call["noise_dtype"] == torch.bfloat16
                for call in model.scheduler.add_noise_calls
            )
        )
        self.assertEqual(clean.dtype, torch.bfloat16)
        self.assertEqual(chunk.dtype, torch.float32)

    def test_chunk_publication_occurs_after_clean_kv_and_cache_resets(self):
        actions = []
        model = object.__new__(SfWanDitModel)
        model.device = torch.device("cpu")
        model._dmd_timesteps_cpu = torch.tensor([1.0])
        model._load_config = SimpleNamespace(
            enable_profile=True,
            enable_nvtx=False,
        )

        def _encode_prompt(_self, _prompt):
            return (
                torch.zeros((1, 512, 4096), dtype=torch.float32),
                {"text_encode_ms": 0.0},
            )

        def _prepare_initial(_self, _request, *, generator, dtype):
            del generator
            return torch.zeros((1, 16, 3, 2, 2), dtype=dtype)

        def _denoise(_self, **kwargs):
            del kwargs
            actions.append("clean_kv_completed")
            return (
                torch.zeros((1, 16, 3, 2, 2), dtype=torch.bfloat16),
                {"clean_kv_cuda_ms": 1.0},
            )

        model._encode_prompt = MethodType(_encode_prompt, model)
        model._prepare_initial_latents = MethodType(_prepare_initial, model)
        model._prepare_caches = MethodType(
            lambda _self, **_kwargs: 1,
            model,
        )
        model._denoise_chunk = MethodType(_denoise, model)
        model._reset_caches = MethodType(
            lambda _self: actions.append("cache_reset"),
            model,
        )
        result = model.generate(
            request=GenerationRequest(
                prompt="test",
                height=16,
                width=16,
                num_frames=9,
            ),
            chunk_callback=lambda _chunk: actions.append("published"),
        )
        self.assertEqual(
            actions,
            ["clean_kv_completed", "published", "cache_reset"],
        )
        execution = result["profile_execution"]
        self.assertEqual(
            [chunk["chunk_index"] for chunk in execution["chunks"]],
            [0],
        )
        for metric_name in (
            "latent_init_cuda_ms",
            "latent_init_ms",
            "cache_prepare_cuda_ms",
            "cache_prepare_ms",
            "timestep_to_device_cuda_ms",
            "timestep_to_device_wall_ms",
            "dit_execution_wall_ms",
        ):
            self.assertIn(metric_name, execution)
            self.assertIsNotNone(execution[metric_name])

    def test_dit_preparation_failure_still_resets_request_caches(self):
        actions = []
        model = object.__new__(SfWanDitModel)
        model.device = torch.device("cpu")
        model._load_config = SimpleNamespace(
            enable_profile=False,
            enable_nvtx=False,
        )

        def _fail_prompt(_self, _prompt):
            raise RuntimeError("expected preparation failure")

        model._encode_prompt = MethodType(_fail_prompt, model)
        model._reset_caches = MethodType(
            lambda _self: actions.append("cache_reset"),
            model,
        )
        with self.assertRaisesRegex(RuntimeError, "preparation failure"):
            model.generate(
                request=GenerationRequest(
                    prompt="test",
                    height=16,
                    width=16,
                    num_frames=9,
                ),
                chunk_callback=lambda _chunk: None,
            )
        self.assertEqual(actions, ["cache_reset"])

    def test_pred_conversion_and_vae_denorm_dtypes(self):
        scheduler = SimpleNamespace(
            timesteps=torch.tensor([1000.0, 500.0]),
            sigmas=torch.tensor([1.0, 0.5]),
        )
        prediction = _pred_noise_to_pred_video_fastvideo(
            pred_noise=torch.ones((3, 1, 1, 1), dtype=torch.bfloat16),
            noise_input_latent=torch.zeros((3, 1, 1, 1), dtype=torch.float32),
            timestep=torch.full((1, 3), 500.0),
            scheduler=scheduler,
        )
        self.assertEqual(prediction.dtype, torch.bfloat16)
        torch.testing.assert_close(
            prediction,
            torch.full_like(prediction, -0.5),
        )

        normalized = torch.ones((1, 16, 3, 1, 1), dtype=torch.bfloat16)
        mean = torch.full((1, 16, 1, 1, 1), 0.25)
        std = torch.full((1, 16, 1, 1, 1), 2.0)
        denormalized = _denormalize_vae_latents_fastvideo(
            normalized,
            latents_mean=mean,
            latents_std=std,
        )
        self.assertEqual(denormalized.dtype, torch.float32)
        torch.testing.assert_close(
            denormalized,
            torch.full_like(denormalized, 2.25),
        )

    def test_vae_ingress_widens_before_decode_and_uint8_truncates(self):
        model = object.__new__(SfWanVaeModel)
        model._request_active = True
        model.device = torch.device("cpu")
        model._load_config = SimpleNamespace(
            enable_profile=False,
            enable_nvtx=False,
        )
        model._latents_mean = torch.zeros((1, 16, 1, 1, 1))
        model._latents_std = torch.ones((1, 16, 1, 1, 1))
        observed = {}

        def _decode_per_latent(_self, *, chunk_index, z):
            observed["dtype"] = z.dtype
            observed["shape"] = tuple(z.shape)
            return torch.zeros((1, 3, 1, 1, 1)), [], 0.0

        model._decode_per_latent = MethodType(_decode_per_latent, model)
        decoded = model.decode_chunk(
            chunk_index=0,
            latents=torch.ones((1, 16, 3, 1, 1), dtype=torch.bfloat16),
        )
        self.assertEqual(observed["dtype"], torch.float32)
        self.assertEqual(observed["shape"], (1, 16, 3, 1, 1))
        self.assertEqual(decoded.frame_count, 1)
        self.assertTrue((decoded.frames == 127).all())

        discarded = model.decode_chunk(
            chunk_index=1,
            latents=torch.ones((1, 16, 3, 1, 1), dtype=torch.bfloat16),
            return_frames=False,
        )
        self.assertIsNone(discarded.frames)

    def test_vae_active_flag_is_cleared_even_if_native_reset_fails(self):
        class _FailingVae:
            @staticmethod
            def reset_causal_decode_state():
                raise RuntimeError("native reset failed")

        model = object.__new__(SfWanVaeModel)
        model.vae = _FailingVae()
        model._request_active = True
        with self.assertRaisesRegex(RuntimeError, "native reset failed"):
            model.finish_request()
        self.assertFalse(model._request_active)

    def test_vae_profile_chunk_reports_execution_before_rgb_output(self):
        model = object.__new__(SfWanVaeModel)
        model._request_active = True
        model.device = torch.device("cpu")
        model._load_config = SimpleNamespace(
            enable_profile=True,
            enable_nvtx=False,
        )
        model._latents_mean = torch.zeros((1, 16, 1, 1, 1))
        model._latents_std = torch.ones((1, 16, 1, 1, 1))

        def _decode_per_latent(_self, *, chunk_index, z):
            self.assertEqual(chunk_index, 0)
            self.assertEqual(z.dtype, torch.float32)
            return (
                torch.zeros((1, 3, 9, 1, 1), dtype=torch.float32),
                [
                    {
                        "latent_index": index,
                        "cuda_ms": 0.5,
                        "decoded_rgb_frames": frames,
                    }
                    for index, frames in enumerate((1, 4, 4))
                ],
                0.25,
            )

        model._decode_per_latent = MethodType(_decode_per_latent, model)
        decoded = model.decode_chunk(
            chunk_index=0,
            latents=torch.zeros(
                (1, 16, 3, 1, 1),
                dtype=torch.bfloat16,
            ),
            return_frames=False,
        )
        execution = decoded.metrics["profile_execution"]
        self.assertEqual(execution["chunk_index"], 0)
        self.assertEqual(execution["decoded_rgb_frames"], 9)
        self.assertEqual(
            [frame["decoded_rgb_frames"] for frame in execution["latent_frames"]],
            [1, 4, 4],
        )
        self.assertIn("bf16_ingress_cuda_ms", execution)
        self.assertIn("denorm_cuda_ms", execution)
        self.assertIn("post_quant_cuda_ms", execution)
        self.assertIn("chunk_execution_cuda_ms", execution)
        self.assertNotIn("rgb_d2h_wall_ms", decoded.metrics)

    def test_vae_calls_post_quant_once_and_decoder_per_latent(self):
        calls = []
        wanvae_module = ModuleType("sglang.multimodal_gen.runtime.models.vaes.wanvae")
        wanvae_module.feat_idx = contextvars.ContextVar("feat_idx", default=0)
        wanvae_module.first_chunk = contextvars.ContextVar(
            "first_chunk",
            default=None,
        )
        wanvae_module.disable_spatial_parallel_decode = nullcontext
        wanvae_module.forward_context = lambda **_kwargs: nullcontext()

        class _FakeVae:
            def __init__(self):
                self.config = SimpleNamespace(patch_size=None)
                self._causal_decode_initialized = False
                self._feat_map = []
                self._conv_idx = 0

            def _should_use_spatial_parallel_decode(self, _z):
                return False

            def post_quant_conv(self, z):
                calls.append(("post_quant", tuple(z.shape), z.dtype))
                return z

            def decoder(self, latent):
                from sglang.multimodal_gen.runtime.models.vaes import wanvae

                calls.append(
                    (
                        "decoder",
                        tuple(latent.shape),
                        bool(wanvae.first_chunk.get()),
                    )
                )
                frame_count = (
                    1
                    if len([call for call in calls if call[0] == "decoder"]) == 1
                    else 4
                )
                return torch.zeros(
                    (1, 3, frame_count, 8, 8),
                    dtype=torch.float32,
                )

        model = object.__new__(SfWanVaeModel)
        model.vae = _FakeVae()
        model.vae_dtype = torch.float32
        model._load_config = SimpleNamespace(
            enable_profile=True,
            enable_nvtx=False,
        )
        with mock.patch.dict(
            sys.modules,
            {("sglang.multimodal_gen.runtime.models.vaes.wanvae"): wanvae_module},
        ):
            first, first_metrics, _ = model._decode_per_latent(
                chunk_index=0,
                z=torch.zeros((1, 16, 3, 1, 1), dtype=torch.float32),
            )
            second, second_metrics, _ = model._decode_per_latent(
                chunk_index=1,
                z=torch.zeros((1, 16, 3, 1, 1), dtype=torch.float32),
            )

        post_quant_calls = [call for call in calls if call[0] == "post_quant"]
        decoder_calls = [call for call in calls if call[0] == "decoder"]
        self.assertEqual(len(post_quant_calls), 2)
        self.assertEqual(len(decoder_calls), 6)
        self.assertTrue(all(call[1] == (1, 16, 1, 1, 1) for call in decoder_calls))
        self.assertEqual(
            [call[2] for call in decoder_calls],
            [True, False, False, False, False, False],
        )
        self.assertEqual(first.shape[2], 9)
        self.assertEqual(second.shape[2], 12)
        self.assertEqual(
            [metric["decoded_rgb_frames"] for metric in first_metrics],
            [1, 4, 4],
        )
        self.assertEqual(
            [metric["decoded_rgb_frames"] for metric in second_metrics],
            [4, 4, 4],
        )


class TestSfWanLocalDistributedCompatibility(CustomTestCase):
    """Jetson builds without C10d get identity groups, never fake multi-rank."""

    def tearDown(self):
        if parallel_state.is_local_single_process_mode():
            parallel_state.destroy_model_parallel()
            parallel_state.destroy_distributed_environment()
        super().tearDown()

    def test_local_group_has_strict_identity_semantics(self):
        group = LocalSingleProcessGroupCoordinator(group_name="test")
        tensor = torch.arange(4)

        self.assertEqual(group.rank, 0)
        self.assertEqual(group.rank_in_group, 0)
        self.assertEqual(group.world_size, 1)
        self.assertEqual(group.ranks, [0])
        self.assertIs(group.ulysses_group, group)
        self.assertIs(group.ring_group, group)
        self.assertIs(group.all_reduce(tensor), tensor)
        self.assertIs(group.all_gather(tensor), tensor)
        gathered = group.all_gather(tensor, separate_tensors=True)
        self.assertEqual(len(gathered), 1)
        self.assertIs(gathered[0], tensor)
        self.assertIs(group.all_to_all_4D(tensor), tensor)
        self.assertIs(group.broadcast(tensor), tensor)
        self.assertIsNone(group.barrier())
        with self.assertRaisesRegex(RuntimeError, "Peer-to-peer"):
            group.send(tensor, dst=0)

    def test_missing_is_initialized_is_safe(self):
        fake_dist = SimpleNamespace(is_available=lambda: True)
        with mock.patch.object(parallel_state.torch, "distributed", fake_dist):
            self.assertFalse(parallel_state.is_torch_distributed_initialized())

    def test_multimodal_public_api_does_not_eagerly_load_generic_runtime(self):
        import sglang.multimodal_gen as multimodal_gen

        probe_name = "_sfwan_multimodal_lazy_import_probe"
        spec = importlib.util.spec_from_file_location(
            probe_name,
            Path(multimodal_gen.__file__),
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertNotIn("DiffGenerator", vars(module))
        self.assertNotIn("PipelineConfig", vars(module))
        self.assertNotIn("SamplingParams", vars(module))
        self.assertTrue(callable(module.__getattr__))

    def test_builtin_quantization_backend_is_loaded_only_when_selected(self):
        from sglang.multimodal_gen.runtime.layers import (
            quantization as diffusion_quantization,
        )

        fake_fp8_config = type("FakeFp8Config", (), {})
        fake_fp8_module = SimpleNamespace(Fp8Config=fake_fp8_config)
        with mock.patch.object(
            diffusion_quantization,
            "import_module",
            return_value=fake_fp8_module,
        ) as import_module:
            result = diffusion_quantization.get_quantization_config("fp8")

        self.assertIs(result, fake_fp8_config)
        import_module.assert_called_once_with(
            "sglang.multimodal_gen.runtime.layers.quantization.fp8"
        )

    def test_transformer_loader_uses_diffusion_quantization_base(self):
        from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
            QuantizationConfig as DiffusionQuantizationConfig,
        )
        from sglang.multimodal_gen.runtime.loader import transformer_load_utils

        self.assertIs(
            transformer_load_utils.QuantizationConfig,
            DiffusionQuantizationConfig,
        )
        source = inspect.getsource(transformer_load_utils)
        self.assertNotIn(
            "from sglang.srt.layers.quantization import QuantizationConfig",
            source,
        )

    def test_cache_package_does_not_eagerly_import_cache_dit(self):
        import sglang.multimodal_gen.runtime.cache as cache_package

        probe_name = "_sfwan_cache_lazy_import_probe"
        spec = importlib.util.spec_from_file_location(
            probe_name,
            Path(cache_package.__file__),
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.endswith(".cache.cache_dit_integration"):
                raise AssertionError("cache_dit integration was imported eagerly")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            spec.loader.exec_module(module)

        self.assertIn("TeaCacheContext", vars(module))
        self.assertNotIn("CacheDitConfig", vars(module))
        self.assertTrue(callable(module.__getattr__))

        fake_config = type("FakeCacheDitConfig", (), {})
        fake_integration = SimpleNamespace(CacheDitConfig=fake_config)
        with mock.patch.object(
            module,
            "import_module",
            return_value=fake_integration,
        ) as import_module:
            self.assertIs(module.CacheDitConfig, fake_config)

        import_module.assert_called_once_with(
            "sglang.multimodal_gen.runtime.cache.cache_dit_integration"
        )

    def test_sm87_flash_attention_dispatches_to_external_fa2(self):
        from sglang.kernels.ops.attention import flash_attention_v3

        is_fa3_supported = inspect.unwrap(flash_attention_v3._is_fa3_supported)
        with (
            mock.patch.object(
                flash_attention_v3,
                "get_device_capability",
                return_value=(8, 7),
            ),
            mock.patch.object(flash_attention_v3, "is_musa", return_value=False),
            mock.patch.object(flash_attention_v3.torch.version, "cuda", "12.9"),
        ):
            self.assertFalse(is_fa3_supported())

        expected = object()
        fa2_flash_attn_func = mock.Mock(return_value=expected)
        fake_flash_attn = ModuleType("flash_attn")
        fake_flash_attn.flash_attn_func = fa2_flash_attn_func

        with (
            mock.patch.object(
                flash_attention_v3,
                "_is_fa3_supported",
                return_value=False,
            ),
            mock.patch.object(
                flash_attention_v3,
                "_load_fa3_kernels",
                side_effect=AssertionError("FA3 loader must not run on SM87"),
            ),
            mock.patch.dict(sys.modules, {"flash_attn": fake_flash_attn}),
        ):
            result = inspect.unwrap(flash_attention_v3.flash_attn_varlen_func)(
                object(),
                object(),
                object(),
                None,
                None,
                softmax_scale=0.125,
                causal=False,
            )

        self.assertIs(result, expected)
        fa2_flash_attn_func.assert_called_once()
        self.assertEqual(
            fa2_flash_attn_func.call_args.kwargs,
            {
                "softmax_scale": 0.125,
                "causal": False,
                "window_size": (-1, -1),
                "softcap": 0.0,
                "return_attn_probs": False,
            },
        )

    def test_sm120_flash_attention_restores_sm80_algorithm_selector(self):
        source_path = (
            Path(__file__).resolve().parents[3]
            / "python"
            / "sglang"
            / "kernels"
            / "ops"
            / "attention"
            / "flash_attn"
            / "cute"
            / "flash_fwd_sm120.py"
        )
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        sm120_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "FlashAttentionForwardSm120"
        )
        initializer = next(
            node
            for node in sm120_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )

        self.assertIsInstance(initializer.body[0], ast.Expr)
        self.assertEqual(
            ast.unparse(initializer.body[0].value),
            "super().__init__(*args, **kwargs)",
        )

        assignments = {
            ast.unparse(statement.targets[0]): ast.unparse(statement.value)
            for statement in initializer.body
            if isinstance(statement, ast.Assign)
        }
        self.assertEqual(assignments["self.arch"], "Arch.sm_80")
        self.assertEqual(assignments["self.is_split_kv"], "False")
        self.assertEqual(assignments["self.pack_gqa"], "False")

    def test_local_initializer_builds_every_world_size_one_group(self):
        with (
            mock.patch.object(
                parallel_state,
                "is_torch_distributed_available",
                return_value=False,
            ),
            mock.patch.object(
                torch.distributed,
                "init_process_group",
                create=True,
            ) as init_process_group,
            mock.patch.dict(
                "os.environ",
                {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1"},
            ),
        ):
            parallel_state.maybe_init_distributed_environment_and_model_parallel(
                tp_size=1,
                sp_size=1,
                cfg_degree=1,
                ulysses_degree=1,
                ring_degree=1,
                dp_size=1,
            )

        init_process_group.assert_not_called()
        self.assertTrue(parallel_state.is_local_single_process_mode())
        self.assertEqual(parallel_state.get_distributed_backend_name(), "local")
        self.assertEqual(parallel_state.get_world_size(), 1)
        self.assertEqual(parallel_state.get_tp_world_size(), 1)
        self.assertEqual(parallel_state.get_sp_world_size(), 1)
        self.assertEqual(parallel_state.get_dp_world_size(), 1)
        self.assertEqual(
            parallel_state.get_classifier_free_guidance_world_size(),
            1,
        )
        self.assertEqual(parallel_state.get_pipeline_parallel_world_size(), 1)
        self.assertEqual(parallel_state.get_decode_parallel_world_size(), 1)

    def test_local_initializer_rejects_multi_rank_settings(self):
        cases = [
            {"WORLD_SIZE": "2"},
            {"RANK": "1"},
        ]
        for environment in cases:
            with (
                self.subTest(environment=environment),
                mock.patch.object(
                    parallel_state,
                    "is_torch_distributed_available",
                    return_value=False,
                ),
                mock.patch.dict(
                    "os.environ",
                    {
                        "LOCAL_RANK": "0",
                        "RANK": "0",
                        "WORLD_SIZE": "1",
                        **environment,
                    },
                ),
                self.assertRaisesRegex(RuntimeError, "world size one"),
            ):
                parallel_state.maybe_init_distributed_environment_and_model_parallel(
                    tp_size=1,
                    sp_size=1,
                    cfg_degree=1,
                    ulysses_degree=1,
                    ring_degree=1,
                    dp_size=1,
                )

        for degree_name in (
            "tp_size",
            "sp_size",
            "cfg_degree",
            "ulysses_degree",
            "ring_degree",
            "dp_size",
        ):
            arguments = {
                "tp_size": 1,
                "sp_size": 1,
                "cfg_degree": 1,
                "ulysses_degree": 1,
                "ring_degree": 1,
                "dp_size": 1,
            }
            arguments[degree_name] = 2
            with (
                self.subTest(degree_name=degree_name),
                mock.patch.object(
                    parallel_state,
                    "is_torch_distributed_available",
                    return_value=False,
                ),
                mock.patch.dict(
                    "os.environ",
                    {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1"},
                ),
                self.assertRaisesRegex(RuntimeError, degree_name),
            ):
                parallel_state.maybe_init_distributed_environment_and_model_parallel(
                    **arguments
                )

    def test_normal_distributed_branch_is_preserved(self):
        with (
            mock.patch.object(
                parallel_state,
                "is_torch_distributed_available",
                return_value=True,
            ),
            mock.patch.object(
                parallel_state,
                "init_distributed_environment",
            ) as init_environment,
            mock.patch.object(
                parallel_state,
                "initialize_model_parallel",
            ) as init_model_parallel,
            mock.patch.dict(
                "os.environ",
                {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1"},
            ),
        ):
            parallel_state.maybe_init_distributed_environment_and_model_parallel(
                tp_size=1,
                sp_size=1,
                cfg_degree=1,
                ulysses_degree=1,
                ring_degree=1,
                dp_size=1,
            )

        init_environment.assert_called_once()
        init_model_parallel.assert_called_once()

    def test_fsdp_unavailable_has_a_clear_failure(self):
        with mock.patch.object(fsdp_load, "_FSDP_AVAILABLE", False):
            self.assertFalse(fsdp_load.is_fsdp_available())
            with self.assertRaisesRegex(RuntimeError, "compiled without"):
                fsdp_load.require_fsdp_support()

    def test_fsdp_unavailable_keeps_the_non_fsdp_loader_path(self):
        class _FakeModel(torch.nn.Module):
            param_names_mapping = {}

            def post_load_weights(self):
                return None

        with (
            mock.patch.object(fsdp_load, "_FSDP_AVAILABLE", False),
            mock.patch.object(fsdp_load, "set_mixed_precision_policy"),
            mock.patch.object(
                fsdp_load,
                "safetensors_weights_iterator",
                return_value=iter(()),
            ),
            mock.patch.object(
                fsdp_load,
                "load_model_from_full_model_state_dict",
            ) as load_state_dict,
        ):
            model = fsdp_load.maybe_load_fsdp_model(
                model_cls=_FakeModel,
                init_params={},
                weight_dir_list=[],
                device=torch.device("cpu"),
                hsdp_replicate_dim=1,
                hsdp_shard_dim=1,
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                fsdp_inference=False,
            )

        self.assertIsInstance(model, _FakeModel)
        load_state_dict.assert_called_once()

    def test_fsdp_shard_model_restores_default_mixed_precision_policy(self):
        model = torch.nn.Module()
        model.layer = torch.nn.Linear(1, 1)
        policy = object()

        with (
            mock.patch.object(fsdp_load, "_FSDP_AVAILABLE", True),
            mock.patch.object(
                fsdp_load,
                "MixedPrecisionPolicy",
                return_value=policy,
            ) as policy_factory,
            mock.patch.object(fsdp_load, "fully_shard") as fully_shard,
        ):
            fsdp_load.shard_model(
                model,
                cpu_offload=False,
                fsdp_shard_conditions=[
                    lambda name, _module: name == "layer",
                ],
            )

        policy_factory.assert_called_once_with()
        self.assertEqual(fully_shard.call_count, 2)
        for call in fully_shard.call_args_list:
            self.assertIs(call.kwargs["mp_policy"], policy)


class TestSfWanCpuOffloadConfiguration(CustomTestCase):
    """CPU-offload choices are explicit, role-safe, and request-stable."""

    @staticmethod
    def _parse_server_cli(*arguments):
        with mock.patch.object(
            sys,
            "argv",
            ["sfwan-server", *arguments],
        ):
            return _parse_args()

    def test_server_cli_offload_defaults(self):
        config = self._parse_server_cli("--role", "monolithic")
        self.assertTrue(config.text_encoder_cpu_offload)
        self.assertFalse(config.dit_cpu_offload)
        self.assertFalse(config.vae_cpu_offload)
        load_config = ModelLoadConfig(model_path="unused")
        self.assertTrue(load_config.text_encoder_cpu_offload)
        self.assertFalse(load_config.dit_cpu_offload)
        self.assertFalse(load_config.vae_cpu_offload)

    def test_server_cli_accepts_bare_and_explicit_booleans(self):
        enabled = self._parse_server_cli(
            "--role",
            "monolithic",
            "--text-encoder-cpu-offload",
            "false",
            "--dit-cpu-offload",
            "--vae-cpu-offload",
            "true",
        )
        self.assertFalse(enabled.text_encoder_cpu_offload)
        self.assertTrue(enabled.dit_cpu_offload)
        self.assertTrue(enabled.vae_cpu_offload)

        disabled = self._parse_server_cli(
            "--role",
            "monolithic",
            "--text-encoder-cpu-offload",
            "true",
            "--dit-cpu-offload",
            "false",
            "--vae-cpu-offload",
            "false",
        )
        self.assertTrue(disabled.text_encoder_cpu_offload)
        self.assertFalse(disabled.dit_cpu_offload)
        self.assertFalse(disabled.vae_cpu_offload)

    def test_runtime_propagates_offload_choices_to_model_load_config(self):
        async def _scenario():
            captured = []

            def _factory(load_config):
                captured.append(load_config)
                return SimpleNamespace(contract={"role": "monolithic"})

            runtime = SfWanRuntime(
                config=ServerConfig(
                    role="monolithic",
                    text_encoder_cpu_offload=False,
                    dit_cpu_offload=True,
                    vae_cpu_offload=True,
                ),
                model_factory=_factory,
            )
            await runtime.start()
            try:
                self.assertEqual(len(captured), 1)
                load_config = captured[0]
                self.assertFalse(load_config.text_encoder_cpu_offload)
                self.assertTrue(load_config.dit_cpu_offload)
                self.assertTrue(load_config.vae_cpu_offload)
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_component_loader_uses_manual_mode_and_couples_dit_fsdp(self):
        model_module = "sglang.multimodal_gen.experimental.jetson_sfwan.model"

        for dit_cpu_offload in (False, True):
            captured = {}

            def _server_args(**kwargs):
                if kwargs["performance_mode"] != "manual":
                    kwargs["text_encoder_cpu_offload"] = True
                    kwargs["dit_cpu_offload"] = False
                    kwargs["vae_cpu_offload"] = False
                    kwargs["use_fsdp_inference"] = False
                captured.update(kwargs)
                return SimpleNamespace(**kwargs)

            class _FakePipelineConfig:
                vae_precision = "fp32"

            pipeline_configs = ModuleType(
                "sglang.multimodal_gen.configs.pipeline_configs"
            )
            pipeline_configs.SelfForcingWanT2V480PConfig = _FakePipelineConfig
            server_args_module = ModuleType("sglang.multimodal_gen.runtime.server_args")
            server_args_module.ServerArgs = _server_args
            server_args_module.set_global_server_args = mock.Mock()
            hf_utils = ModuleType(
                "sglang.multimodal_gen.runtime.utils.hf_diffusers_utils"
            )
            hf_utils.maybe_download_model = lambda *_args, **_kwargs: "unused"

            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        pipeline_configs.__name__: pipeline_configs,
                        server_args_module.__name__: server_args_module,
                        hf_utils.__name__: hf_utils,
                    },
                ),
                mock.patch(
                    f"{model_module}._initialize_single_gpu_runtime",
                    return_value=torch.device("cpu"),
                ),
                mock.patch(
                    f"{model_module}._distributed_runtime_contract",
                    return_value=("nccl", False),
                ),
                mock.patch(
                    "pathlib.Path.open",
                    mock.mock_open(read_data="{}"),
                ),
            ):
                components = _ComponentSet(
                    load_config=ModelLoadConfig(
                        model_path="unused",
                        text_encoder_cpu_offload=False,
                        dit_cpu_offload=dit_cpu_offload,
                        vae_cpu_offload=True,
                    ),
                    component_names=(),
                )

            self.assertEqual(components.server_args.performance_mode, "manual")
            self.assertFalse(components.server_args.text_encoder_cpu_offload)
            self.assertEqual(
                components.server_args.dit_cpu_offload,
                dit_cpu_offload,
            )
            self.assertTrue(components.server_args.vae_cpu_offload)
            self.assertEqual(
                components.server_args.use_fsdp_inference,
                dit_cpu_offload,
            )
            self.assertEqual(components.distributed_backend, "nccl")

    def test_local_offload_resolution_is_role_aware(self):
        defaults = ModelLoadConfig(model_path="unused")
        selected, effective = _resolve_offload_settings(
            load_config=defaults,
            component_names=("transformer", "text_encoder"),
            local_single_process=True,
        )
        self.assertEqual(selected, ["text_encoder"])
        self.assertEqual(
            effective,
            {
                "text_encoder": "layerwise",
                "dit": "resident",
                "vae": "resident",
            },
        )

        all_enabled = ModelLoadConfig(
            model_path="unused",
            text_encoder_cpu_offload=True,
            dit_cpu_offload=True,
            vae_cpu_offload=True,
        )
        selected, effective = _resolve_offload_settings(
            load_config=all_enabled,
            component_names=("transformer", "text_encoder", "vae"),
            local_single_process=True,
        )
        self.assertEqual(selected, ["text_encoder", "dit"])
        self.assertEqual(effective["text_encoder"], "layerwise")
        self.assertEqual(effective["dit"], "layerwise")
        self.assertEqual(effective["vae"], "per_chunk_module")

        vae_only, effective = _resolve_offload_settings(
            load_config=all_enabled,
            component_names=("vae",),
            local_single_process=True,
        )
        self.assertIsNone(vae_only)
        self.assertEqual(effective["vae"], "per_chunk_module")

    def test_runtime_contract_separates_requested_and_effective_offload(self):
        components = SimpleNamespace(
            distributed_backend="local",
            cpu_offload_requested={
                "text_encoder": True,
                "dit": False,
                "vae": True,
            },
            cpu_offload_effective={
                "text_encoder": "layerwise",
                "dit": "resident",
                "vae": "per_chunk_module",
            },
        )
        contract = _component_runtime_contract(
            components,
            ("text_encoder", "dit"),
        )
        self.assertEqual(contract["distributed_backend"], "local")
        self.assertEqual(
            contract["cpu_offload_requested"],
            {"text_encoder": True, "dit": False},
        )
        self.assertEqual(
            contract["cpu_offload_effective"],
            {"text_encoder": "layerwise", "dit": "resident"},
        )

    def test_local_component_set_configures_layerwise_after_loading(self):
        captured = {}
        model_module = "sglang.multimodal_gen.experimental.jetson_sfwan.model"

        def _server_args(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        class _FakePipelineConfig:
            vae_precision = "fp32"

        pipeline_configs = ModuleType("sglang.multimodal_gen.configs.pipeline_configs")
        pipeline_configs.SelfForcingWanT2V480PConfig = _FakePipelineConfig
        server_args_module = ModuleType("sglang.multimodal_gen.runtime.server_args")
        server_args_module.ServerArgs = _server_args
        server_args_module.set_global_server_args = mock.Mock()
        hf_utils = ModuleType("sglang.multimodal_gen.runtime.utils.hf_diffusers_utils")
        hf_utils.maybe_download_model = lambda *_args, **_kwargs: "unused"
        layerwise_module = ModuleType(
            "sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload"
        )
        configure = mock.Mock(return_value=["transformer", "text_encoder"])
        layerwise_module.configure_layerwise_offload_modules = configure

        with (
            mock.patch.dict(
                sys.modules,
                {
                    pipeline_configs.__name__: pipeline_configs,
                    server_args_module.__name__: server_args_module,
                    hf_utils.__name__: hf_utils,
                    layerwise_module.__name__: layerwise_module,
                },
            ),
            mock.patch(
                f"{model_module}._initialize_single_gpu_runtime",
                return_value=torch.device("cpu"),
            ),
            mock.patch(
                f"{model_module}._distributed_runtime_contract",
                return_value=("local", True),
            ),
            mock.patch.object(
                _ComponentSet,
                "_load_component",
                side_effect=lambda name: SimpleNamespace(name=name),
            ),
            mock.patch(
                "pathlib.Path.open",
                mock.mock_open(read_data="{}"),
            ),
        ):
            components = _ComponentSet(
                load_config=ModelLoadConfig(
                    model_path="unused",
                    text_encoder_cpu_offload=True,
                    dit_cpu_offload=True,
                    vae_cpu_offload=True,
                ),
                component_names=("transformer", "text_encoder", "vae"),
            )

        self.assertEqual(
            captured["layerwise_offload_components"], ["text_encoder", "dit"]
        )
        self.assertFalse(captured["text_encoder_cpu_offload"])
        self.assertFalse(captured["dit_cpu_offload"])
        self.assertFalse(captured["use_fsdp_inference"])
        self.assertTrue(captured["vae_cpu_offload"])
        configure.assert_called_once_with(
            components.modules,
            components.server_args,
            component_names=["text_encoder", "dit"],
        )
        self.assertEqual(
            components.cpu_offload_requested,
            {"text_encoder": True, "dit": True, "vae": True},
        )
        self.assertEqual(
            components.cpu_offload_effective,
            {
                "text_encoder": "layerwise",
                "dit": "layerwise",
                "vae": "per_chunk_module",
            },
        )

    def test_vae_offload_helpers_move_only_weights_and_keep_feature_cache(self):
        class _FakeVae:
            def __init__(self):
                self.moves = []
                self._feat_map = ["gpu-feature-cache"]

            def to(self, *, device, dtype):
                self.moves.append((str(device), dtype))
                return self

        model = object.__new__(SfWanVaeModel)
        model._load_config = SimpleNamespace(vae_cpu_offload=True)
        model.device = torch.device("cuda:0")
        model.vae_dtype = torch.float32
        model.vae = _FakeVae()
        model._vae_weights_on_device = False
        feature_cache = model.vae._feat_map

        model._activate_vae_for_chunk()
        self.assertTrue(model._vae_weights_on_device)
        model._offload_vae_weights()
        self.assertFalse(model._vae_weights_on_device)
        self.assertIs(model.vae._feat_map, feature_cache)
        self.assertEqual(
            model.vae.moves,
            [
                ("cuda:0", torch.float32),
                ("cpu", torch.float32),
            ],
        )

        model._load_config = SimpleNamespace(vae_cpu_offload=False)
        model._activate_vae_for_chunk()
        model._offload_vae_weights()
        self.assertEqual(len(model.vae.moves), 2)

    def test_vae_offload_wraps_every_chunk_outside_decode(self):
        actions = []
        model = object.__new__(SfWanVaeModel)
        model._request_active = True
        model.device = torch.device("cpu")
        model._load_config = SimpleNamespace(
            vae_cpu_offload=True,
            enable_profile=False,
            enable_nvtx=False,
        )
        model._latents_mean = torch.zeros((1, 16, 1, 1, 1))
        model._latents_std = torch.ones((1, 16, 1, 1, 1))

        def _activate(_self):
            actions.append("weights_to_gpu")

        def _offload(_self):
            actions.append("weights_to_cpu")

        def _decode(_self, *, chunk_index, z):
            actions.append(f"decode_{chunk_index}")
            return torch.zeros((1, 3, 9 if chunk_index == 0 else 12, 1, 1)), [], 0.0

        model._activate_vae_for_chunk = MethodType(_activate, model)
        model._offload_vae_weights = MethodType(_offload, model)
        model._decode_per_latent = MethodType(_decode, model)
        latents = torch.zeros((1, 16, 3, 1, 1), dtype=torch.bfloat16)

        model.decode_chunk(chunk_index=0, latents=latents, return_frames=False)
        model.decode_chunk(chunk_index=1, latents=latents, return_frames=False)
        self.assertEqual(
            actions,
            [
                "weights_to_gpu",
                "decode_0",
                "weights_to_cpu",
                "weights_to_gpu",
                "decode_1",
                "weights_to_cpu",
            ],
        )

    def test_decode_failure_offloads_weights_and_next_fcfs_job_runs(self):
        class _FakeVae:
            def __init__(self):
                self.moves = []
                self.reset_count = 0
                self._feat_map = ["feature-cache"]

            def to(self, *, device, dtype):
                self.moves.append((str(device), dtype))
                return self

            def reset_causal_decode_state(self):
                self.reset_count += 1

        model = object.__new__(SfWanVaeModel)
        model._load_config = ModelLoadConfig(
            model_path="unused",
            vae_cpu_offload=True,
        )
        model.device = torch.device("cpu")
        model.vae_dtype = torch.float32
        model.vae = _FakeVae()
        model._vae_weights_on_device = False
        model._request_active = False
        model._latents_mean = torch.zeros((1, 16, 1, 1, 1))
        model._latents_std = torch.ones((1, 16, 1, 1, 1))
        model.contract = {"cpu_offload": {"vae": True}}
        attempts = 0

        def _decode(_self, *, chunk_index, z):
            nonlocal attempts
            del chunk_index, z
            attempts += 1
            if attempts == 1:
                raise RuntimeError("expected decode failure")
            return torch.zeros((1, 3, 9, 1, 1)), [], 0.0

        model._decode_per_latent = MethodType(_decode, model)

        async def _scenario():
            runtime = SfWanRuntime(
                config=ServerConfig(role="vae", vae_cpu_offload=True),
                model_factory=lambda _config: model,
            )
            await runtime.start()
            first, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="offload-decode-fails",
                    height=16,
                    width=16,
                    num_frames=9,
                    source="dit",
                    discard_output=True,
                )
            )
            second, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="offload-after-failure",
                    height=16,
                    width=16,
                    num_frames=9,
                    source="dit",
                    discard_output=True,
                )
            )
            latent = torch.zeros((1, 16, 3, 2, 2), dtype=torch.bfloat16)
            try:
                await first.put_chunk(
                    chunk_index=0,
                    data=latent,
                    digest="first",
                )
                await second.put_chunk(
                    chunk_index=0,
                    data=latent,
                    digest="second",
                )
                await _wait_until_terminal(first, second)
                self.assertEqual(first.state, JobState.FAILED)
                self.assertIn("expected decode failure", first.error)
                self.assertEqual(second.state, JobState.COMPLETED)
                self.assertFalse(model._vae_weights_on_device)
                self.assertEqual(attempts, 2)
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_engine_contract_reports_role_specific_offload_config(self):
        async def _scenario():
            def _factory(load_config):
                return SimpleNamespace(
                    contract={
                        "distributed_backend": "local",
                        "cpu_offload_requested": {
                            "text_encoder": load_config.text_encoder_cpu_offload,
                            "dit": load_config.dit_cpu_offload,
                        },
                        "cpu_offload_effective": {
                            "text_encoder": "resident",
                            "dit": "layerwise",
                        },
                        "cpu_offload": {
                            "text_encoder": load_config.text_encoder_cpu_offload,
                            "dit": load_config.dit_cpu_offload,
                        },
                    }
                )

            runtime = SfWanRuntime(
                config=ServerConfig(
                    role="dit",
                    text_encoder_cpu_offload=False,
                    dit_cpu_offload=True,
                    vae_cpu_offload=True,
                ),
                model_factory=_factory,
            )
            await runtime.start()
            try:
                contract = runtime.engine_status().contract
                self.assertEqual(
                    contract["cpu_offload"],
                    {
                        "text_encoder": False,
                        "dit": True,
                    },
                )
                self.assertNotIn(
                    "vae",
                    contract["cpu_offload"],
                )
                self.assertEqual(contract["distributed_backend"], "local")
                self.assertEqual(
                    contract["cpu_offload_requested"],
                    {
                        "text_encoder": False,
                        "dit": True,
                    },
                )
                self.assertEqual(
                    contract["cpu_offload_effective"],
                    {
                        "text_encoder": "resident",
                        "dit": "layerwise",
                    },
                )
            finally:
                await runtime.close()

        asyncio.run(_scenario())


class TestSfWanProfilingGate(CustomTestCase):
    """Detailed timers must be a true no-op unless explicitly enabled."""

    def test_disabled_dit_step_metrics_are_not_constructed(self):
        class _Scheduler:
            timesteps = torch.tensor(
                [1000.0, 937.5, 833.3333, 625.0],
                dtype=torch.float32,
            )
            sigmas = torch.tensor(
                [1.0, 0.9375, 0.8333333, 0.625],
                dtype=torch.float32,
            )

            @staticmethod
            def add_noise(_original_samples, noise, _timestep):
                return noise

        model = object.__new__(SfWanDitModel)
        model.device = torch.device("cpu")
        model.target_dtype = torch.bfloat16
        model.scheduler = _Scheduler()
        model.pipeline_config = SimpleNamespace(context_noise=0)
        model._load_config = SimpleNamespace(
            enable_profile=False,
            enable_nvtx=False,
        )
        forward_calls = []

        def _forward(_self, **kwargs):
            forward_calls.append(kwargs["label"])
            return torch.zeros_like(
                kwargs["latent"],
                dtype=torch.bfloat16,
            ), None

        model._forward_transformer = MethodType(_forward, model)
        clean, metrics = model._denoise_chunk(
            chunk=torch.randn((1, 16, 3, 2, 2), dtype=torch.float32),
            prompt_embeds=torch.zeros((1, 512, 4096), dtype=torch.float32),
            timesteps=model.scheduler.timesteps,
            generator=torch.Generator(device="cpu").manual_seed(1024),
            current_start_tokens=0,
            start_frame=0,
        )
        self.assertEqual(len(forward_calls), 5)
        self.assertEqual(metrics, {})
        self.assertEqual(clean.dtype, torch.bfloat16)

    def test_disabled_timer_calls_function_without_clock_or_cuda_event(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda,
                "Event",
                side_effect=AssertionError("timing Event must not be created"),
            ),
            mock.patch.object(
                time,
                "perf_counter",
                side_effect=AssertionError("wall clock must not be read"),
            ),
        ):
            result, elapsed = _timed_cuda_call(
                label="disabled",
                enabled_profile=False,
                enabled_nvtx=False,
                function=lambda: "result",
            )
        self.assertEqual(result, "result")
        self.assertIsNone(elapsed)

    def test_nvtx_only_does_not_enable_timing(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda,
                "Event",
                side_effect=AssertionError("timing Event must not be created"),
            ),
            mock.patch.object(torch.cuda.nvtx, "range_push") as push,
            mock.patch.object(torch.cuda.nvtx, "range_pop") as pop,
        ):
            result, elapsed = _timed_cuda_call(
                label="nvtx-only",
                enabled_profile=False,
                enabled_nvtx=True,
                function=lambda: 7,
            )
        self.assertEqual(result, 7)
        self.assertIsNone(elapsed)
        push.assert_called_once_with("nvtx-only")
        pop.assert_called_once_with()

    def test_ready_event_remains_a_correctness_barrier_without_timing(self):
        ready = mock.Mock()
        staged = StagedDeviceChunk(
            tensor="device-latent",
            ready_event=ready,
            start_event=None,
        )
        self.assertEqual(staged.completed_metrics(), {})
        ready.synchronize.assert_not_called()
        staged.release()
        ready.synchronize.assert_called_once_with()


class TestSfWanWireProtocol(CustomTestCase):
    """Guard the external BF16/BCTHW safetensors contract."""

    def test_bfloat16_roundtrip_and_shape_rejection(self):
        expected_shape = (1, 16, 3, 4, 6)
        source = torch.arange(
            math.prod(expected_shape),
            dtype=torch.float32,
        ).reshape(expected_shape)
        payload = serialize_latent_tensor(source)
        restored = deserialize_latent_tensor(
            payload,
            expected_shape=expected_shape,
        )
        self.assertEqual(restored.dtype, torch.bfloat16)
        torch.testing.assert_close(restored, source.to(torch.bfloat16))

        with self.assertRaises(ValueError):
            deserialize_latent_tensor(
                payload,
                expected_shape=(1, 16, 3, 6, 4),
            )
        with self.assertRaisesRegex(ValueError, "valid safetensors"):
            deserialize_latent_tensor(
                b"not-safetensors",
                expected_shape=expected_shape,
            )

    def test_header_only_parse_and_direct_cpu_storage_copy(self):
        expected_shape = (1, 16, 3, 2, 4)
        source = (
            torch.arange(
                math.prod(expected_shape),
                dtype=torch.float32,
            )
            .reshape(expected_shape)
            .to(torch.bfloat16)
        )
        payload = serialize_latent_tensor(source)

        with mock.patch(
            "safetensors.torch.load",
            side_effect=AssertionError("header-only ingress must not load a tensor"),
        ):
            parsed = parse_latent_safetensors_payload(
                payload,
                expected_shape=expected_shape,
            )
            destination = torch.empty(expected_shape, dtype=torch.bfloat16)
            _copy_payload_data_to_pinned(parsed, destination)

        self.assertIs(parsed.payload, payload)
        data_view = parsed.data_view()
        try:
            self.assertIs(data_view.obj, payload)
            expected_bytes = data_view.tobytes()
        finally:
            data_view.release()
        actual_bytes = memoryview(
            destination.view(torch.uint8).reshape(-1).numpy()
        ).tobytes()
        self.assertEqual(actual_bytes, expected_bytes)
        torch.testing.assert_close(destination, source)

    def test_header_only_parser_rejects_ambiguous_or_malformed_documents(self):
        expected_shape = (1, 16, 3, 2, 4)
        data_nbytes = math.prod(expected_shape) * 2
        data = bytes(data_nbytes)
        tensor_header = {
            "dtype": "BF16",
            "shape": list(expected_shape),
            "data_offsets": [0, data_nbytes],
        }
        valid = _build_safetensors_document({"latents": tensor_header}, data)
        parsed = parse_latent_safetensors_payload(
            valid,
            expected_shape=expected_shape,
        )
        self.assertEqual(parsed.data_nbytes, data_nbytes)

        invalid_documents = {
            "truncated header": valid[:10],
            "oversized header": (
                (MAX_SAFETENSORS_HEADER_BYTES + 1).to_bytes(8, "little")
            ),
            "extra tensor": _build_safetensors_document(
                {
                    "latents": tensor_header,
                    "other": tensor_header,
                },
                data,
            ),
            "wrong dtype": _build_safetensors_document(
                {
                    "latents": {
                        **tensor_header,
                        "dtype": "F16",
                    }
                },
                data,
            ),
            "wrong shape": _build_safetensors_document(
                {
                    "latents": {
                        **tensor_header,
                        "shape": [1, 16, 3, 4, 2],
                    }
                },
                data,
            ),
            "hole": _build_safetensors_document(
                {
                    "latents": {
                        **tensor_header,
                        "data_offsets": [1, data_nbytes + 1],
                    }
                },
                data + b"\0",
            ),
            "trailing data": valid + b"\0",
            "invalid metadata": _build_safetensors_document(
                {
                    "__metadata__": {"source": 1},
                    "latents": tensor_header,
                },
                data,
            ),
        }
        for label, document in invalid_documents.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_latent_safetensors_payload(
                    document,
                    expected_shape=expected_shape,
                )

        tensor_json = json.dumps(tensor_header, separators=(",", ":"))
        duplicate_header = (
            f'{{"latents":{tensor_json},"latents":{tensor_json}}}'.encode()
        )
        duplicate_header += b" " * (-len(duplicate_header) % 8)
        duplicate_document = (
            len(duplicate_header).to_bytes(8, "little") + duplicate_header + data
        )
        with self.assertRaises(ValueError):
            parse_latent_safetensors_payload(
                duplicate_document,
                expected_shape=expected_shape,
            )

    def test_vae_chunk_idempotency_and_reordering(self):
        async def _scenario():
            record = VaeJobRecord(
                spec=LatentJobSpec(
                    request_id="wire-order",
                    num_frames=81,
                    discard_output=True,
                )
            )
            self.assertTrue(
                await record.put_chunk(
                    chunk_index=1,
                    data="second",
                    digest="digest-1",
                )
            )
            self.assertTrue(
                await record.put_chunk(
                    chunk_index=0,
                    data="first",
                    digest="digest-0",
                )
            )
            self.assertFalse(
                await record.put_chunk(
                    chunk_index=1,
                    data="ignored-duplicate",
                    digest="digest-1",
                )
            )
            first = await record.wait_for_chunk(
                chunk_index=0,
                timeout_seconds=1,
            )
            second = await record.wait_for_chunk(
                chunk_index=1,
                timeout_seconds=1,
            )
            self.assertEqual((first.data, second.data), ("first", "second"))
            self.assertFalse(
                await record.put_chunk(
                    chunk_index=1,
                    data="duplicate-after-consume",
                    digest="digest-1",
                )
            )
            self.assertTrue(
                await record.put_chunk(
                    chunk_index=2,
                    data="original",
                    digest="original",
                )
            )
            with self.assertRaises(ValueError):
                await record.put_chunk(
                    chunk_index=2,
                    data="changed",
                    digest="changed",
                )
            record.mark_failed("terminal retry test")
            self.assertFalse(
                await record.put_chunk(
                    chunk_index=2,
                    data="same-terminal-retry",
                    digest="original",
                )
            )
            with self.assertRaisesRegex(ValueError, "new chunk"):
                await record.put_chunk(
                    chunk_index=3,
                    data="late-new-content",
                    digest="late-new-content",
                )

        asyncio.run(_scenario())

    def test_http_duplicate_is_idempotent_and_conflict_is_409(self):
        fake = _FakeVae()
        app = create_app(
            config=ServerConfig(role="vae"),
            model_factory=lambda _config: fake,
        )
        spec = LatentJobSpec(
            request_id="http-idempotency",
            height=16,
            width=16,
            num_frames=9,
            source="dit",
            discard_output=True,
        )
        first_payload = serialize_latent_tensor(
            torch.zeros(spec.latent_chunk_shape, dtype=torch.bfloat16)
        )
        conflicting_payload = serialize_latent_tensor(
            torch.ones(spec.latent_chunk_shape, dtype=torch.bfloat16)
        )
        path = f"/v1/latent-jobs/{spec.request_id}/chunks/0"
        headers = {"content-type": "application/x-safetensors"}

        with mock.patch(
            "safetensors.torch.load",
            side_effect=AssertionError("HTTP ingress must not load a pageable tensor"),
        ):
            with TestClient(app) as client:
                registration = client.post(
                    "/v1/latent-jobs",
                    json=spec.model_dump(mode="json"),
                )
                self.assertEqual(registration.status_code, 202)
                first = client.put(path, content=first_payload, headers=headers)
                duplicate = client.put(path, content=first_payload, headers=headers)
                conflict = client.put(
                    path,
                    content=conflicting_payload,
                    headers=headers,
                )

        self.assertEqual(first.status_code, 202)
        self.assertTrue(first.json()["accepted"])
        self.assertEqual(duplicate.status_code, 202)
        self.assertFalse(duplicate.json()["accepted"])
        self.assertEqual(conflict.status_code, 409)


class TestSfWanTransportState(CustomTestCase):
    """Exercise transport layout and event lifetime without a CUDA device."""

    def test_http_body_is_staged_only_when_the_running_job_needs_it(self):
        class _RecordingH2DLoader:
            def __init__(self):
                self.payloads = []
                self.closed = False

            async def stage(self, payload):
                self.payloads.append(payload)
                return StagedDeviceChunk(tensor="device-from-pinned")

            def close(self):
                self.closed = True

        async def _scenario():
            fake = _FakeVae()
            runtime = SfWanRuntime(
                config=ServerConfig(role="vae"),
                model_factory=lambda _config: fake,
            )
            await runtime.start()
            loader = _RecordingH2DLoader()
            runtime._h2d_loader = loader
            spec = LatentJobSpec(
                request_id="header-only-http-ingress",
                height=16,
                width=16,
                num_frames=9,
                source="dit",
                discard_output=True,
            )
            record, _, _ = await runtime.register_latent_job(spec)
            payload = serialize_latent_tensor(
                torch.zeros(spec.latent_chunk_shape, dtype=torch.bfloat16)
            )
            try:
                with mock.patch(
                    "safetensors.torch.load",
                    side_effect=AssertionError(
                        "running HTTP ingress must not load a pageable tensor"
                    ),
                ):
                    accepted = await runtime.put_latent_chunk(
                        request_id=spec.request_id,
                        chunk_index=0,
                        payload=payload,
                    )
                    await _wait_until_terminal(record)
                self.assertTrue(accepted)
                self.assertEqual(record.state, JobState.COMPLETED)
                self.assertEqual(fake.decode_order, ["device-from-pinned"])
                self.assertEqual(len(loader.payloads), 1)
                self.assertIsInstance(
                    loader.payloads[0],
                    SafetensorsLatentPayload,
                )
                self.assertIs(loader.payloads[0].payload, payload)
            finally:
                await runtime.close()
            self.assertTrue(loader.closed)

        asyncio.run(_scenario())

    def test_waiting_http_job_does_not_reserve_a_pinned_stage(self):
        class _NamedH2DLoader:
            def __init__(self):
                self.names_by_body_id = {}
                self.staged_names = []

            async def stage(self, payload):
                name = self.names_by_body_id[id(payload.payload)]
                self.staged_names.append(name)
                return StagedDeviceChunk(tensor=name)

            def close(self):
                return None

        async def _scenario():
            fake = _FakeVae()
            runtime = SfWanRuntime(
                config=ServerConfig(role="vae"),
                model_factory=lambda _config: fake,
            )
            await runtime.start()
            loader = _NamedH2DLoader()
            runtime._h2d_loader = loader
            first_spec = LatentJobSpec(
                request_id="running-http-job",
                height=16,
                width=16,
                num_frames=9,
                source="dit",
                discard_output=True,
            )
            second_spec = LatentJobSpec(
                request_id="waiting-http-job",
                height=16,
                width=16,
                num_frames=9,
                source="dit",
                discard_output=True,
            )
            first, _, _ = await runtime.register_latent_job(first_spec)
            second, _, _ = await runtime.register_latent_job(second_spec)
            first_body = serialize_latent_tensor(
                torch.zeros(first_spec.latent_chunk_shape, dtype=torch.bfloat16)
            )
            second_body = serialize_latent_tensor(
                torch.ones(second_spec.latent_chunk_shape, dtype=torch.bfloat16)
            )
            loader.names_by_body_id = {
                id(first_body): "first-device",
                id(second_body): "second-device",
            }
            try:
                await runtime.put_latent_chunk(
                    request_id=second_spec.request_id,
                    chunk_index=0,
                    payload=second_body,
                )
                for _ in range(100):
                    if runtime.engine_status().running_ids == [first_spec.request_id]:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(
                    runtime.engine_status().running_ids,
                    [first_spec.request_id],
                )
                self.assertEqual(loader.staged_names, [])

                await runtime.put_latent_chunk(
                    request_id=first_spec.request_id,
                    chunk_index=0,
                    payload=first_body,
                )
                await _wait_until_terminal(first, second)
                self.assertEqual(
                    loader.staged_names,
                    ["first-device", "second-device"],
                )
                self.assertEqual(
                    fake.decode_order,
                    ["first-device", "second-device"],
                )
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_shm_cleanup_releases_mapping_but_keeps_idempotency_descriptor(self):
        async def _scenario():
            runtime = SfWanRuntime(config=ServerConfig(role="vae"))
            region = mock.Mock()
            runtime._shm_regions["terminal-job"] = region
            runtime._shm_descriptors["terminal-job"] = mock.Mock()
            try:
                runtime._cleanup_shm_region("terminal-job")
                self.assertEqual(runtime._shm_regions, {})
                self.assertIn("terminal-job", runtime._shm_descriptors)
                region.close.assert_called_once_with()
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_terminal_shm_ready_retry_remains_idempotent_without_mapping(self):
        async def _scenario():
            server_module = "sglang.multimodal_gen.experimental.jetson_sfwan.server"
            with mock.patch(f"{server_module}.os.name", "posix"):
                runtime = SfWanRuntime(
                    config=ServerConfig(
                        role="vae",
                        latent_transport="shm",
                    )
                )
            spec = LatentJobSpec(
                request_id="terminal-shm-retry",
                height=16,
                width=16,
                num_frames=9,
                transport="shm",
            )
            record = VaeJobRecord(spec=spec)
            digest = "b" * 64
            await record.put_chunk(
                chunk_index=0,
                data="already-consumed",
                digest=digest,
            )
            runtime.jobs[spec.request_id] = record
            descriptor = build_shared_memory_descriptor(
                name="sfwan-terminal",
                lease_token="a" * 32,
                shape=spec.latent_chunk_shape,
                total_chunks=1,
            )
            runtime._shm_descriptors[spec.request_id] = descriptor
            try:
                accepted = await runtime.ready_shared_memory_chunk(
                    request_id=spec.request_id,
                    chunk_index=0,
                    ready=SharedMemoryChunkReady(
                        lease_token=descriptor.lease_token,
                        digest=digest,
                    ),
                )
                self.assertFalse(accepted)
                self.assertEqual(runtime._shm_regions, {})
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_shared_memory_layout_is_page_aligned_and_non_overlapping(self):
        descriptor = build_shared_memory_descriptor(
            name="sfwan-test",
            lease_token="0" * 32,
            shape=(1, 16, 3, 60, 104),
            total_chunks=7,
            page_size=4096,
        )
        self.assertEqual(descriptor.tensor_nbytes, 599040)
        self.assertEqual(descriptor.data_offset % 4096, 0)
        self.assertEqual(descriptor.chunk_stride % 4096, 0)
        self.assertGreaterEqual(descriptor.chunk_stride, descriptor.tensor_nbytes)
        self.assertEqual(
            descriptor.total_bytes,
            descriptor.data_offset + descriptor.chunk_stride * 7,
        )
        self.assertLess(len(_shared_memory_header(descriptor)), descriptor.data_offset)
        changed_lease = descriptor.model_copy(update={"lease_token": "1" * 32})
        self.assertNotEqual(
            _shared_memory_header(descriptor),
            _shared_memory_header(changed_lease),
        )

    def test_staged_chunk_waits_for_local_event_before_release(self):
        order = []

        class _Event:
            def synchronize(self):
                order.append("event")

        staged = StagedDeviceChunk(
            tensor="gpu",
            ready_event=_Event(),
            release_callback=lambda: order.append("release"),
        )
        staged.release()
        staged.release()
        self.assertEqual(order, ["event", "release"])

    def test_staged_chunk_records_compute_stream_lifetime(self):
        order = []

        class _Tensor:
            device = "cuda:0"

            def record_stream(self, stream):
                order.append(("record_stream", stream))

        class _Stream:
            def wait_event(self, event):
                order.append(("wait_event", event))

        tensor = _Tensor()
        stream = _Stream()
        ready = object()
        staged = StagedDeviceChunk(tensor=tensor, ready_event=ready)
        with mock.patch.object(
            torch.cuda,
            "current_stream",
            return_value=stream,
        ):
            returned = staged.wait_on_current_stream()
        self.assertIs(returned, tensor)
        self.assertEqual(
            order,
            [
                ("wait_event", ready),
                ("record_stream", stream),
            ],
        )

    def test_shm_ready_control_waits_for_d2h_and_sends_no_cuda_event(self):
        async def _scenario():
            order = []
            descriptor = build_shared_memory_descriptor(
                name="sfwan-test",
                lease_token="a" * 32,
                shape=(1, 16, 3, 2, 2),
                total_chunks=1,
                page_size=4096,
            )

            class _Region:
                def __init__(self):
                    self.descriptor = descriptor

                def digest(self, _chunk_index):
                    order.append("digest")
                    return "b" * 64

            class _Event:
                def synchronize(self):
                    order.append("d2h_event")

                def elapsed_time(self, _other):
                    raise AssertionError(
                        "profile-disabled SHM must not query event timing"
                    )

            class _Response:
                def raise_for_status(self):
                    return None

            class _Client:
                async def post(self, _url, **kwargs):
                    order.append("ready_control")
                    body = kwargs["json"]
                    if "cuda_event" in body:
                        raise AssertionError(
                            "CUDA events must not cross the process boundary"
                        )
                    return _Response()

            async def _event_callback(_request_id, kind, _details):
                order.append(kind)

            async def _failure_callback(_request_id, _error):
                raise AssertionError("unexpected SHM transfer failure")

            spec = LatentJobSpec(
                request_id="shm-order",
                height=16,
                width=16,
                num_frames=9,
                transport="shm",
            )
            region = _Region()
            sender = SharedMemoryChunkSender(
                vae_url="http://vae",
                queue_depth=2,
                event_callback=_event_callback,
                failure_callback=_failure_callback,
                client=_Client(),
            )
            state = _SharedTransferState(spec=spec, region=region)
            event = _Event()
            state.pending_copy_events[0] = event
            state.pending_sources[0] = "gpu-source"
            sender._states[spec.request_id] = state
            await sender._send_chunk(
                _SharedPendingChunk(
                    request_id=spec.request_id,
                    chunk_index=0,
                    source_tensor="gpu",
                    copy_start_event=event,
                    ready_event=event,
                )
            )
            self.assertEqual(
                order[:3],
                ["d2h_event", "digest", "ready_control"],
            )
            self.assertTrue(state.done.is_set())
            self.assertEqual(state.pending_copy_events, {})
            self.assertEqual(state.pending_sources, {})

        asyncio.run(_scenario())

    def test_shm_release_retains_source_until_d2h_event_completes(self):
        order = []
        spec = LatentJobSpec(
            request_id="shm-cancel-lifetime",
            height=16,
            width=16,
            num_frames=9,
            transport="shm",
        )

        class _Region:
            def close(self):
                order.append("region_close")

        state = _SharedTransferState(spec=spec, region=_Region())

        test_case = self

        class _Event:
            def synchronize(self):
                self_source = state.pending_sources.get(0)
                test_case.assertIsNotNone(self_source)
                order.append("event")

        state.pending_copy_events[0] = _Event()
        state.pending_sources[0] = "gpu-source"
        sender = SharedMemoryChunkSender(
            vae_url="http://vae",
            queue_depth=1,
            event_callback=mock.AsyncMock(),
            failure_callback=mock.AsyncMock(),
        )
        sender._states[spec.request_id] = state
        sender.release_job(spec.request_id)
        self.assertEqual(order, ["event", "region_close"])
        self.assertEqual(state.pending_copy_events, {})
        self.assertEqual(state.pending_sources, {})


class TestSfWanArrivalSchedules(CustomTestCase):
    """Pin deterministic client arrival semantics independently of the server."""

    def test_arrival_modes(self):
        self.assertEqual(
            build_interarrival_delays(
                count=4,
                mode="burst",
                seed=0,
            ),
            [0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            build_interarrival_delays(
                count=3,
                mode="fixed",
                seed=0,
                fixed_interval_seconds=2.5,
            ),
            [0.0, 2.5, 2.5],
        )
        poisson_a = build_interarrival_delays(
            count=5,
            mode="poisson",
            seed=7,
            poisson_lambda=0.5,
        )
        poisson_b = build_interarrival_delays(
            count=5,
            mode="poisson",
            seed=7,
            poisson_lambda=0.5,
        )
        self.assertEqual(poisson_a, poisson_b)
        with mock.patch.object(
            random.Random,
            "expovariate",
            autospec=True,
            return_value=2.0,
        ) as expovariate:
            sampled = build_interarrival_delays(
                count=3,
                mode="poisson",
                seed=9,
                poisson_lambda=0.25,
            )
        self.assertEqual(sampled, [0.0, 2.0, 2.0])
        self.assertEqual(
            [call.args[1] for call in expovariate.call_args_list],
            [0.25, 0.25],
        )

    def test_cli_exposes_lambda_and_rejects_removed_uniform_mode(self):
        parser = _build_parser()
        parsed = parser.parse_args(
            [
                "generate",
                "--server-url",
                "http://server",
                "--arrival-mode",
                "poisson",
                "--poisson-lambda",
                "0.2",
            ]
        )
        self.assertEqual(parsed.poisson_lambda, 0.2)
        with (
            mock.patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(
                [
                    "generate",
                    "--server-url",
                    "http://server",
                    "--arrival-mode",
                    "uniform",
                ]
            )
        with (
            mock.patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(
                [
                    "generate",
                    "--server-url",
                    "http://server",
                    "--poisson-rate",
                    "0.2",
                ]
            )

    def test_scheduled_posts_are_concurrent(self):
        async def _scenario():
            release = asyncio.Event()
            all_started = asyncio.Event()

            class _Response:
                def __init__(self, request_id):
                    self._request_id = request_id

                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "request_id": self._request_id,
                        "status": "waiting",
                        "status_url": f"http://server/jobs/{self._request_id}",
                        "result_url": (f"http://server/jobs/{self._request_id}/result"),
                        "warnings": [],
                    }

            class _Client:
                def __init__(self):
                    self.active = 0
                    self.max_active = 0
                    self.count = 0

                async def post(self, _url, **_kwargs):
                    self.count += 1
                    request_id = f"request-{self.count}"
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    if self.active == 3:
                        all_started.set()
                    await release.wait()
                    self.active -= 1
                    return _Response(request_id)

            client = _Client()
            start_time = time.perf_counter()
            tasks = [
                asyncio.create_task(
                    _post_at_offset(
                        client=client,
                        url="http://server",
                        generation=GenerationRequest(prompt=f"request {index}"),
                        offset_seconds=0,
                        start_time=start_time,
                    )
                )
                for index in range(3)
            ]
            await asyncio.wait_for(all_started.wait(), timeout=1)
            self.assertEqual(client.max_active, 3)
            release.set()
            results = await asyncio.gather(*tasks)
            self.assertEqual(len(results), 3)

        asyncio.run(_scenario())


class TestSfWanProfileClient(CustomTestCase):
    """Profile summaries contain all chunks but only execution measurements."""

    def test_aggregated_measurement_view_strips_output_and_diagnostics(self):
        execution = {"component": "vae", "chunks": []}
        view = _profile_measurement_view(
            {
                "iteration": 2,
                "warmup": False,
                "request_id": "profile-id",
                "state": "completed",
                "error": None,
                "profile_execution": execution,
                "_saved_output_path": "video.mp4",
                "transfers": [{"upload_rtt_ms": 9.0}],
                "status": {"metrics": {"queue_wait_ms": 7.0}},
            }
        )
        self.assertEqual(
            view,
            {
                "iteration": 2,
                "warmup": False,
                "request_id": "profile-id",
                "state": "completed",
                "error": None,
                "profile_execution": execution,
            },
        )

    def test_vae_profile_uploads_all_seven_chunks_and_hides_transfer_metrics(self):
        async def _scenario():
            chunk_profile = [
                {
                    "chunk_index": chunk_index,
                    "chunk_execution_cuda_ms": float(chunk_index + 1),
                    "decoded_rgb_frames": decoded_frames_for_chunk(chunk_index),
                }
                for chunk_index in range(7)
            ]
            profile_execution = {
                "component": "vae",
                "num_chunks": 7,
                "chunks": chunk_profile,
                "vae_execution_cuda_ms": 28.0,
            }

            class _Response:
                status_code = 200

                def __init__(self, body):
                    self._body = body

                def raise_for_status(self):
                    return None

                def json(self):
                    return self._body

            class _Client:
                def __init__(self):
                    self.put_chunk_ids = []

                async def post(self, _url, **_kwargs):
                    return _Response({"created": True})

                async def put(self, url, **_kwargs):
                    self.put_chunk_ids.append(int(url.rsplit("/", 1)[-1]))
                    return _Response({"accepted": True})

                async def get(self, _url):
                    return _Response(
                        {
                            "state": "completed",
                            "error": None,
                            "metrics": {
                                "profile_execution": profile_execution,
                                "queue_wait_ms": 99.0,
                                "chunks": [
                                    {
                                        "transfer": {
                                            "h2d_ms": 42.0,
                                            "upload_rtt_ms": 43.0,
                                        }
                                    }
                                ],
                            },
                        }
                    )

            client = _Client()
            result = await _run_profile_iteration(
                client=client,
                server_url="http://vae",
                args=SimpleNamespace(
                    height=16,
                    width=16,
                    num_frames=81,
                    fps=16,
                    save_output=False,
                    seed=1024,
                    poll_interval_seconds=0.0,
                    output_dir="unused",
                ),
                warmup=False,
                iteration=3,
            )
            self.assertEqual(client.put_chunk_ids, list(range(7)))
            self.assertEqual(
                set(result),
                {
                    "iteration",
                    "warmup",
                    "request_id",
                    "state",
                    "error",
                    "profile_execution",
                },
            )
            self.assertEqual(
                [
                    chunk["chunk_index"]
                    for chunk in result["profile_execution"]["chunks"]
                ],
                list(range(7)),
            )
            self.assertNotIn("transfers", result)
            self.assertNotIn("status", result)
            self.assertNotIn("queue_wait_ms", result)

        asyncio.run(_scenario())

    def test_dit_profile_summary_contains_seven_chunks_without_submit_rtt(self):
        async def _scenario():
            profile_execution = {
                "component": "dit",
                "num_chunks": 7,
                "chunks": [
                    {
                        "chunk_index": chunk_index,
                        "denoise_steps": [
                            {"step_index": step_index} for step_index in range(4)
                        ],
                        "clean_kv_cuda_ms": 1.0,
                    }
                    for chunk_index in range(7)
                ],
                "dit_execution_wall_ms": 35.0,
            }

            class _Response:
                status_code = 200

                def __init__(self, body):
                    self._body = body

                def raise_for_status(self):
                    return None

                def json(self):
                    return self._body

            class _Client:
                async def post(self, _url, **_kwargs):
                    return _Response(
                        {
                            "request_id": "dit-profile-id",
                            "status": "waiting",
                            "status_url": "http://dit/v1/jobs/dit-profile-id",
                            "result_url": ("http://dit/v1/jobs/dit-profile-id/result"),
                            "warnings": [],
                        }
                    )

                async def get(self, _url):
                    return _Response(
                        {
                            "state": "completed",
                            "error": None,
                            "metrics": {
                                "profile_execution": profile_execution,
                                "queue_wait_ms": 99.0,
                            },
                        }
                    )

            result = await _run_dit_profile_iteration(
                client=_Client(),
                server_url="http://dit",
                args=SimpleNamespace(
                    prompt="profile",
                    height=16,
                    width=16,
                    num_frames=81,
                    duration_seconds=None,
                    fps=16,
                    seed=1024,
                    poll_interval_seconds=0.0,
                ),
                warmup=False,
                iteration=4,
            )
            self.assertEqual(
                set(result),
                {
                    "iteration",
                    "warmup",
                    "request_id",
                    "state",
                    "error",
                    "profile_execution",
                },
            )
            self.assertEqual(
                [
                    chunk["chunk_index"]
                    for chunk in result["profile_execution"]["chunks"]
                ],
                list(range(7)),
            )
            self.assertNotIn("submit_rtt_ms", result)
            self.assertNotIn("status", result)

        asyncio.run(_scenario())


class TestSfWanFcfsEngine(CustomTestCase):
    """Guard FIFO order and the one-running-request invariant."""

    def test_fcfs_and_single_running_slot(self):
        async def _scenario():
            release_first = asyncio.Event()
            started_first = asyncio.Event()
            order = []
            active = 0
            max_active = 0

            async def _handler(record):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                order.append(record.request_id)
                if record.request_id == "a":
                    started_first.set()
                    await release_first.wait()
                await asyncio.sleep(0)
                active -= 1

            engine = SingleWorkerEngine(role="dit", handler=_handler)
            await engine.start()
            records = [
                JobRecord(
                    request_id=name,
                    role="dit",
                    payload=GenerationRequest(prompt=name),
                )
                for name in ("a", "b", "c")
            ]
            try:
                await engine.submit(records[0])
                await started_first.wait()
                await engine.submit(records[1])
                await engine.submit(records[2])
                snapshot = engine.snapshot()
                self.assertEqual(snapshot.running_ids, ["a"])
                self.assertEqual(snapshot.waiting_ids, ["b", "c"])
                release_first.set()
                await _wait_until_terminal(*records)
                self.assertEqual(order, ["a", "b", "c"])
                self.assertEqual(max_active, 1)
                self.assertEqual(
                    [record.queue_sequence for record in records],
                    [0, 1, 2],
                )
            finally:
                await engine.close()

        asyncio.run(_scenario())

    def test_failure_does_not_block_the_next_request(self):
        async def _scenario():
            order = []

            async def _handler(record):
                order.append(record.request_id)
                if record.request_id == "fails":
                    raise RuntimeError("expected fake failure")

            engine = SingleWorkerEngine(role="dit", handler=_handler)
            await engine.start()
            failed = JobRecord(
                request_id="fails",
                role="dit",
                payload=GenerationRequest(prompt="fails"),
            )
            succeeds = JobRecord(
                request_id="succeeds",
                role="dit",
                payload=GenerationRequest(prompt="succeeds"),
            )
            try:
                await engine.submit(failed)
                await engine.submit(succeeds)
                await _wait_until_terminal(failed, succeeds)
                self.assertEqual(order, ["fails", "succeeds"])
                self.assertEqual(failed.state, JobState.FAILED)
                self.assertEqual(succeeds.state, JobState.COMPLETED)
            finally:
                await engine.close()

        asyncio.run(_scenario())

    def test_close_wakes_a_running_vae_waiter(self):
        async def _scenario():
            started = asyncio.Event()

            async def _handler(record):
                started.set()
                await record.wait_for_chunk(
                    chunk_index=0,
                    timeout_seconds=600,
                )

            engine = SingleWorkerEngine(role="vae", handler=_handler)
            record = VaeJobRecord(
                spec=LatentJobSpec(
                    request_id="shutdown-waiter",
                    height=16,
                    width=16,
                    num_frames=9,
                    source="dit",
                    discard_output=True,
                )
            )
            await engine.start()
            await engine.submit(record)
            await started.wait()
            await asyncio.wait_for(engine.close(), timeout=1)
            self.assertEqual(record.state, JobState.CANCELLED)
            self.assertEqual(engine.snapshot().running_ids, [])

        asyncio.run(_scenario())


class _FakeVae:
    def __init__(self):
        self.reset_count = 0
        self.finish_count = 0
        self.decode_order = []

    def reset_request(self):
        self.reset_count += 1

    def finish_request(self):
        self.finish_count += 1

    def decode_chunk(self, *, chunk_index, latents, return_frames=True):
        self.decode_order.append(latents)
        frame_count = decoded_frames_for_chunk(chunk_index)
        return DecodedChunk(
            chunk_index=chunk_index,
            frames=(
                torch.zeros(
                    (frame_count, 16, 16, 3),
                    dtype=torch.uint8,
                )
                if return_frames
                else None
            ),
            frame_count=frame_count,
            metrics={"fake": True},
        )


class TestSfWanVaeRequestLifecycle(CustomTestCase):
    """A ready later job must not steal the VAE feature-cache owner."""

    def test_partial_reset_failure_still_finishes_and_allows_next_job(self):
        class _ResetFailsOnceVae(_FakeVae):
            def reset_request(self):
                self.reset_count += 1
                if self.reset_count == 1:
                    raise RuntimeError("expected reset failure")

        async def _scenario():
            fake = _ResetFailsOnceVae()
            runtime = SfWanRuntime(
                config=ServerConfig(role="vae"),
                model_factory=lambda _config: fake,
            )
            await runtime.start()
            first, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="reset-fails",
                    height=16,
                    width=16,
                    num_frames=9,
                    source="dit",
                    discard_output=True,
                )
            )
            second, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="after-reset-failure",
                    height=16,
                    width=16,
                    num_frames=9,
                    source="dit",
                    discard_output=True,
                )
            )
            try:
                await second.put_chunk(
                    chunk_index=0,
                    data="after-reset-failure",
                    digest="after-reset-failure",
                )
                await _wait_until_terminal(first, second)
                self.assertEqual(first.state, JobState.FAILED)
                self.assertIn("reset failure", first.error)
                self.assertEqual(second.state, JobState.COMPLETED)
                self.assertEqual(fake.reset_count, 2)
                self.assertEqual(fake.finish_count, 2)
                self.assertEqual(
                    fake.decode_order,
                    ["after-reset-failure"],
                )
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_profile_decodes_all_chunks_with_first_then_steady_frame_counts(self):
        class _ProfileVae:
            def __init__(self):
                self.reset_count = 0
                self.finish_count = 0
                self.decode_order = []

            def reset_request(self):
                self.reset_count += 1

            def finish_request(self):
                self.finish_count += 1

            def decode_chunk(
                self,
                *,
                chunk_index,
                latents,
                return_frames=True,
            ):
                del return_frames
                self.decode_order.append((chunk_index, latents))
                frame_counts = [1, 4, 4] if chunk_index == 0 else [4, 4, 4]
                execution = {
                    "chunk_index": chunk_index,
                    "chunk_execution_cuda_ms": float(chunk_index + 1),
                    "post_quant_cuda_ms": 0.25,
                    "latent_frames": [
                        {
                            "latent_index": latent_index,
                            "cuda_ms": 0.5,
                            "decoded_rgb_frames": frame_count,
                        }
                        for latent_index, frame_count in enumerate(frame_counts)
                    ],
                    "decoded_rgb_frames": sum(frame_counts),
                }
                return DecodedChunk(
                    chunk_index=chunk_index,
                    frames=None,
                    frame_count=sum(frame_counts),
                    metrics={"profile_execution": execution},
                )

        async def _scenario():
            fake = _ProfileVae()
            runtime = SfWanRuntime(
                config=ServerConfig(role="vae", enable_profile=True),
                model_factory=lambda _config: fake,
            )
            await runtime.start()
            record, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="vae-seven-chunk-profile",
                    height=16,
                    width=16,
                    num_frames=81,
                    source="profile",
                    discard_output=True,
                )
            )
            try:
                for chunk_index in range(7):
                    await record.put_chunk(
                        chunk_index=chunk_index,
                        data=f"latent-{chunk_index}",
                        digest=f"digest-{chunk_index}",
                    )
                await _wait_until_terminal(record)
                self.assertEqual(record.state, JobState.COMPLETED)
                self.assertEqual(
                    [item[0] for item in fake.decode_order],
                    list(range(7)),
                )
                self.assertEqual(fake.reset_count, 1)
                self.assertEqual(fake.finish_count, 1)
                execution = record.metrics["profile_execution"]
                self.assertEqual(execution["num_chunks"], 7)
                self.assertEqual(
                    [chunk["chunk_index"] for chunk in execution["chunks"]],
                    list(range(7)),
                )
                self.assertEqual(
                    [chunk["decoded_rgb_frames"] for chunk in execution["chunks"]],
                    [9, 12, 12, 12, 12, 12, 12],
                )
                self.assertEqual(execution["vae_execution_cuda_ms"], 28.0)
                self.assertEqual(record.metrics["decoded_rgb_frames"], 81)
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_request_scoped_cache_lifecycle_is_fcfs(self):
        async def _scenario():
            fake = _FakeVae()
            runtime = SfWanRuntime(
                config=ServerConfig(role="vae"),
                model_factory=lambda _config: fake,
            )
            await runtime.start()
            first, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="first",
                    height=16,
                    width=16,
                    num_frames=21,
                    source="dit",
                    discard_output=True,
                )
            )
            second, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="second",
                    height=16,
                    width=16,
                    num_frames=21,
                    source="dit",
                    discard_output=True,
                )
            )
            try:
                await second.put_chunk(
                    chunk_index=0,
                    data="second-0",
                    digest="second-0",
                )
                await second.put_chunk(
                    chunk_index=1,
                    data="second-1",
                    digest="second-1",
                )
                for _ in range(100):
                    if runtime.engine_status().running_ids == ["first"]:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(
                    runtime.engine_status().running_ids,
                    ["first"],
                )
                self.assertEqual(fake.decode_order, [])

                await first.put_chunk(
                    chunk_index=0,
                    data="first-0",
                    digest="first-0",
                )
                await first.put_chunk(
                    chunk_index=1,
                    data="first-1",
                    digest="first-1",
                )
                await _wait_until_terminal(first, second)
                self.assertEqual(
                    fake.decode_order,
                    ["first-0", "first-1", "second-0", "second-1"],
                )
                self.assertEqual(fake.reset_count, 2)
                self.assertEqual(fake.finish_count, 2)
                self.assertEqual(first.metrics["decoded_rgb_frames"], 21)
                self.assertEqual(first.metrics["num_chunks"], 2)
                self.assertNotIn("steady_chunk_ms", first.metrics)
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_timeout_releases_cache_and_continues_fcfs(self):
        async def _scenario():
            fake = _FakeVae()
            runtime = SfWanRuntime(
                config=ServerConfig(
                    role="vae",
                    chunk_timeout_seconds=0.01,
                ),
                model_factory=lambda _config: fake,
            )
            await runtime.start()
            timed_out, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="timed-out",
                    height=16,
                    width=16,
                    num_frames=9,
                    source="dit",
                    discard_output=True,
                )
            )
            next_job, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="after-timeout",
                    height=16,
                    width=16,
                    num_frames=9,
                    source="dit",
                    discard_output=True,
                )
            )
            try:
                await next_job.put_chunk(
                    chunk_index=0,
                    data="after-timeout",
                    digest="after-timeout",
                )
                await _wait_until_terminal(timed_out, next_job)
                self.assertEqual(timed_out.state, JobState.FAILED)
                self.assertIn("timed out", timed_out.error)
                self.assertEqual(next_job.state, JobState.COMPLETED)
                self.assertEqual(fake.decode_order, ["after-timeout"])
                self.assertEqual(fake.reset_count, 2)
                self.assertEqual(fake.finish_count, 2)
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_profile_dummy_shape_dtype_and_seed(self):
        spec = LatentJobSpec(
            request_id="profile-shape",
            num_frames=81,
            source="profile",
            discard_output=True,
        )
        with mock.patch("torch.randn", wraps=torch.randn) as randn:
            first = _make_dummy_latents(spec=spec, seed=1024)
        second = _make_dummy_latents(spec=spec, seed=1024)
        self.assertEqual(randn.call_args.kwargs["dtype"], torch.float32)
        self.assertEqual(len(first), 7)
        self.assertEqual(tuple(first[0].shape), (1, 16, 3, 60, 104))
        self.assertEqual(first[0].dtype, torch.bfloat16)
        torch.testing.assert_close(first[0], second[0])

    def test_staged_ingress_waits_before_decoder(self):
        async def _scenario():
            actions = []

            class _OrderedFakeVae(_FakeVae):
                def decode_chunk(
                    self,
                    *,
                    chunk_index,
                    latents,
                    return_frames=True,
                ):
                    actions.append(("decode", latents))
                    return super().decode_chunk(
                        chunk_index=chunk_index,
                        latents=latents,
                        return_frames=return_frames,
                    )

            fake = _OrderedFakeVae()
            runtime = SfWanRuntime(
                config=ServerConfig(role="vae"),
                model_factory=lambda _config: fake,
            )
            await runtime.start()
            record, _, _ = await runtime.register_latent_job(
                LatentJobSpec(
                    request_id="staged-ingress",
                    height=16,
                    width=16,
                    num_frames=9,
                    source="dit",
                    discard_output=True,
                )
            )
            staged = StagedDeviceChunk(tensor="device-latent")
            try:
                with mock.patch.object(
                    StagedDeviceChunk,
                    "wait_on_current_stream",
                    autospec=True,
                    side_effect=lambda _self: (
                        actions.append(("wait", "device-latent")) or "device-latent"
                    ),
                ):
                    await record.put_chunk(
                        chunk_index=0,
                        data=staged,
                        digest="staged",
                    )
                    await _wait_until_terminal(record)
                self.assertEqual(
                    actions,
                    [
                        ("wait", "device-latent"),
                        ("decode", "device-latent"),
                    ],
                )
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_monolithic_uses_the_clean_latent_chunk_contract(self):
        class _FakeDit:
            def generate(self, *, request, chunk_callback):
                chunk_callback(
                    LatentChunk(
                        chunk_index=0,
                        tensor="clean-after-kv",
                        metrics={"clean_kv_cuda_ms": 1.0},
                    )
                )
                return {"num_chunks": request.total_chunks}

        fake_vae = _FakeVae()
        monolithic = object.__new__(SfWanMonolithicModel)
        monolithic._load_config = SimpleNamespace(enable_profile=False)
        monolithic.dit = _FakeDit()
        monolithic.vae = fake_vae
        result = monolithic.generate(
            request=GenerationRequest(
                prompt="test",
                height=16,
                width=16,
                num_frames=9,
            )
        )
        self.assertEqual(fake_vae.decode_order, ["clean-after-kv"])
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(
            result.chunks[0].metrics["dit"]["clean_kv_cuda_ms"],
            1.0,
        )
        self.assertEqual(fake_vae.reset_count, 1)
        self.assertEqual(fake_vae.finish_count, 1)


class TestSfWanModeIsolation(CustomTestCase):
    """Ensure role-specific paths do not allocate or call unrelated stages."""

    def test_profile_http_endpoints_return_422_when_disabled(self):
        dit_app = create_app(
            config=ServerConfig(role="dit"),
            model_factory=lambda _config: SimpleNamespace(contract={"role": "dit"}),
        )
        with TestClient(dit_app) as client:
            response = client.post(
                "/v1/dit-profiles",
                json=DitProfileRequest(
                    prompt="profile",
                    height=16,
                    width=16,
                    num_frames=9,
                ).model_dump(mode="json"),
            )
        self.assertEqual(response.status_code, 422)
        self.assertIn("--enable-profile", response.json()["detail"])

        vae_app = create_app(
            config=ServerConfig(role="vae"),
            model_factory=lambda _config: _FakeVae(),
        )
        profile = LatentJobSpec(
            request_id="profile-http-disabled",
            height=16,
            width=16,
            num_frames=9,
            source="profile",
            discard_output=True,
        )
        normal = profile.model_copy(
            update={
                "request_id": "normal-http-enabled",
                "source": "dit",
            }
        )
        with TestClient(vae_app) as client:
            profile_response = client.post(
                "/v1/latent-jobs",
                json=profile.model_dump(mode="json"),
            )
            normal_response = client.post(
                "/v1/latent-jobs",
                json=normal.model_dump(mode="json"),
            )
        self.assertEqual(profile_response.status_code, 422)
        self.assertIn(
            "--enable-profile",
            profile_response.json()["detail"],
        )
        self.assertEqual(normal_response.status_code, 202)

    def test_profile_requests_require_explicit_server_flag(self):
        async def _scenario():
            dit_runtime = SfWanRuntime(config=ServerConfig(role="dit"))
            with self.assertRaisesRegex(ValueError, "--enable-profile"):
                await dit_runtime.submit_dit_profile(
                    DitProfileRequest(prompt="profile")
                )

            vae_runtime = SfWanRuntime(config=ServerConfig(role="vae"))
            with self.assertRaisesRegex(ValueError, "--enable-profile"):
                await vae_runtime.register_latent_job(
                    LatentJobSpec(
                        request_id="profile-disabled",
                        source="profile",
                    )
                )

        asyncio.run(_scenario())

    def test_monolithic_constructs_one_component_set(self):
        fake_components = SimpleNamespace(
            device=torch.device("cpu"),
            distributed_backend="local",
            cpu_offload_requested={
                "text_encoder": True,
                "dit": False,
                "vae": False,
            },
            cpu_offload_effective={
                "text_encoder": "layerwise",
                "dit": "resident",
                "vae": "resident",
            },
        )
        fake_dit = SimpleNamespace(contract={"role": "dit"})
        fake_vae = SimpleNamespace(contract={"role": "vae"})
        model_module = "sglang.multimodal_gen.experimental.jetson_sfwan.model"
        with (
            mock.patch(
                f"{model_module}._ComponentSet",
                return_value=fake_components,
            ) as component_loader,
            mock.patch(
                f"{model_module}.SfWanDitModel",
                return_value=fake_dit,
            ) as dit_constructor,
            mock.patch(
                f"{model_module}.SfWanVaeModel",
                return_value=fake_vae,
            ) as vae_constructor,
        ):
            model = SfWanMonolithicModel(
                load_config=ModelLoadConfig(model_path="unused")
            )
        component_loader.assert_called_once()
        dit_constructor.assert_called_once()
        vae_constructor.assert_called_once()
        self.assertEqual(model.contract["latent_transport"], "in_memory_gpu")
        self.assertEqual(model.contract["distributed_backend"], "local")
        self.assertEqual(
            model.contract["cpu_offload_effective"]["text_encoder"],
            "layerwise",
        )

    def test_dit_profile_never_constructs_a_sender_or_vae_job(self):
        class _FakeDitProfileModel:
            contract = {"role": "dit"}

            def generate(self, *, request, chunk_callback):
                chunk_metrics = []
                for chunk_index in range(request.total_chunks):
                    metrics = {
                        "chunk_index": chunk_index,
                        "denoise_steps": [
                            {"step_index": step_index} for step_index in range(4)
                        ],
                        "clean_kv_cuda_ms": 1.0,
                        "chunk_execution_wall_ms": 5.0,
                    }
                    chunk_metrics.append(metrics)
                    chunk_callback(
                        LatentChunk(
                            chunk_index=chunk_index,
                            tensor="discarded",
                            metrics=metrics,
                        )
                    )
                profile_execution = {
                    "component": "dit",
                    "num_chunks": request.total_chunks,
                    "chunks": chunk_metrics,
                    "dit_execution_wall_ms": 35.0,
                }
                return {
                    "num_chunks": request.total_chunks,
                    "profile_execution": profile_execution,
                }

        async def _scenario():
            runtime = SfWanRuntime(
                config=ServerConfig(role="dit", enable_profile=True),
                model_factory=lambda _config: _FakeDitProfileModel(),
            )
            await runtime.start()
            try:
                self.assertIsNone(runtime.sender)
                with self.assertRaisesRegex(ValueError, "--vae-url"):
                    await runtime.submit_generation(GenerationRequest(prompt="normal"))
                record = await runtime.submit_dit_profile(
                    DitProfileRequest(
                        prompt="profile",
                        height=16,
                        width=16,
                        num_frames=81,
                    )
                )
                await _wait_until_terminal(record)
                self.assertEqual(record.state, JobState.COMPLETED)
                self.assertEqual(record.metrics["profile"], "dit")
                self.assertEqual(record.metrics["num_chunks"], 7)
                profile_events = [
                    event
                    for event in record.events
                    if event.kind == "dit_profile_chunk_completed"
                ]
                self.assertEqual(
                    [event.details["chunk_index"] for event in profile_events],
                    list(range(7)),
                )
                self.assertTrue(
                    all(
                        len(chunk["denoise_steps"]) == 4
                        for chunk in record.metrics["profile_execution"]["chunks"]
                    )
                )
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_monolithic_runtime_has_no_latent_transport_resources(self):
        class _FakeMonolithic:
            contract = {"role": "monolithic"}

        async def _scenario():
            runtime = SfWanRuntime(
                config=ServerConfig(role="monolithic"),
                model_factory=lambda _config: _FakeMonolithic(),
            )
            await runtime.start()
            try:
                self.assertIsNone(runtime.sender)
                self.assertIsNone(runtime._h2d_loader)
                self.assertIsNone(runtime._transfer_executor)
                self.assertEqual(runtime._shm_regions, {})
            finally:
                await runtime.close()

        asyncio.run(_scenario())

    def test_vae_profile_client_has_no_dit_endpoint(self):
        source = inspect.getsource(_run_profile_iteration)
        self.assertIn("/v1/latent-jobs", source)
        self.assertNotIn("/v1/dit", source)
        self.assertNotIn("/v1/generations", source)


class TestSfWanTensorRTVaeConfiguration(CustomTestCase):
    def test_trt_shape_capture_includes_causal_padding(self):
        class _FakeCausalConv(torch.nn.Module):
            _padding = (1, 1, 2, 2, 3, 0)

            def forward(self, value):
                return torch.empty(
                    value.shape[0],
                    8,
                    value.shape[2],
                    value.shape[3],
                    value.shape[4],
                )

        class _FakeVae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = _FakeCausalConv()

        vae = _FakeVae()
        value = torch.empty(1, 4, 1, 5, 6)
        with _capture_target_conv_call_shapes(
            vae=vae,
            target_module_names=("conv",),
        ) as (select_graph, records):
            for graph_kind in ("initial", "steady"):
                select_graph(graph_kind)
                for _ in range(3):
                    vae.conv(value)

        expected = {
            "input_shape": [1, 4, 4, 9, 8],
            "output_shape": [1, 8, 1, 5, 6],
        }
        self.assertEqual(records["initial"]["conv"], [expected] * 3)
        self.assertEqual(records["steady"]["conv"], [expected] * 3)

    def test_trt_qdq_v5_uses_opset_19(self):
        self.assertEqual(QDQ_SCHEMA_VERSION, 5)
        self.assertEqual(QDQ_OPSET, 19)
        self.assertEqual(EXPECTED_CONV_SIGNATURES, 9)
        self.assertEqual(
            QDQ_TOPOLOGY,
            "fp16_cast_fp32_input_qdq_fp32_conv_output_qdq_fp32_cast_fp16",
        )

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx is not installed")
    def test_trt_qdq_v5_quantizes_input_weight_and_output_per_call_site(self):
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        targets = tuple(
            f"decoder.residual.{index}.conv1" for index in range(EXPECTED_LOGICAL_CONVS)
        )
        nodes = []
        initializers = []
        for module_name in targets:
            weight_name = f"{module_name}.weight"
            bias_name = f"{module_name}.bias"
            initializers.extend(
                [
                    numpy_helper.from_array(
                        np.ones((1, 1, 1, 1, 1), dtype=np.float16),
                        name=weight_name,
                    ),
                    numpy_helper.from_array(
                        np.zeros((1,), dtype=np.float16),
                        name=bias_name,
                    ),
                ]
            )
            for call_index in range(3):
                nodes.append(
                    helper.make_node(
                        "Conv",
                        ["input", weight_name, bias_name],
                        [f"{module_name}.output.{call_index}"],
                        name=f"source/{module_name}/{call_index}",
                        kernel_shape=[1, 1, 1],
                    )
                )
        graph = helper.make_graph(
            nodes,
            "qdq-v5-test",
            [
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT16, [1, 1, 1, 1, 1]
                )
            ],
            [
                helper.make_tensor_value_info(
                    f"{targets[-1]}.output.2",
                    TensorProto.FLOAT16,
                    [1, 1, 1, 1, 1],
                )
            ],
            initializer=initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", QDQ_OPSET)],
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.onnx"
            destination = Path(directory) / "qdq.onnx"
            onnx.save(model, source)
            report = rewrite_onnx_with_int8_qdq(
                source_path=source,
                destination_path=destination,
                graph_kind="initial",
                target_module_names=targets,
                activation_input_scales={name: 1.0 for name in targets},
                activation_output_scales={name: 1.0 for name in targets},
                call_site_shape_contracts={
                    name: [
                        {
                            "input_shape": [1, 1, 1, 1, 1],
                            "output_shape": [1, 1, 1, 1, 1],
                        }
                        for _ in range(3)
                    ]
                    for name in targets
                },
            )
            self.assertEqual(report["activation_cast_count"], 84)
            self.assertEqual(report["activation_quantize_count"], 84)
            self.assertEqual(report["weight_quantize_count"], 84)
            self.assertEqual(report["weight_dequantize_count"], 84)
            self.assertEqual(report["output_quantize_count"], 84)
            self.assertEqual(report["output_dequantize_count"], 84)
            self.assertEqual(report["output_cast_count"], 84)
            self.assertEqual(report["unique_weight_source_count"], 84)
            self.assertEqual(report["unique_bias_count"], 84)
            self.assertEqual(report["unexpected_target_cast_nodes"], [])
            self.assertEqual(len(report["conv_signatures"]), 84)
            self.assertEqual(set(report["shape_sources"].values()), {"captured"})

            rewritten_contract = onnx.load(destination)
            initializer_by_name = {
                initializer.name: initializer
                for initializer in rewritten_contract.graph.initializer
            }
            activation_casts = [
                node
                for node in rewritten_contract.graph.node
                if node.op_type == "Cast"
                and node.name.startswith("qdq/initial/activation/")
            ]
            output_casts = [
                node
                for node in rewritten_contract.graph.node
                if node.op_type == "Cast"
                and node.name.startswith("qdq/initial/output/")
            ]
            self.assertEqual(len(activation_casts), 84)
            self.assertEqual(len(output_casts), 84)
            self.assertTrue(
                all(
                    next(
                        attribute.i
                        for attribute in node.attribute
                        if attribute.name == "to"
                    )
                    == TensorProto.FLOAT
                    for node in activation_casts
                )
            )
            self.assertTrue(
                all(
                    next(
                        attribute.i
                        for attribute in node.attribute
                        if attribute.name == "to"
                    )
                    == TensorProto.FLOAT16
                    for node in output_casts
                )
            )
            producers = {
                output_name: node
                for node in rewritten_contract.graph.node
                for output_name in node.output
                if output_name
            }
            consumers = {}
            for node in rewritten_contract.graph.node:
                for input_name in node.input:
                    if input_name:
                        consumers.setdefault(input_name, []).append(node)
            target_convs = [
                node
                for node in rewritten_contract.graph.node
                if node.op_type == "Conv" and node.name.startswith("int8/initial/")
            ]
            self.assertEqual(len(target_convs), 84)
            for conv in target_convs:
                activation_dq = producers[conv.input[0]]
                weight_dq = producers[conv.input[1]]
                self.assertEqual(activation_dq.op_type, "DequantizeLinear")
                self.assertEqual(weight_dq.op_type, "DequantizeLinear")
                activation_q = producers[activation_dq.input[0]]
                activation_cast = producers[activation_q.input[0]]
                self.assertEqual(activation_q.op_type, "QuantizeLinear")
                self.assertEqual(activation_cast.op_type, "Cast")
                self.assertEqual(len(consumers[conv.output[0]]), 1)
                output_q = consumers[conv.output[0]][0]
                self.assertEqual(output_q.op_type, "QuantizeLinear")
                output_dq = consumers[output_q.output[0]][0]
                self.assertEqual(output_dq.op_type, "DequantizeLinear")
                self.assertEqual(consumers[output_dq.output[0]][0].op_type, "Cast")
                bias = initializer_by_name[conv.input[2]]
                self.assertEqual(bias.data_type, TensorProto.FLOAT)
            for q_node in (
                node
                for node in rewritten_contract.graph.node
                if node.op_type == "QuantizeLinear"
            ):
                scale = initializer_by_name[q_node.input[1]]
                self.assertEqual(scale.data_type, TensorProto.FLOAT)
            for weight_q_node in (
                node
                for node in rewritten_contract.graph.node
                if node.op_type == "QuantizeLinear"
                and node.name.startswith("qdq/initial/weight/")
            ):
                source_weight = initializer_by_name[weight_q_node.input[0]]
                self.assertEqual(source_weight.data_type, TensorProto.FLOAT)

            prequantized_destination = Path(directory) / "qdq-prequantized.onnx"
            prequantized_report = rewrite_onnx_with_int8_qdq(
                source_path=source,
                destination_path=prequantized_destination,
                graph_kind="initial",
                target_module_names=targets,
                activation_input_scales={name: 1.0 for name in targets},
                activation_output_scales={name: 1.0 for name in targets},
                weight_encoding=WEIGHT_ENCODING_PREQUANTIZED_INT8_DQ,
                call_site_shape_contracts={
                    name: [
                        {
                            "input_shape": [1, 1, 1, 1, 1],
                            "output_shape": [1, 1, 1, 1, 1],
                        }
                        for _ in range(3)
                    ]
                    for name in targets
                },
            )
            self.assertEqual(prequantized_report["weight_quantize_count"], 0)
            self.assertEqual(prequantized_report["weight_dequantize_count"], 84)
            self.assertEqual(
                prequantized_report["weight_encoding"],
                WEIGHT_ENCODING_PREQUANTIZED_INT8_DQ,
            )
            prequantized_model = onnx.load(prequantized_destination)
            prequantized_initializers = {
                initializer.name: initializer
                for initializer in prequantized_model.graph.initializer
            }
            for weight_dq_node in (
                node
                for node in prequantized_model.graph.node
                if node.op_type == "DequantizeLinear"
                and node.name.startswith("qdq/initial/weight/")
            ):
                self.assertEqual(
                    prequantized_initializers[weight_dq_node.input[0]].data_type,
                    TensorProto.INT8,
                )

            shape_overrides = {
                signature["call_site"]: {
                    "input_shape": signature["input_shape"],
                    "output_shape": signature["output_shape"],
                }
                for signature in report["conv_signatures"]
            }
            missing_value_info = onnx.load(destination)
            del missing_value_info.graph.value_info[:]
            captured_report = audit_qdq_model(
                missing_value_info,
                graph_kind="initial",
                target_module_names=targets,
                shape_overrides=shape_overrides,
            )
            self.assertTrue(captured_report["passed"], captured_report["errors"])
            self.assertEqual(
                set(captured_report["shape_sources"].values()),
                {"captured"},
            )

            mismatched_shapes = json.loads(json.dumps(shape_overrides))
            first_call_site = report["conv_signatures"][0]["call_site"]
            mismatched_shapes[first_call_site]["input_shape"][-1] = 2
            mismatch_report = audit_qdq_model(
                missing_value_info,
                graph_kind="initial",
                target_module_names=targets,
                shape_overrides=mismatched_shapes,
            )
            self.assertFalse(mismatch_report["passed"])
            self.assertTrue(
                any(
                    "input_shape_mismatch" in error
                    for error in mismatch_report["invalid_bindings"]
                )
            )

            rewritten = onnx.load(destination)
            weight_q = [
                node
                for node in rewritten.graph.node
                if node.op_type == "QuantizeLinear"
                and node.name.startswith("qdq/initial/weight/")
            ]
            weight_q[1].input[0] = weight_q[0].input[0]
            failed = audit_qdq_model(
                rewritten,
                graph_kind="initial",
                target_module_names=targets,
            )
            self.assertFalse(failed["passed"])
            self.assertTrue(
                any("weight sources are shared" in error for error in failed["errors"])
            )

            dq_only = onnx.load(destination)
            dq_only_weight_q = next(
                node
                for node in dq_only.graph.node
                if node.op_type == "QuantizeLinear"
                and node.name.startswith("qdq/initial/weight/")
            )
            dq_only_weight_dq = next(
                node
                for node in dq_only.graph.node
                if node.op_type == "DequantizeLinear"
                and node.name.startswith("qdq/initial/weight/")
            )
            dq_only_weight_dq.input[0] = dq_only_weight_q.input[2]
            dq_only_report = audit_qdq_model(
                dq_only,
                graph_kind="initial",
                target_module_names=targets,
            )
            self.assertFalse(dq_only_report["passed"])
            self.assertTrue(
                any(
                    "weight_missing_q" in error
                    for error in dq_only_report["invalid_bindings"]
                )
            )

            wrong_axis = onnx.load(destination)
            wrong_axis_weight_q = next(
                node
                for node in wrong_axis.graph.node
                if node.op_type == "QuantizeLinear"
                and node.name.startswith("qdq/initial/weight/")
            )
            next(
                attribute
                for attribute in wrong_axis_weight_q.attribute
                if attribute.name == "axis"
            ).i = 1
            wrong_axis_report = audit_qdq_model(
                wrong_axis,
                graph_kind="initial",
                target_module_names=targets,
            )
            self.assertFalse(wrong_axis_report["passed"])
            self.assertTrue(
                any(
                    "weight_axis_not_zero" in error
                    for error in wrong_axis_report["invalid_bindings"]
                )
            )

            nonzero_zero_point = onnx.load(destination)
            nonzero_weight_q = next(
                node
                for node in nonzero_zero_point.graph.node
                if node.op_type == "QuantizeLinear"
                and node.name.startswith("qdq/initial/weight/")
            )
            initializer_by_name = {
                initializer.name: initializer
                for initializer in nonzero_zero_point.graph.initializer
            }
            zero_initializer = initializer_by_name[nonzero_weight_q.input[2]]
            zero_initializer.CopyFrom(
                numpy_helper.from_array(
                    np.ones(tuple(zero_initializer.dims), dtype=np.int8),
                    name=zero_initializer.name,
                )
            )
            nonzero_report = audit_qdq_model(
                nonzero_zero_point,
                graph_kind="initial",
                target_module_names=targets,
            )
            self.assertFalse(nonzero_report["passed"])
            self.assertTrue(
                any(
                    "weight_zero_invalid" in error
                    for error in nonzero_report["invalid_bindings"]
                )
            )

            wrong_cast = onnx.load(destination)
            first_activation_cast = next(
                node
                for node in wrong_cast.graph.node
                if node.op_type == "Cast"
                and node.name.startswith("qdq/initial/activation/")
            )
            next(
                attribute
                for attribute in first_activation_cast.attribute
                if attribute.name == "to"
            ).i = TensorProto.FLOAT16
            wrong_cast_report = audit_qdq_model(
                wrong_cast,
                graph_kind="initial",
                target_module_names=targets,
            )
            self.assertFalse(wrong_cast_report["passed"])
            self.assertTrue(
                any(
                    "activation_cast_not_fp32" in error
                    for error in wrong_cast_report["invalid_bindings"]
                )
            )

            invalid_output_scale = onnx.load(destination)
            output_q = next(
                node
                for node in invalid_output_scale.graph.node
                if node.op_type == "QuantizeLinear"
                and node.name.startswith("qdq/initial/output/")
            )
            invalid_output_initializers = {
                initializer.name: initializer
                for initializer in invalid_output_scale.graph.initializer
            }
            output_scale = invalid_output_initializers[output_q.input[1]]
            output_scale.CopyFrom(
                numpy_helper.from_array(
                    np.asarray(float("nan"), dtype=np.float32),
                    name=output_scale.name,
                )
            )
            invalid_output_report = audit_qdq_model(
                invalid_output_scale,
                graph_kind="initial",
                target_module_names=targets,
            )
            self.assertFalse(invalid_output_report["passed"])
            self.assertTrue(
                any(
                    "output_scale_invalid" in error
                    for error in invalid_output_report["invalid_bindings"]
                )
            )

            malformed_qdq = onnx.load(destination)
            malformed_activation_q = next(
                node
                for node in malformed_qdq.graph.node
                if node.op_type == "QuantizeLinear"
                and node.name.startswith("qdq/initial/activation/")
            )
            del malformed_activation_q.input[2:]
            malformed_report = audit_qdq_model(
                malformed_qdq,
                graph_kind="initial",
                target_module_names=targets,
            )
            self.assertFalse(malformed_report["passed"])
            self.assertTrue(
                any(
                    "activation_q_contract_invalid" in error
                    for error in malformed_report["invalid_bindings"]
                )
            )

    def test_trt_qdq_rejects_relabelled_legacy_opset(self):
        legacy_model = SimpleNamespace(
            opset_import=[SimpleNamespace(domain="", version=17)]
        )
        fake_onnx = SimpleNamespace(load=mock.Mock(return_value=legacy_model))
        targets = tuple(
            f"decoder.residual.{index}.conv1" for index in range(EXPECTED_LOGICAL_CONVS)
        )

        with (
            mock.patch(
                "sglang.multimodal_gen.experimental.jetson_sfwan."
                "vae_trt_qdq._lazy_onnx",
                return_value=(fake_onnx, object(), object(), (object(), object())),
            ),
            self.assertRaisesRegex(
                ValueError,
                "requires a real ONNX opset 19 source graph, got opset 17",
            ),
        ):
            rewrite_onnx_with_int8_qdq(
                source_path=Path("legacy_opset17.onnx"),
                destination_path=Path("invalid_qdq.onnx"),
                graph_kind="initial",
                target_module_names=targets,
                activation_input_scales={name: 1.0 for name in targets},
                activation_output_scales={name: 1.0 for name in targets},
            )

        self.assertEqual(legacy_model.opset_import[0].version, 17)

    def test_trt_qdq_source_reuse_requires_exact_opset(self):
        legacy_model = SimpleNamespace(
            opset_import=[SimpleNamespace(domain="ai.onnx", version=17)]
        )
        fake_onnx = ModuleType("onnx")
        fake_onnx.load = mock.Mock(return_value=legacy_model)
        fake_onnx.checker = SimpleNamespace(check_model=mock.Mock())

        with (
            mock.patch.dict(sys.modules, {"onnx": fake_onnx}),
            self.assertRaisesRegex(
                ValueError,
                "uses ONNX opset 17, but Q/DQ requires a real opset 19 export",
            ),
        ):
            _load_validated_fp16_onnx(
                path=Path("initial_fp16.onnx"),
                expected_input_shapes={},
                expected_output_shapes={},
                required_opset=QDQ_OPSET,
            )

        self.assertEqual(_onnx_default_opset_version(legacy_model), 17)
        fake_onnx.checker.check_model.assert_called_once_with(
            legacy_model,
            full_check=True,
        )

    def test_trt_tactic_audit_requires_int8_convolution_evidence(self):
        call_site = "int8/initial/decoder.block.conv1/call_0"
        int8_inspector = json.dumps(
            {
                "Layers": [
                    {
                        "Name": call_site,
                        "LayerType": "CaskConvolution",
                        "HasDynamicFilter": 0,
                        "Inputs": [{"Format/Datatype": "Int8"}],
                        "Outputs": [{"Format/Datatype": "Int8"}],
                        "Weights": {"Count": 4096, "Type": "Int8"},
                        "TacticName": "sm87_xmma_fprop_implicit_gemm_i8i8_i32",
                    }
                ]
            }
        )
        passed = _audit_tensorrt_tactics(
            graph_kind="initial",
            call_site_names=[call_site],
            inspector_json=int8_inspector,
            expected_call_sites=1,
        )
        self.assertTrue(passed["passed"])

        tf32_inspector = json.dumps(
            {
                "Layers": [
                    {
                        "Name": "fused_conv",
                        "Metadata": f"[ONNX Layer: {call_site}]",
                        "LayerType": "CaskConvolution",
                        "HasDynamicFilter": 1,
                        "Inputs": [{"Format/Datatype": "Half"}],
                        "Outputs": [{"Format/Datatype": "Half"}],
                        "Weights": {"Count": 0, "Type": "Half"},
                        "TacticName": (
                            "sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16"
                        ),
                    }
                ]
            }
        )
        failed = _audit_tensorrt_tactics(
            graph_kind="initial",
            call_site_names=[call_site],
            inspector_json=tf32_inspector,
            expected_call_sites=1,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["non_int8_call_sites"], [call_site])
        self.assertEqual(failed["dynamic_filter_call_sites"], [call_site])
        self.assertEqual(failed["fp16_fallback_call_sites"], [call_site])
        self.assertFalse(failed["matches"][call_site]["activation_int8"])
        self.assertFalse(failed["matches"][call_site]["static_weight_int8"])

        static_half_inspector = json.dumps(
            {
                "Layers": [
                    {
                        "Name": "fused_static_half_conv",
                        "Metadata": f"[ONNX Layer: {call_site}]",
                        "LayerType": "CaskConvolution",
                        "HasDynamicFilter": 0,
                        "Inputs": [{"Format/Datatype": "Float"}],
                        "Outputs": [{"Format/Datatype": "Float"}],
                        "Weights": {"Count": 3981312, "Type": "Float"},
                        "TacticName": (
                            "sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32"
                        ),
                    }
                ]
            }
        )
        static_half = _audit_tensorrt_tactics(
            graph_kind="initial",
            call_site_names=[call_site],
            inspector_json=static_half_inspector,
            expected_call_sites=1,
        )
        self.assertFalse(static_half["passed"])
        self.assertEqual(static_half["dynamic_filter_call_sites"], [])
        self.assertEqual(static_half["activation_not_int8_call_sites"], [call_site])
        self.assertEqual(
            static_half["static_weight_not_int8_call_sites"],
            [call_site],
        )
        self.assertEqual(static_half["output_not_int8_call_sites"], [call_site])
        self.assertEqual(static_half["fp32_or_tf32_fallback_call_sites"], [call_site])

        deceptive_inspector = json.dumps(
            {
                "Layers": [
                    {
                        "Name": call_site,
                        "LayerType": "CaskConvolution",
                        "HasDynamicFilter": 0,
                        "Inputs": [{"Format/Datatype": "Int8"}],
                        "Outputs": [{"Format/Datatype": "Int8"}],
                        "Weights": {"Count": 4096, "Type": "Int8"},
                        "TacticName": "sm80_xmma_fprop_f16f16_f16f16_f16",
                    }
                ]
            }
        )
        deceptive = _audit_tensorrt_tactics(
            graph_kind="initial",
            call_site_names=[call_site],
            inspector_json=deceptive_inspector,
            expected_call_sites=1,
        )
        self.assertFalse(deceptive["passed"])
        self.assertTrue(deceptive["matches"][call_site]["activation_int8"])
        self.assertTrue(deceptive["matches"][call_site]["static_weight_int8"])
        self.assertTrue(deceptive["matches"][call_site]["fp16_fallback"])

        mixed_output_inspector = json.dumps(
            {
                "Layers": [
                    {
                        "Name": call_site,
                        "LayerType": "CaskConvolution",
                        "HasDynamicFilter": 0,
                        "Inputs": [{"Format/Datatype": "Int8"}],
                        "Outputs": [{"Format/Datatype": "Float"}],
                        "Weights": {"Count": 4096, "Type": "Int8"},
                        "TacticName": "sm87_xmma_fprop_implicit_gemm_i8i8_i32",
                    }
                ]
            }
        )
        mixed = _audit_tensorrt_tactics(
            graph_kind="initial",
            call_site_names=[call_site],
            inspector_json=mixed_output_inspector,
            expected_call_sites=1,
        )
        self.assertFalse(mixed["passed"])
        self.assertEqual(mixed["output_not_int8_call_sites"], [call_site])

    def test_trt_weight_fallback_requires_static_mapped_weight_failure(self):
        eligible = {
            "signatures": {
                "signature": {
                    "passed": False,
                    "mapped_count": 1,
                    "unmapped_call_sites": [],
                    "dynamic_filter_call_sites": [],
                    "static_weight_not_int8_call_sites": ["conv"],
                }
            }
        }
        self.assertTrue(_probe_suite_supports_prequantized_weight_fallback(eligible))
        dynamic = copy.deepcopy(eligible)
        dynamic["signatures"]["signature"]["dynamic_filter_call_sites"] = ["conv"]
        self.assertFalse(_probe_suite_supports_prequantized_weight_fallback(dynamic))

    def test_trt_feature_cache_audit_rejects_int8_binding(self):
        initial = [
            {
                "name": f"cache_out_{index:03d}",
                "dtype": "float16",
                "mode": "output",
            }
            for index in range(32)
        ]
        steady = [
            {
                "name": f"cache_in_{index:03d}",
                "dtype": "float16",
                "mode": "input",
            }
            for index in range(32)
        ] + [
            {
                "name": f"cache_out_{index:03d}",
                "dtype": "float16",
                "mode": "output",
            }
            for index in range(32)
        ]
        passed = _audit_feature_cache_engine_io(
            initial_io=initial,
            steady_io=steady,
        )
        self.assertTrue(passed["passed"])
        steady[0]["dtype"] = "int8"
        failed = _audit_feature_cache_engine_io(
            initial_io=initial,
            steady_io=steady,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["quantized_bindings"], ["cache_in_000"])

    def test_trt_build_stage_resume_requires_matching_source_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "build_state.json"
            source_path = root / "graph.onnx"
            output_path = root / "engine.plan"
            source_path.write_bytes(b"graph-v2")
            output_path.write_bytes(b"plan-v2")
            source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            state = {"identity": {"schema_version": 1}, "stages": {}}
            _record_stage(
                state_path=state_path,
                state=state,
                stage="initial_int8_detailed",
                source_sha256=source_hash,
                output_path=output_path,
                profiling_verbosity="detailed",
            )
            self.assertTrue(
                _stage_is_current(
                    state=state,
                    stage="initial_int8_detailed",
                    source_sha256=source_hash,
                    output_path=output_path,
                    profiling_verbosity="detailed",
                )
            )
            audit_path = root / "audit.json"
            audit_path.write_text('{"passed": false}', encoding="utf-8")
            _mark_stage_audit(
                state_path=state_path,
                state=state,
                stage="initial_int8_detailed",
                passed=False,
                audit_path=audit_path,
            )
            self.assertEqual(
                state["stages"]["initial_int8_detailed"]["status"],
                "audit_failed",
            )
            self.assertTrue(
                _stage_is_current(
                    state=state,
                    stage="initial_int8_detailed",
                    source_sha256=source_hash,
                    output_path=output_path,
                    profiling_verbosity="detailed",
                )
            )
            output_path.write_bytes(b"corrupted")
            self.assertFalse(
                _stage_is_current(
                    state=state,
                    stage="initial_int8_detailed",
                    source_sha256=source_hash,
                    output_path=output_path,
                    profiling_verbosity="detailed",
                )
            )

    def test_trt_build_parser_exposes_resume_and_timing_cache(self):
        argv = [
            "vae_trt_build",
            "--model-path",
            "model",
            "--output-dir",
            "/engines",
            "--resume",
            "--preflight-only",
            "--timing-cache",
            "/engines/timing.cache",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = _parse_trt_build_args()
        self.assertTrue(args.resume)
        self.assertTrue(args.preflight_only)
        self.assertEqual(args.timing_cache, "/engines/timing.cache")

    def test_trt_timing_cache_uses_builder_config_api(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "timing.cache"
            cache_path.write_bytes(b"existing-cache")
            timing_cache = object()
            config = SimpleNamespace(
                create_timing_cache=mock.Mock(return_value=timing_cache),
                set_timing_cache=mock.Mock(return_value=True),
                get_timing_cache=mock.Mock(),
            )
            trt = SimpleNamespace(__version__="10.3.0")

            attached = _attach_timing_cache(
                trt=trt,
                config=config,
                timing_cache_path=cache_path,
            )

            self.assertIs(attached, timing_cache)
            config.create_timing_cache.assert_called_once_with(b"existing-cache")
            config.set_timing_cache.assert_called_once_with(
                timing_cache,
                ignore_mismatch=False,
            )
            self.assertIsNone(
                _attach_timing_cache(
                    trt=trt,
                    config=SimpleNamespace(),
                    timing_cache_path=None,
                )
            )

    def test_trt_timing_cache_fails_closed_on_incompatible_api_or_cache(self):
        trt = SimpleNamespace(__version__="10.3.0")
        with self.assertRaisesRegex(
            RuntimeError,
            "IBuilderConfig is missing timing-cache APIs",
        ):
            _attach_timing_cache(
                trt=trt,
                config=SimpleNamespace(),
                timing_cache_path=Path("timing.cache"),
            )

        config = SimpleNamespace(
            create_timing_cache=mock.Mock(return_value=object()),
            set_timing_cache=mock.Mock(return_value=False),
            get_timing_cache=mock.Mock(),
        )
        with self.assertRaisesRegex(RuntimeError, "rejected timing cache"):
            _attach_timing_cache(
                trt=trt,
                config=config,
                timing_cache_path=Path("timing.cache"),
            )

    def test_trt_timing_cache_is_committed_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "timing.cache"
            timing_cache = SimpleNamespace(
                serialize=mock.Mock(return_value=b"new-cache")
            )
            config = SimpleNamespace(
                get_timing_cache=mock.Mock(return_value=timing_cache)
            )

            candidate = _candidate_timing_cache_bytes(
                config=config,
                timing_cache_path=cache_path,
            )
            self.assertEqual(candidate, b"new-cache")
            self.assertFalse(cache_path.exists())
            _commit_timing_cache(
                timing_cache_path=cache_path,
                candidate=candidate,
            )

            self.assertEqual(cache_path.read_bytes(), b"new-cache")
            self.assertFalse(cache_path.with_name("timing.cache.partial").exists())

    def test_trt_failed_build_does_not_persist_timing_cache(self):
        config = SimpleNamespace(
            set_memory_pool_limit=mock.Mock(),
            profiling_verbosity=None,
        )
        parser = SimpleNamespace(parse_from_file=mock.Mock(return_value=True))
        builder = SimpleNamespace(
            create_network=mock.Mock(return_value=object()),
            create_builder_config=mock.Mock(return_value=config),
            build_serialized_network=mock.Mock(return_value=None),
        )
        trt = SimpleNamespace(
            Builder=mock.Mock(return_value=builder),
            OnnxParser=mock.Mock(return_value=parser),
            MemoryPoolType=SimpleNamespace(WORKSPACE=object()),
        )

        with (
            mock.patch(
                "sglang.multimodal_gen.experimental.jetson_sfwan."
                "vae_trt_build._trt_logger",
                return_value=object(),
            ),
            mock.patch(
                "sglang.multimodal_gen.experimental.jetson_sfwan."
                "vae_trt_build._network_flags",
                return_value=0,
            ),
            mock.patch(
                "sglang.multimodal_gen.experimental.jetson_sfwan."
                "vae_trt_build._profiling_verbosity",
                return_value=object(),
            ),
            mock.patch(
                "sglang.multimodal_gen.experimental.jetson_sfwan."
                "vae_trt_build._attach_timing_cache",
                return_value=object(),
            ),
            mock.patch(
                "sglang.multimodal_gen.experimental.jetson_sfwan."
                "vae_trt_build._candidate_timing_cache_bytes"
            ) as candidate,
            self.assertRaisesRegex(RuntimeError, "could not build"),
        ):
            _build_engine_bytes(
                trt=trt,
                onnx_path=Path("probe.onnx"),
                workspace_gib=1,
                profiling_verbosity="none",
                timing_cache_path=Path("timing.cache"),
            )

        candidate.assert_not_called()

    def test_trt_export_cache_boundary_normalizes_fp16(self):
        active_indices = tuple(range(32))
        reference = torch.zeros(1, dtype=torch.float16)
        source_dtypes = (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        )
        cache = [None] * 33
        for index in active_indices:
            cache[index] = (
                torch.arange(
                    6,
                    dtype=source_dtypes[index % len(source_dtypes)],
                )
                .reshape(2, 3)
                .transpose(0, 1)
            )

        normalized = _normalize_export_cache_tensors(
            cache=cache,
            active_cache_indices=active_indices,
            reference_tensor=reference,
        )

        self.assertEqual(len(normalized), 32)
        self.assertTrue(all(tensor.dtype == torch.float16 for tensor in normalized))
        self.assertTrue(all(tensor.is_contiguous() for tensor in normalized))
        self.assertTrue(
            all(cache[index] is normalized[index] for index in active_indices)
        )

        with self.assertRaisesRegex(ValueError, "32 active cache"):
            _normalize_export_cache_tensors(
                cache=cache,
                active_cache_indices=active_indices[:-1],
                reference_tensor=reference,
            )

        integer_cache = [torch.zeros(1, dtype=torch.int64) for _ in range(32)] + [None]
        with self.assertRaisesRegex(TypeError, "unsupported dtype"):
            _normalize_export_cache_tensors(
                cache=integer_cache,
                active_cache_indices=active_indices,
                reference_tensor=reference,
            )

    def test_trt_export_wrappers_emit_fp16_cache_outputs(self):
        active_indices = tuple(range(32))
        fake_vae = SimpleNamespace(
            config=SimpleNamespace(patch_size=None),
            post_quant_conv=torch.nn.Identity(),
            decoder=torch.nn.Identity(),
        )
        initial_wrapper, steady_wrapper = _make_export_wrappers(
            vae=fake_vae,
            cache_slot_count=33,
            active_cache_indices=active_indices,
        )
        latent = torch.zeros((1, 16, 3, 2, 2), dtype=torch.float16)

        def _fake_decoder_chunk(**kwargs):
            for index in active_indices:
                kwargs["cache"][index] = torch.full(
                    (1, 1, 1, 1, 1),
                    float(index),
                    dtype=torch.float32,
                )
            frames = 9 if kwargs["first_request_chunk"] else 12
            return torch.zeros(
                (1, 3, frames, 2, 2),
                dtype=torch.float16,
            )

        with mock.patch(
            "sglang.multimodal_gen.experimental.jetson_sfwan."
            "vae_trt_build._run_decoder_chunk",
            side_effect=_fake_decoder_chunk,
        ):
            initial_outputs = initial_wrapper(latent)
            steady_outputs = steady_wrapper(latent, *initial_outputs[1:])

        self.assertEqual(len(initial_outputs), 33)
        self.assertEqual(len(steady_outputs), 33)
        self.assertTrue(
            all(tensor.dtype == torch.float16 for tensor in initial_outputs)
        )
        self.assertTrue(all(tensor.dtype == torch.float16 for tensor in steady_outputs))

    def test_trt_onnx_io_contract_rejects_non_fp16_binding(self):
        fake_onnx = ModuleType("onnx")
        fake_onnx.TensorProto = SimpleNamespace(FLOAT16=10)

        def _value_info(name, dtype, shape):
            return SimpleNamespace(
                name=name,
                type=SimpleNamespace(
                    tensor_type=SimpleNamespace(
                        elem_type=dtype,
                        shape=SimpleNamespace(
                            dim=[
                                SimpleNamespace(
                                    dim_param="",
                                    dim_value=dimension,
                                )
                                for dimension in shape
                            ]
                        ),
                    )
                ),
            )

        expected_inputs = {"latent": (1, 16, 3, 2, 2)}
        expected_outputs = {
            "rgb": (1, 3, 9, 16, 16),
            "cache_out_000": (1, 1, 2, 2, 2),
        }
        valid_model = SimpleNamespace(
            graph=SimpleNamespace(
                input=[
                    _value_info(
                        "latent",
                        10,
                        expected_inputs["latent"],
                    )
                ],
                output=[
                    _value_info(name, 10, shape)
                    for name, shape in expected_outputs.items()
                ],
            )
        )
        with mock.patch.dict(sys.modules, {"onnx": fake_onnx}):
            self.assertFalse(
                _validate_onnx_fp16_io_contract(
                    model=valid_model,
                    path=Path("initial_fp16.onnx"),
                    expected_input_shapes=expected_inputs,
                    expected_output_shapes=expected_outputs,
                )
            )

            symbolic_model = SimpleNamespace(
                graph=SimpleNamespace(
                    input=valid_model.graph.input,
                    output=[
                        _value_info("rgb", 10, (0, 0, 0, 0, 0)),
                        valid_model.graph.output[1],
                    ],
                )
            )
            for dimension in symbolic_model.graph.output[0].type.tensor_type.shape.dim:
                dimension.dim_param = "legacy_export_symbol"
            self.assertTrue(
                _validate_onnx_fp16_io_contract(
                    model=symbolic_model,
                    path=Path("initial_fp16.onnx"),
                    expected_input_shapes=expected_inputs,
                    expected_output_shapes=expected_outputs,
                    materialize_symbolic_shapes=True,
                )
            )
            self.assertEqual(
                tuple(
                    dimension.dim_value
                    for dimension in symbolic_model.graph.output[
                        0
                    ].type.tensor_type.shape.dim
                ),
                expected_outputs["rgb"],
            )
            self.assertTrue(
                all(
                    not dimension.dim_param
                    for dimension in symbolic_model.graph.output[
                        0
                    ].type.tensor_type.shape.dim
                )
            )
            self.assertFalse(
                _validate_onnx_fp16_io_contract(
                    model=symbolic_model,
                    path=Path("initial_fp16.onnx"),
                    expected_input_shapes=expected_inputs,
                    expected_output_shapes=expected_outputs,
                )
            )

            concrete_mismatch = SimpleNamespace(
                graph=SimpleNamespace(
                    input=valid_model.graph.input,
                    output=[
                        _value_info("rgb", 10, (1, 3, 8, 16, 16)),
                        valid_model.graph.output[1],
                    ],
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                r"shape is \(1, 3, 8, 16, 16\)",
            ):
                _validate_onnx_fp16_io_contract(
                    model=concrete_mismatch,
                    path=Path("initial_fp16.onnx"),
                    expected_input_shapes=expected_inputs,
                    expected_output_shapes=expected_outputs,
                    materialize_symbolic_shapes=True,
                )

            invalid_model = SimpleNamespace(
                graph=SimpleNamespace(
                    input=valid_model.graph.input,
                    output=[
                        _value_info("rgb", 1, expected_outputs["rgb"]),
                        valid_model.graph.output[1],
                    ],
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                "output 'rgb' must be float16",
            ):
                _validate_onnx_fp16_io_contract(
                    model=invalid_model,
                    path=Path("initial_fp16.onnx"),
                    expected_input_shapes=expected_inputs,
                    expected_output_shapes=expected_outputs,
                )

    def test_onnx_export_layout_context_is_scoped_and_exception_safe(self):
        from sglang.multimodal_gen.runtime.layers import parallel_conv
        from sglang.multimodal_gen.runtime.models.vaes import wanvae as wanvae_module

        original_wan_match = wanvae_module.match_conv3d_input_format
        original_parallel_match = getattr(
            parallel_conv,
            "_match_conv3d_input_format",
        )
        value = object()
        weight = object()

        with _portable_conv3d_layout_for_export():
            self.assertIs(
                wanvae_module.match_conv3d_input_format(value, weight),
                value,
            )
            self.assertIs(
                getattr(parallel_conv, "_match_conv3d_input_format")(
                    value,
                    weight,
                ),
                value,
            )

        self.assertIs(
            wanvae_module.match_conv3d_input_format,
            original_wan_match,
        )
        self.assertIs(
            getattr(parallel_conv, "_match_conv3d_input_format"),
            original_parallel_match,
        )

        with self.assertRaisesRegex(RuntimeError, "export failed"):
            with _portable_conv3d_layout_for_export():
                raise RuntimeError("export failed")

        self.assertIs(
            wanvae_module.match_conv3d_input_format,
            original_wan_match,
        )
        self.assertIs(
            getattr(parallel_conv, "_match_conv3d_input_format"),
            original_parallel_match,
        )

    def test_onnx_export_nearest_context_is_scoped_and_exception_safe(self):
        from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanUpsample

        wrapper = torch.nn.Sequential(
            *(
                WanUpsample(
                    scale_factor=(2.0, 2.0),
                    mode="nearest-exact",
                )
                for _ in range(3)
            )
        )
        upsamplers = list(wrapper.children())

        with _portable_nearest_upsample_for_export(wrapper):
            self.assertEqual(
                [module.mode for module in upsamplers],
                ["nearest", "nearest", "nearest"],
            )

        self.assertEqual(
            [module.mode for module in upsamplers],
            ["nearest-exact", "nearest-exact", "nearest-exact"],
        )

        with self.assertRaisesRegex(RuntimeError, "export failed"):
            with _portable_nearest_upsample_for_export(wrapper):
                raise RuntimeError("export failed")

        self.assertEqual(
            [module.mode for module in upsamplers],
            ["nearest-exact", "nearest-exact", "nearest-exact"],
        )

    def test_onnx_export_nearest_context_rejects_non_contract_modules(self):
        from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanUpsample

        wrong_count = torch.nn.Sequential(
            WanUpsample(
                scale_factor=(2.0, 2.0),
                mode="nearest-exact",
            )
        )
        with self.assertRaisesRegex(ValueError, "expects 3"):
            with _portable_nearest_upsample_for_export(wrong_count):
                pass

        wrong_scale = torch.nn.Sequential(
            *(
                WanUpsample(
                    scale_factor=(3.0, 3.0) if index == 0 else (2.0, 2.0),
                    mode="nearest-exact",
                )
                for index in range(3)
            )
        )
        with self.assertRaisesRegex(ValueError, "scale_factor"):
            with _portable_nearest_upsample_for_export(wrong_scale):
                pass

    def test_trt_precision_requires_engine_dir_and_rejects_cpu_offload(self):
        self.assertTrue(_uses_trt_vae("fp16_trt"))
        self.assertTrue(_uses_trt_vae("int8_trt"))
        self.assertFalse(_uses_trt_vae("fp16"))

        with self.assertRaisesRegex(ValueError, "--vae-engine-dir"):
            SfWanRuntime(config=ServerConfig(role="vae", vae_precision="int8_trt"))
        with self.assertRaisesRegex(ValueError, "--vae-cpu-offload"):
            SfWanRuntime(
                config=ServerConfig(
                    role="monolithic",
                    vae_precision="fp16_trt",
                    vae_engine_dir="/engines",
                    vae_cpu_offload=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "valid only"):
            SfWanRuntime(
                config=ServerConfig(
                    role="vae",
                    vae_precision="fp16",
                    vae_engine_dir="/engines",
                )
            )

    def test_server_parser_exposes_trt_precision_and_engine_dir(self):
        argv = [
            "server",
            "--role",
            "vae",
            "--vae-precision",
            "int8_trt",
            "--vae-engine-dir",
            "/workspace/engines",
        ]
        with mock.patch.object(sys, "argv", argv):
            config = _parse_args()
        self.assertEqual(config.vae_precision, "int8_trt")
        self.assertEqual(config.vae_engine_dir, "/workspace/engines")

    def test_manifest_validation_checks_static_cache_and_plan_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest_by_file = {}
            for precision in ("fp16", "int8"):
                for kind in ("initial", "steady"):
                    name = f"{kind}_{precision}.plan"
                    payload = f"{precision}-{kind}".encode()
                    (root / name).write_bytes(payload)
                    digest_by_file[name] = hashlib.sha256(payload).hexdigest()
            empty_tactic_lists = {
                "unmapped_call_sites": [],
                "non_int8_call_sites": [],
                "activation_not_int8_call_sites": [],
                "output_not_int8_call_sites": [],
                "static_weight_not_int8_call_sites": [],
                "dynamic_filter_call_sites": [],
                "fp16_fallback_call_sites": [],
                "fp32_or_tf32_fallback_call_sites": [],
            }
            plan_sha = {
                kind: digest_by_file[f"{kind}_int8.plan"]
                for kind in ("initial", "steady")
            }
            audit_report = {
                "schema_version": QDQ_SCHEMA_VERSION,
                "weight_encoding": WEIGHT_ENCODING_FP32_QDQ,
                "passed": True,
                "preflight_passed": True,
                "complete": True,
                "target_conv_call_sites_per_graph": 84,
                "probe_suite": {
                    "schema_version": QDQ_SCHEMA_VERSION,
                    "weight_encoding": WEIGHT_ENCODING_FP32_QDQ,
                    "selected_weight_encoding": WEIGHT_ENCODING_FP32_QDQ,
                    "passed": True,
                    "signature_count": EXPECTED_CONV_SIGNATURES,
                    "probed_signature_count": EXPECTED_CONV_SIGNATURES,
                    "source_call_site_counts": {"initial": 84, "steady": 84},
                    "errors": [],
                    "signatures": {
                        f"signature-{signature_index}": {
                            "signature_id": f"signature-{signature_index}",
                            "passed": True,
                            "mapped_count": 1,
                            "errors": [],
                            "weight_encoding": WEIGHT_ENCODING_FP32_QDQ,
                            "source_call_sites": {
                                kind: [
                                    f"int8/{kind}/call_{index}"
                                    for index in range(signature_index, 84, 9)
                                ]
                                for kind in ("initial", "steady")
                            },
                            "matches": {
                                f"int8/probe/{signature_index}/call_0": {
                                    "passed": True,
                                    "activation_int8": True,
                                    "output_int8": True,
                                    "static_weight_int8": True,
                                    "dynamic_filter": False,
                                    "int8_tactic": True,
                                    "fp16_fallback": False,
                                    "fp32_or_tf32_fallback": False,
                                }
                            },
                            **empty_tactic_lists,
                        }
                        for signature_index in range(EXPECTED_CONV_SIGNATURES)
                    },
                },
                "structural": {
                    kind: {
                        "schema_version": QDQ_SCHEMA_VERSION,
                        "qdq_topology": QDQ_TOPOLOGY,
                        "weight_encoding": WEIGHT_ENCODING_FP32_QDQ,
                        "passed": True,
                        "errors": [],
                        "target_conv_call_site_count": 84,
                        "activation_cast_count": 84,
                        "activation_quantize_count": 84,
                        "activation_dequantize_count": 84,
                        "weight_quantize_count": 84,
                        "weight_dequantize_count": 84,
                        "output_quantize_count": 84,
                        "output_dequantize_count": 84,
                        "output_cast_count": 84,
                        "unique_weight_source_count": 84,
                        "unique_weight_quantize_output_count": 84,
                        "unique_weight_dequantize_output_count": 84,
                        "unique_output_quantize_output_count": 84,
                        "unique_output_dequantize_output_count": 84,
                        "unique_bias_count": 84,
                        "unexpected_target_cast_nodes": [],
                    }
                    for kind in ("initial", "steady")
                },
                "feature_cache": {
                    "passed": True,
                    "errors": [],
                    "dtype": "float16",
                    "tensor_count": 32,
                    "binding_counts": {
                        "initial": {"inputs": 0, "outputs": 32},
                        "steady": {"inputs": 32, "outputs": 32},
                    },
                    "single_bank_bytes": TRT_VAE_CACHE_BANK_BYTES,
                    "double_bank_bytes": TRT_VAE_CACHE_BANK_BYTES * 2,
                    "quantized_bindings": [],
                    "physical_engine_io_checked": True,
                },
                "tactics": {
                    kind: {
                        "passed": True,
                        "errors": [],
                        "mapped_count": 84,
                        "build_profiling_verbosity": "detailed",
                        "plan_sha256": plan_sha[kind],
                        "weight_encoding": WEIGHT_ENCODING_FP32_QDQ,
                        "matches": {
                            f"int8/{kind}/call_{index}": {
                                "passed": True,
                                "activation_int8": True,
                                "output_int8": True,
                                "static_weight_int8": True,
                                "dynamic_filter": False,
                                "int8_tactic": True,
                                "fp16_fallback": False,
                                "fp32_or_tf32_fallback": False,
                            }
                            for index in range(84)
                        },
                        **empty_tactic_lists,
                    }
                    for kind in ("initial", "steady")
                },
                "plan_sha256": plan_sha,
            }
            audit_payload = json.dumps(audit_report, sort_keys=True).encode()
            (root / "int8_audit_v5.json").write_bytes(audit_payload)
            audit_digest = hashlib.sha256(audit_payload).hexdigest()

            cache_shapes = [[1, 1, 1, 1, TRT_VAE_CACHE_TOTAL_ELEMENTS - 31]]
            cache_shapes.extend([[1, 1, 1, 1, 1] for _ in range(31)])
            manifest = {
                "schema_version": 1,
                "model_id": "model",
                "batch_size": 1,
                "height": 480,
                "width": 832,
                "latent_shape": list(TRT_VAE_LATENT_SHAPE),
                "latent_dtype": "float16",
                "quantization": {
                    "qdq_schema_version": QDQ_SCHEMA_VERSION,
                    "topology": QDQ_TOPOLOGY,
                    "weight_encoding": WEIGHT_ENCODING_FP32_QDQ,
                },
                "cache": {
                    "allocated_slot_count": 33,
                    "active_slot_indices": list(range(32)),
                    "bindings": [
                        {
                            "index": index,
                            "shape": shape,
                            "dtype": "float16",
                        }
                        for index, shape in enumerate(cache_shapes)
                    ],
                    "total_elements": TRT_VAE_CACHE_TOTAL_ELEMENTS,
                    "single_bank_bytes": TRT_VAE_CACHE_BANK_BYTES,
                    "double_bank_bytes": TRT_VAE_CACHE_BANK_BYTES * 2,
                },
                "engines": {
                    precision: {
                        kind: {
                            "file": f"{kind}_{precision}.plan",
                            "sha256": digest_by_file[f"{kind}_{precision}.plan"],
                            "rgb_shape": [
                                1,
                                3,
                                9 if kind == "initial" else 12,
                                480,
                                832,
                            ],
                            **(
                                {
                                    "profiling_verbosity": "detailed",
                                    "audit_passed": True,
                                }
                                if precision == "int8"
                                else {}
                            ),
                        }
                        for kind in ("initial", "steady")
                    }
                    for precision in ("fp16", "int8")
                },
                "int8_audit": {
                    "passed": True,
                    "schema_version": QDQ_SCHEMA_VERSION,
                    "weight_encoding": WEIGHT_ENCODING_FP32_QDQ,
                    "preflight_passed": True,
                    "target_conv_call_sites_per_graph": 84,
                    "report_file": "int8_audit_v5.json",
                    "report_sha256": audit_digest,
                    "plan_sha256": plan_sha,
                },
            }
            validated = validate_trt_vae_manifest(
                manifest,
                engine_dir=root,
                precision="int8",
                model_path="model",
                verify_plan_hashes=True,
            )
            self.assertEqual(len(validated["cache_shapes"]), 32)
            self.assertEqual(set(validated["engines"]), {"initial", "steady"})

            first_initial_evidence = next(
                iter(audit_report["tactics"]["initial"]["matches"].values())
            )
            first_initial_evidence["fp16_fallback"] = True
            deceptive_payload = json.dumps(audit_report, sort_keys=True).encode()
            (root / "int8_audit_v5.json").write_bytes(deceptive_payload)
            manifest["int8_audit"]["report_sha256"] = hashlib.sha256(
                deceptive_payload
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "tactic evidence"):
                validate_trt_vae_manifest(
                    manifest,
                    engine_dir=root,
                    precision="int8",
                    model_path="model",
                    verify_plan_hashes=True,
                )
            first_initial_evidence["fp16_fallback"] = False
            audit_payload = json.dumps(audit_report, sort_keys=True).encode()
            (root / "int8_audit_v5.json").write_bytes(audit_payload)
            manifest["int8_audit"]["report_sha256"] = hashlib.sha256(
                audit_payload
            ).hexdigest()

            manifest["quantization"]["qdq_schema_version"] = 4
            with self.assertRaisesRegex(
                ValueError, f"Q/DQ schema {QDQ_SCHEMA_VERSION}"
            ):
                validate_trt_vae_manifest(
                    manifest,
                    engine_dir=root,
                    precision="int8",
                    model_path="model",
                    verify_plan_hashes=True,
                )
            validated_fp16 = validate_trt_vae_manifest(
                manifest,
                engine_dir=root,
                precision="fp16",
                model_path="model",
                verify_plan_hashes=True,
            )
            self.assertEqual(set(validated_fp16["engines"]), {"initial", "steady"})
            manifest["quantization"]["qdq_schema_version"] = QDQ_SCHEMA_VERSION

            manifest["quantization"]["topology"] = "legacy_half_qdq"
            with self.assertRaisesRegex(
                ValueError, f"Q/DQ schema {QDQ_SCHEMA_VERSION}"
            ):
                validate_trt_vae_manifest(
                    manifest,
                    engine_dir=root,
                    precision="int8",
                    model_path="model",
                    verify_plan_hashes=True,
                )
            manifest["quantization"]["topology"] = QDQ_TOPOLOGY

            manifest["engines"]["int8"]["steady"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_trt_vae_manifest(
                    manifest,
                    engine_dir=root,
                    precision="int8",
                    model_path="model",
                    verify_plan_hashes=True,
                )
            manifest["engines"]["int8"]["steady"]["sha256"] = digest_by_file[
                "steady_int8.plan"
            ]
            (root / "int8_audit_v5.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audit digest"):
                validate_trt_vae_manifest(
                    manifest,
                    engine_dir=root,
                    precision="int8",
                    model_path="model",
                    verify_plan_hashes=True,
                )

    def test_trt_runtime_lowers_int8_nvtx_unless_explicitly_enabled(self):
        class _FakeContext:
            nvtx_verbosity = "detailed"

        trt = SimpleNamespace(
            ProfilingVerbosity=SimpleNamespace(NONE="none", DETAILED="detailed")
        )
        context = _FakeContext()
        self.assertEqual(
            _configure_context_nvtx(
                context=context,
                trt=trt,
                precision="int8",
                enable_nvtx=False,
            ),
            "none",
        )
        self.assertEqual(context.nvtx_verbosity, "none")
        self.assertEqual(
            _configure_context_nvtx(
                context=context,
                trt=trt,
                precision="int8",
                enable_nvtx=True,
            ),
            "detailed",
        )

    def test_trt_runtime_enforces_initial_then_steady_and_reset(self):
        class _FakeTensor:
            def __init__(self, dtype):
                self.shape = TRT_VAE_LATENT_SHAPE
                self.dtype = dtype

            def to(self, *, dtype):
                return _FakeTensor(dtype)

        runtime = TensorRTVaeRuntime.__new__(TensorRTVaeRuntime)
        runtime.precision = "int8"
        runtime.device = "cuda:0"
        runtime.enable_profile = False
        runtime.enable_nvtx = False
        runtime._torch = SimpleNamespace(float16="fp16", float32="fp32")
        calls = []

        def _execute(_self, *, kind, latent):
            calls.append((kind, latent.dtype))
            return f"{kind}-output"

        runtime._execute = MethodType(_execute, runtime)
        runtime.reset_request()
        first, first_metrics = runtime.decode_chunk(
            chunk_index=0,
            denormalized_fp32=_FakeTensor("fp32"),
        )
        second, second_metrics = runtime.decode_chunk(
            chunk_index=1,
            denormalized_fp32=_FakeTensor("fp32"),
        )
        self.assertEqual(first, "initial-output")
        self.assertEqual(second, "steady-output")
        self.assertEqual(calls, [("initial", "fp16"), ("steady", "fp16")])
        self.assertEqual(first_metrics["trt_engine_kind"], "initial")
        self.assertEqual(second_metrics["trt_engine_kind"], "steady")

        runtime.finish_request()
        runtime.reset_request()
        with self.assertRaisesRegex(ValueError, "expected chunk 0"):
            runtime.decode_chunk(
                chunk_index=1,
                denormalized_fp32=_FakeTensor("fp32"),
            )

    def test_trt_runtime_alternates_cache_banks_without_in_place_overwrite(self):
        class _FakeContext:
            def __init__(self):
                self.stream_handles = []

            def execute_async_v3(self, *, stream_handle):
                self.stream_handles.append(stream_handle)
                return True

        runtime = TensorRTVaeRuntime.__new__(TensorRTVaeRuntime)
        runtime.device = "cuda:0"
        runtime._torch = SimpleNamespace(
            cuda=SimpleNamespace(
                current_stream=lambda **_kwargs: SimpleNamespace(cuda_stream=17)
            )
        )
        runtime._contexts = {
            "initial": _FakeContext(),
            "steady": _FakeContext(),
        }
        runtime._cache_banks = [
            [f"a-{index}" for index in range(32)],
            [f"b-{index}" for index in range(32)],
        ]
        runtime._rgb_outputs = {
            "initial": "rgb-initial",
            "steady": "rgb-steady",
        }
        runtime._read_bank_index = None
        addresses = []

        def _set_address(_self, context, name, tensor):
            addresses.append((context, name, tensor))

        runtime._set_address = MethodType(_set_address, runtime)
        self.assertEqual(
            runtime._execute(kind="initial", latent="latent-0"),
            "rgb-initial",
        )
        self.assertEqual(runtime._read_bank_index, 0)
        first_steady_start = len(addresses)
        self.assertEqual(
            runtime._execute(kind="steady", latent="latent-1"),
            "rgb-steady",
        )
        self.assertEqual(runtime._read_bank_index, 1)
        steady_addresses = {
            name: tensor for _context, name, tensor in addresses[first_steady_start:]
        }
        self.assertEqual(steady_addresses["cache_in_000"], "a-0")
        self.assertEqual(steady_addresses["cache_out_000"], "b-0")
        self.assertNotEqual(
            steady_addresses["cache_in_000"],
            steady_addresses["cache_out_000"],
        )

    def test_trt_monolithic_component_loader_skips_torch_vae(self):
        components = SimpleNamespace(
            device="cuda:0",
            modules={},
            pipeline_config=SimpleNamespace(),
            server_args=SimpleNamespace(),
            distributed_backend="local",
            cpu_offload_requested={
                "text_encoder": True,
                "dit": False,
                "vae": False,
            },
            cpu_offload_effective={
                "text_encoder": "layerwise",
                "dit": "resident",
                "vae": "resident",
            },
        )
        fake_dit = SimpleNamespace(contract={"role": "dit"})
        fake_vae = SimpleNamespace(contract={"role": "vae", "vae_backend": "tensorrt"})
        with (
            mock.patch(
                "sglang.multimodal_gen.experimental.jetson_sfwan.model._ComponentSet",
                return_value=components,
            ) as component_loader,
            mock.patch(
                "sglang.multimodal_gen.experimental.jetson_sfwan.model.SfWanDitModel",
                return_value=fake_dit,
            ),
            mock.patch(
                "sglang.multimodal_gen.experimental.jetson_sfwan.model.SfWanVaeModel",
                return_value=fake_vae,
            ),
        ):
            model = SfWanMonolithicModel(
                load_config=ModelLoadConfig(
                    model_path="model",
                    vae_precision="int8_trt",
                    vae_engine_dir="/engines",
                )
            )
        self.assertEqual(
            component_loader.call_args.kwargs["component_names"],
            DIT_COMPONENT_NAMES,
        )
        self.assertEqual(model.contract["vae"]["vae_backend"], "tensorrt")


class TestSfWanTensorRTLayerProfile(CustomTestCase):
    """Import-light coverage for the opt-in TensorRT diagnostic path."""

    @staticmethod
    def _inspector(*names):
        return {
            "Layers": [
                {
                    "Name": name,
                    "LayerType": "Convolution",
                    "ParameterType": "Convolution",
                    "TacticName": "fake_tactic",
                    "Metadata": "",
                }
                for name in names
            ]
        }

    @staticmethod
    def _catalog(*, kind, precision="fp16", name=None, mapped=False):
        catalog = build_physical_layer_catalog(
            engine_kind=kind,
            plan_sha256=("a" if kind == "initial" else "b") * 64,
            inspector=TestSfWanTensorRTLayerProfile._inspector(name or f"{kind}-layer"),
            precision=precision,
        )
        if mapped:
            catalog["layers"][0]["logical_call_sites"] = [
                f"int8/{kind}/module_{index // 3}/call_{index % 3}"
                for index in range(EXPECTED_CALL_SITES)
            ]
            catalog["layers"][0]["category"] = "target_quantized_conv"
            catalog["layers"][0]["classification_source"] = "source_onnx"
            catalog["expected_target_call_site_count"] = EXPECTED_CALL_SITES
            catalog["catalog_sha256"] = catalog_sha256(catalog)
        return catalog

    def test_cli_is_strictly_opt_in_and_role_scoped(self):
        with self.assertRaisesRegex(ValueError, "requires --enable-profile"):
            SfWanRuntime(
                config=ServerConfig(
                    role="vae",
                    vae_precision="int8_trt",
                    vae_engine_dir="/engines",
                    enable_trt_layer_profile=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "only for --role vae"):
            SfWanRuntime(
                config=ServerConfig(
                    role="monolithic",
                    vae_precision="int8_trt",
                    vae_engine_dir="/engines",
                    enable_profile=True,
                    enable_trt_layer_profile=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "fp16_trt or int8_trt"):
            SfWanRuntime(
                config=ServerConfig(
                    role="vae",
                    vae_precision="fp16",
                    enable_profile=True,
                    enable_trt_layer_profile=True,
                )
            )

        with mock.patch.object(
            sys,
            "argv",
            [
                "server",
                "--role",
                "vae",
                "--vae-precision",
                "fp16_trt",
                "--vae-engine-dir",
                "/engines",
                "--enable-profile",
                "--enable-trt-layer-profile",
            ],
        ):
            config = _parse_args()
        self.assertTrue(config.enable_trt_layer_profile)

        parsed = _build_parser().parse_args(
            [
                "profile-vae",
                "--server-url",
                "http://vae",
                "--trt-layer-profile-json",
                "/results/layers.json",
            ]
        )
        self.assertEqual(parsed.trt_layer_profile_json, "/results/layers.json")

    def test_profile_builder_parser_exposes_resume_without_production_outputs(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "vae_trt_profile_build",
                "--engine-dir",
                "/engines",
                "--workspace-gib",
                "6",
                "--resume",
            ],
        ):
            args = _parse_trt_profile_build_args()
        self.assertEqual(args.engine_dir, "/engines")
        self.assertEqual(args.workspace_gib, 6.0)
        self.assertTrue(args.resume)

    def test_client_requires_exact_layer_profile_artifact_switch_match(self):
        class _Response:
            def __init__(self, enabled):
                self._enabled = enabled

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "contract": {
                        "trt_layer_profile_enabled": self._enabled,
                    }
                }

        class _Client:
            def __init__(self, enabled):
                self.enabled = enabled

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def get(self, _url):
                return _Response(self.enabled)

        def _args(path):
            return SimpleNamespace(
                warmup=0,
                repeat=1,
                request_timeout_seconds=1.0,
                server_url="http://vae",
                trt_layer_profile_json=path,
                summary_json="summary.json",
            )

        with mock.patch("httpx.AsyncClient", return_value=_Client(True)):
            with self.assertRaisesRegex(ValueError, "provide"):
                asyncio.run(run_profile_vae(_args(None)))
        with mock.patch("httpx.AsyncClient", return_value=_Client(False)):
            with self.assertRaisesRegex(ValueError, "requires"):
                asyncio.run(run_profile_vae(_args("layers.json")))
        same_path = _args("summary.json")
        with mock.patch("httpx.AsyncClient", return_value=_Client(True)):
            with self.assertRaisesRegex(ValueError, "different files"):
                asyncio.run(run_profile_vae(same_path))

    def test_profiler_requires_api_and_enforces_enqueue_report_order(self):
        class _IProfiler:
            pass

        trt = SimpleNamespace(IProfiler=_IProfiler)
        order = []

        class _Context:
            profiler = None
            enqueue_emits_profile = True

            def report_to_profiler(self):
                order.append("report")
                self.profiler.report_layer_time("layer", 2.5)
                return True

        context = _Context()
        self.assertTrue(
            probe_trt_layer_profile_api(
                trt=trt,
                contexts={"initial": context},
            )["available"]
        )
        with self.assertRaisesRegex(RuntimeError, "missing APIs"):
            probe_trt_layer_profile_api(
                trt=SimpleNamespace(),
                contexts={"initial": SimpleNamespace()},
            )

        capture = TrtLayerProfileCapture(
            trt=trt,
            context=context,
            engine_kind="initial",
            catalog=self._catalog(kind="initial", name="layer", mapped=True),
        )
        capture.begin_capture(0)
        order.append("enqueue")
        capture.mark_enqueue_succeeded()
        metrics = capture.report_and_finish(engine_event_ms=2.5)
        self.assertEqual(order, ["enqueue", "report"])
        self.assertEqual(metrics["reported_layer_count"], 1)
        self.assertEqual(metrics["layer_times_ms"], [2.5])
        self.assertEqual(
            capture.catalog["expected_target_call_site_count"],
            EXPECTED_CALL_SITES,
        )

    def test_profiler_fails_closed_for_zero_callbacks_and_catalog_drift(self):
        class _IProfiler:
            pass

        trt = SimpleNamespace(IProfiler=_IProfiler)

        class _Context:
            profiler = None
            enqueue_emits_profile = True

            def __init__(self, callbacks):
                self.callbacks = callbacks

            def report_to_profiler(self):
                for name, elapsed in self.callbacks.pop(0):
                    self.profiler.report_layer_time(name, elapsed)
                return True

        empty_context = _Context([[]])
        empty = TrtLayerProfileCapture(
            trt=trt,
            context=empty_context,
            engine_kind="initial",
            catalog=self._catalog(kind="initial", name="layer"),
        )
        empty.begin_capture(0)
        empty.mark_enqueue_succeeded()
        with self.assertRaisesRegex(RuntimeError, "zero layers"):
            empty.report_and_finish(engine_event_ms=1.0)

        false_context = _Context([[("layer", 1.0)]])
        false_context.report_to_profiler = lambda: False
        false_report = TrtLayerProfileCapture(
            trt=trt,
            context=false_context,
            engine_kind="initial",
            catalog=self._catalog(kind="initial", name="layer"),
        )
        false_report.begin_capture(0)
        false_report.mark_enqueue_succeeded()
        with self.assertRaisesRegex(RuntimeError, "returned False"):
            false_report.report_and_finish(engine_event_ms=1.0)

        drift_context = _Context([[("same", 1.0)], [("other", 1.0)]])
        drift = TrtLayerProfileCapture(
            trt=trt,
            context=drift_context,
            engine_kind="steady",
            catalog=build_physical_layer_catalog(
                engine_kind="steady",
                plan_sha256="b" * 64,
                inspector=self._inspector("same", "other"),
                precision="fp16",
            ),
        )
        drift.begin_capture(1)
        drift.mark_enqueue_succeeded()
        drift.report_and_finish(engine_event_ms=1.0)
        drift.begin_capture(2)
        drift.mark_enqueue_succeeded()
        with self.assertRaisesRegex(RuntimeError, "catalog drifted"):
            drift.report_and_finish(engine_event_ms=1.0)

    def test_duplicate_names_and_fused_call_sites_are_counted_once(self):
        int8_audit = {
            "schema_version": QDQ_SCHEMA_VERSION,
            "passed": True,
            "complete": True,
            "errors": [],
            "plan_sha256": {"initial": "a" * 64},
            "tactics": {
                "initial": {
                    "passed": True,
                    "mapped_count": 2,
                    "errors": [],
                    "matches": {
                        "call-a": {
                            "layer_names": ["fused"],
                            "metadata": [],
                        },
                        "call-b": {
                            "layer_names": ["fused"],
                            "metadata": [],
                        },
                    },
                }
            },
        }
        with mock.patch(
            "sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_profile."
            "EXPECTED_CALL_SITES",
            2,
        ):
            catalog = build_physical_layer_catalog(
                engine_kind="initial",
                plan_sha256="a" * 64,
                inspector=self._inspector("fused", "fused"),
                precision="int8",
                int8_audit=int8_audit,
            )
        self.assertEqual(
            [layer["exact_key"] for layer in catalog["layers"]],
            ["fused#0", "fused#1"],
        )
        self.assertEqual(
            catalog["layers"][0]["logical_call_sites"],
            ["call-a", "call-b"],
        )
        metrics = make_compact_layer_profile_metrics(
            catalog=catalog,
            layer_times_ms=[2.0, 3.0],
            engine_event_ms=5.0,
        )
        self.assertEqual(metrics["layer_sum_ms"], 5.0)
        self.assertEqual(
            metrics["category_totals_ms"]["target_quantized_conv"],
            5.0,
        )
        self.assertAlmostEqual(
            sum(metrics["category_totals_ms"].values()),
            metrics["layer_sum_ms"],
        )

    def test_catalog_does_not_guess_generic_layout_work_is_feature_cache(self):
        inspector = {
            "Layers": [
                {
                    "Name": "generic reshape",
                    "LayerType": "Shuffle",
                    "Metadata": "",
                },
                {
                    "Name": "cache_out_003 reshape",
                    "LayerType": "Shuffle",
                    "Metadata": "feature_cache",
                },
            ]
        }
        catalog = build_physical_layer_catalog(
            engine_kind="steady",
            plan_sha256="b" * 64,
            inspector=inspector,
            precision="fp16",
        )
        self.assertEqual(
            [layer["category"] for layer in catalog["layers"]],
            ["other", "cache_layout_copy"],
        )

    def test_trt10_norm_and_causal_slice_layers_are_classified(self):
        inspector = {
            "Layers": [
                {
                    "Name": (
                        "(Unnamed Layer* 123) [ElementWise] + "
                        "/decoder/up_blocks.3/resnets.1/norm2_2/ReduceL2"
                    ),
                    "LayerType": "Reduce",
                    "ParameterType": "Reduce",
                    "Metadata": "",
                },
                {
                    "Name": ("__myl_CastCastMaxMinDivMulTanhMulAddMulCast_myl808_0"),
                    "LayerType": "kgen",
                    "Metadata": "",
                },
                {
                    "Name": "__myl_SlicCast_myl939_33",
                    "LayerType": "kgen",
                    "Metadata": "",
                },
                {
                    "Name": "/decoder/up_blocks.3/resnets.1_1/Slice",
                    "LayerType": "Slice",
                    "ParameterType": "Slice",
                    "Metadata": "",
                },
                {
                    "Name": "unrelated layout conversion",
                    "LayerType": "Reformat",
                    "ParameterType": "Reformat",
                    "Metadata": "",
                },
            ]
        }
        catalog = build_physical_layer_catalog(
            engine_kind="steady",
            plan_sha256="b" * 64,
            inspector=inspector,
            precision="fp16",
        )
        self.assertEqual(
            [layer["category"] for layer in catalog["layers"]],
            [
                "norm_activation_residual",
                "norm_activation_residual",
                "cache_layout_copy",
                "cache_layout_copy",
                "other",
            ],
        )

    def test_int8_catalog_classifies_anonymous_audited_boundary_kernels(self):
        int8_audit = {
            "schema_version": QDQ_SCHEMA_VERSION,
            "passed": True,
            "complete": True,
            "errors": [],
            "plan_sha256": {"initial": "a" * 64},
            "tactics": {
                "initial": {
                    "passed": True,
                    "mapped_count": 2,
                    "errors": [],
                    "matches": {
                        "call-a": {
                            "layer_names": ["target-a"],
                            "metadata": [],
                        },
                        "call-b": {
                            "layer_names": ["target-b"],
                            "metadata": [],
                        },
                    },
                }
            },
        }
        layers = [
            {
                "Name": "__myl_TranResh_myl_unrelated_0",
                "LayerType": "kgen",
            },
            {
                "Name": "qdq/initial/activation/a/call_0/cast_to_fp32",
                "LayerType": "NoOp",
            },
            {
                "Name": "cache_in_000 concat",
                "LayerType": "Concatenation",
            },
            {
                "Name": "__myl_TranReshSlic_myl_a_0",
                "LayerType": "kgen",
            },
            {
                "Name": "__myl_ReshTran_myl_a_1",
                "LayerType": "kgen",
            },
            {
                "Name": "target-a",
                "LayerType": "CaskConvolution",
                "ParameterType": "Convolution",
            },
            {
                "Name": "qdq/initial/output/a/call_0/cast_to_fp16",
                "LayerType": "NoOp",
            },
            {
                "Name": "Reformatting CopyNode to PWN(/decoder/resnet/Add)",
                "LayerType": "Reformat",
            },
            {
                "Name": "__myl_CastCastAddCast_myl_residual_0",
                "LayerType": "kgen",
            },
            {
                "Name": "__myl_CastCastMulCast_myl_norm_0",
                "LayerType": "kgen",
            },
            {
                "Name": "__myl_TranResh_myl_not_adjacent_0",
                "LayerType": "kgen",
            },
            {
                "Name": "/decoder/norm/ReduceL2",
                "LayerType": "Reduce",
            },
            {
                "Name": "qdq/initial/activation/b/call_0/quantize",
                "LayerType": "Reformat",
            },
            {
                "Name": "__myl_SlicResh_myl_b_0",
                "LayerType": "kgen",
            },
            {
                "Name": "__myl_Tran_myl_b_1",
                "LayerType": "kgen",
            },
            {
                "Name": "target-b",
                "LayerType": "CaskConvolution",
                "ParameterType": "Convolution",
            },
        ]
        with mock.patch(
            "sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_profile."
            "EXPECTED_CALL_SITES",
            2,
        ):
            catalog = build_physical_layer_catalog(
                engine_kind="initial",
                plan_sha256="a" * 64,
                inspector={"Layers": layers},
                precision="int8",
                int8_audit=int8_audit,
            )

        categories = [layer["category"] for layer in catalog["layers"]]
        self.assertEqual(categories[0], "other")
        self.assertEqual(categories[10], "other")
        self.assertEqual(categories[1], "target_qdq_cast_reformat")
        self.assertEqual(categories[6], "target_qdq_cast_reformat")
        self.assertEqual(categories[2], "cache_layout_copy")
        self.assertEqual(categories[5], "target_quantized_conv")
        self.assertEqual(categories[15], "target_quantized_conv")
        for index in (3, 4, 13, 14):
            self.assertEqual(categories[index], "target_qdq_cast_reformat")
            self.assertEqual(
                catalog["layers"][index]["classification_source"],
                "int8_audited_boundary",
            )
        for index in (7, 8, 9, 11):
            self.assertEqual(categories[index], "norm_activation_residual")

    def test_fp16_catalog_maps_the_same_84_target_call_sites_exactly(self):
        target_map = []
        layers = []
        for index in range(EXPECTED_CALL_SITES):
            module_index = index // 3
            call_index = index % 3
            if index == 0:
                source_node = "/decoder/shared/Conv"
            elif index == 1:
                source_node = "/decoder/shared/Conv_1"
            else:
                source_node = f"/decoder/residual_{module_index}/Conv_{call_index}"
            target_map.append(
                {
                    "logical_call_site": (
                        f"int8/initial/module_{module_index}/call_{call_index}"
                    ),
                    "module_name": f"module_{module_index}",
                    "call_index": call_index,
                    "source_onnx_node_name": source_node,
                    "source_weight_initializer": f"module_{module_index}.weight",
                }
            )
            layers.append(
                {
                    "Name": f"{source_node} + fused activation",
                    "LayerType": "Convolution",
                    "ParameterType": "Convolution",
                    "TacticName": "fp16_conv",
                    "Metadata": "",
                }
            )
        catalog = build_physical_layer_catalog(
            engine_kind="initial",
            plan_sha256="a" * 64,
            inspector={"Layers": layers},
            precision="fp16",
            fp16_target_call_sites=target_map,
        )
        mapped = {
            call_site
            for layer in catalog["layers"]
            for call_site in layer["logical_call_sites"]
        }
        self.assertEqual(len(mapped), EXPECTED_CALL_SITES)
        self.assertTrue(
            all(len(layer["logical_call_sites"]) == 1 for layer in catalog["layers"])
        )
        self.assertTrue(
            all(
                layer["category"] == "target_quantized_conv"
                and layer["classification_source"] == "source_onnx"
                for layer in catalog["layers"]
            )
        )

    def test_measured_aggregation_excludes_warmup_and_writes_sha_bound_artifact(self):
        catalogs = {
            "initial": self._catalog(kind="initial", mapped=True),
            "steady": self._catalog(kind="steady", mapped=True),
        }

        def _iteration(index, warmup):
            chunks = []
            for chunk_index in range(7):
                kind = "initial" if chunk_index == 0 else "steady"
                elapsed = float(100 * index + chunk_index + 1)
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "trt_layer_profile": make_compact_layer_profile_metrics(
                            catalog=catalogs[kind],
                            layer_times_ms=[elapsed],
                            engine_event_ms=elapsed,
                        ),
                    }
                )
            return {
                "iteration": index,
                "warmup": warmup,
                "request_id": f"request-{index}",
                "state": "completed",
                "error": None,
                "profile_execution": {"chunks": chunks},
            }

        result = aggregate_trt_layer_profile_iterations(
            [_iteration(0, True), _iteration(1, False), _iteration(2, False)],
            catalogs=catalogs,
            precision="fp16",
            plan_sha256={"initial": "a" * 64, "steady": "b" * 64},
        )
        summary = result["summary"]
        self.assertEqual(summary["initial"]["sample_count"], 2)
        self.assertEqual(summary["steady_pooled"]["sample_count"], 12)
        self.assertEqual(summary["whole_request"]["sample_count"], 2)
        self.assertTrue(summary["valid_for_optimization_decision"])
        self.assertEqual(
            set(summary["category_percentages"]),
            set(PROFILE_CATEGORIES),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layers.json"
            reference = write_trt_layer_profile_artifact(
                path=path,
                detailed=result["detailed"],
            )
            self.assertEqual(reference["path"], str(path.resolve()))
            self.assertEqual(
                reference["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_dedicated_layer_profile_server_rejects_normal_vae_jobs(self):
        async def _scenario():
            runtime = SfWanRuntime(
                config=ServerConfig(
                    role="vae",
                    vae_precision="int8_trt",
                    vae_engine_dir="/engines",
                    enable_profile=True,
                    enable_trt_layer_profile=True,
                )
            )
            try:
                with self.assertRaisesRegex(ValueError, "source=profile"):
                    await runtime.register_latent_job(
                        LatentJobSpec(
                            request_id="normal-job",
                            height=480,
                            width=832,
                            num_frames=81,
                            source="dit",
                        )
                    )
            finally:
                runtime._executor.shutdown(wait=True)
                assert runtime._transfer_executor is not None
                runtime._transfer_executor.shutdown(wait=True)

        asyncio.run(_scenario())


class TestSfWanTensorRTFusionExperiment(CustomTestCase):
    """Protect fusion-v1 isolation, evidence, and comparison contracts."""

    @staticmethod
    def _production_summary(
        *,
        backend: str,
        precision: str,
        variant: str,
        base_chunk_ms: float,
    ) -> dict:
        measured = []
        for iteration in range(2):
            chunks = []
            for chunk_index in range(7):
                elapsed = base_chunk_ms + chunk_index + iteration
                chunk = {
                    "chunk_index": chunk_index,
                    "chunk_execution_cuda_ms": elapsed,
                }
                if backend == "tensorrt":
                    chunk["trt_engine_cuda_ms"] = elapsed - 1.0
                chunks.append(chunk)
            measured.append(
                {
                    "iteration": iteration + 1,
                    "warmup": False,
                    "request_id": f"request-{iteration}",
                    "state": "completed",
                    "error": None,
                    "profile_execution": {
                        "chunks": chunks,
                        "vae_execution_cuda_ms": sum(
                            chunk["chunk_execution_cuda_ms"] for chunk in chunks
                        ),
                    },
                }
            )
        server = {
            "role": "vae",
            "profile_enabled": True,
            "trt_layer_profile_enabled": False,
            "vae_backend": backend,
            "vae_engine_precision": precision,
            "vae_trt_variant": variant,
            "vae_runtime_gpu_name": "Orin",
            "vae_runtime_compute_capability": [8, 7],
            "vae_runtime_cuda_version": "12.9",
            "vae_engine_sm": [8, 7] if backend == "tensorrt" else None,
            "vae_engine_tensorrt_version": (
                "10.3.0" if backend == "tensorrt" else None
            ),
            "vae_engine_plan_sha256": (
                {"initial": "a" * 64, "steady": "b" * 64}
                if backend == "tensorrt"
                else None
            ),
            "vae_trt_plugin_sha256": "c" * 64 if variant == FUSION_VARIANT else None,
        }
        return {
            "mode": "profile-vae",
            "warmup": 1,
            "repeat": 2,
            "measurement_context": {
                "schema_version": 1,
                "request": {
                    "height": 480,
                    "width": 832,
                    "num_frames": 81,
                    "fps": 16,
                    "seed": 1024,
                    "latent_frames": 21,
                    "latent_frames_per_chunk": 3,
                    "total_chunks": 7,
                },
                "run": {
                    "warmup": 1,
                    "repeat": 2,
                    "measured_iterations": 2,
                },
                "server": server,
            },
            "measured": measured,
            # A deliberately absurd warmup proves the comparison consumes
            # only the client's measured view.
            "all_iterations": [
                {
                    "iteration": 0,
                    "warmup": True,
                    "profile_execution": {"vae_execution_cuda_ms": 10**9},
                },
                *measured,
            ],
        }

    def test_server_cli_and_validation_keep_fusion_opt_in(self):
        argv = [
            "server",
            "--role",
            "vae",
            "--vae-precision",
            "int8_trt",
            "--vae-engine-dir",
            "/workspace/engines",
            "--vae-trt-variant",
            "fusion_v1",
        ]
        with mock.patch.object(sys, "argv", argv):
            config = _parse_args()
        self.assertEqual(config.vae_trt_variant, "fusion_v1")

        with self.assertRaisesRegex(ValueError, "requires --vae-precision int8_trt"):
            SfWanRuntime(
                config=ServerConfig(
                    role="vae",
                    vae_precision="fp16_trt",
                    vae_engine_dir="/engines",
                    vae_trt_variant="fusion_v1",
                )
            )
        with self.assertRaisesRegex(ValueError, "valid only for VAE execution"):
            SfWanRuntime(
                config=ServerConfig(
                    role="dit",
                    vae_precision="int8_trt",
                    vae_engine_dir="/engines",
                    vae_trt_variant="fusion_v1",
                )
            )

    def test_graph_rewrite_helpers_reject_shared_consumers_and_public_outputs(self):
        first = SimpleNamespace(name="first", output=["middle"])
        allowed = SimpleNamespace(name="allowed", output=["final"])
        unexpected = SimpleNamespace(name="unexpected", output=["side"])
        consumers = {"middle": [allowed, unexpected], "final": []}
        with self.assertRaisesRegex(ValueError, "shared consumers"):
            _require_only_consumers(
                tensor="middle",
                consumers=consumers,
                allowed_node_names={"allowed"},
            )
        with self.assertRaisesRegex(ValueError, "shared consumers"):
            _require_removable_subgraph(
                nodes=[first, allowed],
                replacement_output="final",
                consumers=consumers,
                graph_outputs=set(),
            )
        with self.assertRaisesRegex(ValueError, "public graph output"):
            _require_removable_subgraph(
                nodes=[first],
                replacement_output="different",
                consumers={"middle": []},
                graph_outputs={"middle"},
            )

    def test_probe_selection_requires_every_focused_call_site(self):
        call_sites = {
            "initial": "int8/initial/decoder.up_blocks.3.block.conv1/call_0",
            "steady": "int8/steady/decoder.up_blocks.3.block.conv1/call_0",
        }
        analysis = {
            "graphs": {
                kind: {"call_sites": [{"call_site": value, "focused": True}]}
                for kind, value in call_sites.items()
            }
        }
        probe = {
            "signatures": {
                "signature": {
                    "passed": True,
                    "selected_cache_update_mode": "dual",
                    "call_sites": {kind: [value] for kind, value in call_sites.items()},
                }
            }
        }
        selected, modes = _fusion_selected_call_sites(analysis=analysis, probe=probe)
        self.assertEqual(
            selected, {kind: {value} for kind, value in call_sites.items()}
        )
        self.assertEqual(
            modes,
            {kind: {value: "dual"} for kind, value in call_sites.items()},
        )
        probe["signatures"]["signature"]["call_sites"]["steady"] = []
        with self.assertRaisesRegex(RuntimeError, "every focused steady"):
            _fusion_selected_call_sites(analysis=analysis, probe=probe)

    def test_fusion_signature_is_stable_and_shape_sensitive(self):
        record = {
            "conv_kind": "conv1",
            "current_shape": [1, 96, 1, 60, 104],
            "cache_shape": [1, 96, 2, 60, 104],
            "padded_shape": [1, 96, 3, 62, 106],
            "pads": [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            "epilogue_output_shape": [1, 96, 1, 60, 104],
        }
        first = _fusion_analysis_signature(record)
        self.assertEqual(first, _fusion_analysis_signature(copy.deepcopy(record)))
        record["padded_shape"][-1] += 1
        self.assertNotEqual(first, _fusion_analysis_signature(record))

    def test_causal_concat_keeps_cache_first_for_four_frame_current(self):
        cache, current = _ordered_causal_concat_inputs(
            ["cache", "current"],
            {
                "cache": [1, 96, 2, 60, 104],
                "current": [1, 96, 4, 60, 104],
            },
        )
        self.assertEqual((cache, current), ("cache", "current"))
        with self.assertRaisesRegex(ValueError, "1/2-frame cache"):
            _ordered_causal_concat_inputs(
                ["current", "cache"],
                {
                    "cache": [1, 96, 2, 60, 104],
                    "current": [1, 96, 4, 60, 104],
                },
            )

    def test_fusion_analysis_v2_solves_pad_current_and_cache_shapes(self):
        self.assertEqual(FUSION_ANALYSIS_SCHEMA_VERSION, 2)
        padded = [1, 96, 6, 62, 106]
        pads = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
        prepad = _unpad_shape(padded, pads)
        self.assertEqual(prepad, [1, 96, 5, 60, 104])
        current = _current_shape_from_prepad(
            prepad_shape=prepad,
            cache_shape=[1, 96, 1, 60, 104],
        )
        self.assertEqual(current, [1, 96, 4, 60, 104])
        self.assertEqual(
            _cache_update_shape(current_shape=current, cache_shape=[1, 96, 1, 60, 104]),
            [1, 96, 2, 60, 104],
        )
        self.assertEqual(
            _cache_update_shape(
                current_shape=[1, 96, 1, 60, 104],
                cache_shape=[1, 96, 1, 60, 104],
            ),
            [1, 96, 2, 60, 104],
        )

        shapes: dict[str, list[int]] = {}
        ranks: dict[str, int] = {}
        sources: dict[str, list[str]] = {}
        _record_static_shape(
            shapes=shapes,
            ranks=ranks,
            sources=sources,
            tensor="activation",
            shape=current,
            source="captured",
        )
        with self.assertRaisesRegex(ValueError, "static shape conflict"):
            _record_static_shape(
                shapes=shapes,
                ranks=ranks,
                sources=sources,
                tensor="activation",
                shape=[1, 96, 3, 60, 104],
                source="manifest",
            )

    def test_final_cache_output_rejects_downstream_conv_dependencies(self):
        exact = SimpleNamespace(
            name="exact_slice",
            op_type="Slice",
            input=["current"],
            output=["exact_cache"],
        )
        downstream_conv = SimpleNamespace(
            name="downstream_conv",
            op_type="Conv",
            input=["current"],
            output=["downstream_activation"],
        )
        downstream_slice = SimpleNamespace(
            name="downstream_slice",
            op_type="Slice",
            input=["downstream_activation"],
            output=["wrong_cache"],
        )
        producer = {
            "exact_cache": exact,
            "downstream_activation": downstream_conv,
            "wrong_cache": downstream_slice,
        }
        result = _find_cache_output(
            current="current",
            cache=None,
            graph_outputs={"exact_cache", "wrong_cache"},
            shapes={
                "current": [1, 96, 4, 60, 104],
                "exact_cache": [1, 96, 2, 60, 104],
                "wrong_cache": [1, 96, 2, 60, 104],
            },
            producer=producer,
        )
        self.assertEqual(result, "exact_cache")

    def test_fusion_analysis_contracts_are_sha_bound_to_v5_audit_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            structural = {}
            for kind in ("initial", "steady"):
                structural[kind] = {
                    "conv_signatures": [
                        {
                            "call_site": f"int8/{kind}/module_{index}/call_0",
                            "input_shape": [1, 96, 3, 62, 106],
                            "output_shape": [1, 96, 1, 60, 104],
                        }
                        for index in range(EXPECTED_CALL_SITES)
                    ]
                }
            audit = {
                "schema_version": QDQ_SCHEMA_VERSION,
                "passed": True,
                "complete": True,
                "errors": [],
                "structural": structural,
            }
            audit_path = root / "int8_audit_v5.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            manifest = {
                "int8_audit": {
                    "report_file": audit_path.name,
                    "report_sha256": audit_sha,
                },
                "cache": {
                    "bindings": [
                        {
                            "input_name": f"cache_in_{index}",
                            "output_name": f"cache_out_{index}",
                            "shape": [1, 96, 2, 60, 104],
                            "dtype": "float16",
                        }
                        for index in range(32)
                    ]
                },
            }
            contracts = _load_analysis_contracts(
                base_root=root,
                base_manifest=manifest,
            )
            self.assertEqual(
                contracts["analysis_schema_version"],
                FUSION_ANALYSIS_SCHEMA_VERSION,
            )
            self.assertEqual(contracts["int8_audit_sha256"], audit_sha)
            self.assertEqual(len(contracts["graphs"]["initial"]), 84)
            self.assertEqual(len(contracts["graphs"]["steady"]), 84)
            self.assertEqual(len(contracts["cache_shapes"]["initial"]), 32)
            self.assertEqual(len(contracts["cache_shapes"]["steady"]), 64)
            self.assertEqual(len(contracts["contract_sha256"]), 64)

            audit_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                _load_analysis_contracts(
                    base_root=root,
                    base_manifest=manifest,
                )

    def test_pad_metadata_constant_node_chain_is_static(self):
        class FakeHelper:
            @staticmethod
            def get_attribute_value(attribute):
                return attribute.value

            @staticmethod
            def tensor_dtype_to_np_dtype(value):
                if value != 7:
                    raise ValueError("unexpected fake ONNX dtype")
                return np.int64

        numpy_helper = SimpleNamespace(to_array=np.asarray)
        producer = {
            "pads_constant": SimpleNamespace(
                op_type="Constant",
                input=[],
                attribute=[
                    SimpleNamespace(
                        name="value",
                        value=np.asarray(
                            [0, 0, 2, 1, 1, 0, 0, 0, 1, 1],
                            dtype=np.int64,
                        ),
                    )
                ],
            ),
            "pads_cast": SimpleNamespace(
                op_type="Cast",
                input=["pads_constant"],
                attribute=[SimpleNamespace(name="to", value=7)],
            ),
        }
        value = _constant_array(
            name="pads_cast",
            initializers={},
            numpy_helper=numpy_helper,
            producer=producer,
            helper=FakeHelper,
            np_module=np,
            shapes={},
        )
        self.assertEqual(
            np.asarray(value).tolist(),
            [0, 0, 2, 1, 1, 0, 0, 0, 1, 1],
        )

        def constant(value):
            return SimpleNamespace(
                op_type="Constant",
                input=[],
                attribute=[SimpleNamespace(name="value", value=np.asarray(value))],
            )

        # This mirrors the legacy Torch exporter: extend the short PyTorch
        # padding vector to 2*rank, reshape/flip/transpose it, then cast it to
        # the INT64 vector consumed by ONNX Pad.
        producer.update(
            {
                "raw": constant([1, 1, 2, 0, 0, 0]),
                "raw_size": SimpleNamespace(
                    op_type="Size", input=["raw"], attribute=[]
                ),
                "activation_shape": SimpleNamespace(
                    op_type="Shape", input=["activation"], attribute=[]
                ),
                "rank": SimpleNamespace(
                    op_type="Size", input=["activation_shape"], attribute=[]
                ),
                "two": constant(2),
                "twice_rank": SimpleNamespace(
                    op_type="Mul", input=["rank", "two"], attribute=[]
                ),
                "extension": SimpleNamespace(
                    op_type="Sub",
                    input=["twice_rank", "raw_size"],
                    attribute=[],
                ),
                "axis": constant([0]),
                "extension_vector": SimpleNamespace(
                    op_type="Unsqueeze",
                    input=["extension", "axis"],
                    attribute=[],
                ),
                "zeros": SimpleNamespace(
                    op_type="ConstantOfShape",
                    input=["extension_vector"],
                    attribute=[
                        SimpleNamespace(
                            name="value", value=np.asarray([0], dtype=np.int64)
                        )
                    ],
                ),
                "extended": SimpleNamespace(
                    op_type="Concat",
                    input=["raw", "zeros"],
                    attribute=[SimpleNamespace(name="axis", value=0)],
                ),
                "matrix_shape": constant([5, 2]),
                "matrix": SimpleNamespace(
                    op_type="Reshape",
                    input=["extended", "matrix_shape"],
                    attribute=[],
                ),
                "reversed": SimpleNamespace(
                    op_type="Transpose",
                    input=["matrix"],
                    attribute=[SimpleNamespace(name="perm", value=[1, 0])],
                ),
                "flat_shape": constant([-1]),
                "flat": SimpleNamespace(
                    op_type="Reshape",
                    input=["reversed", "flat_shape"],
                    attribute=[],
                ),
                "exported_pads": SimpleNamespace(
                    op_type="Cast",
                    input=["flat"],
                    attribute=[SimpleNamespace(name="to", value=7)],
                ),
            }
        )
        exported = _constant_array(
            name="exported_pads",
            initializers={},
            numpy_helper=numpy_helper,
            producer=producer,
            helper=FakeHelper,
            np_module=np,
            shapes={"activation": [1, 16, 3, 60, 104]},
        )
        self.assertEqual(np.asarray(exported).shape, (10,))
        self.assertEqual(np.asarray(exported).dtype, np.dtype(np.int64))
        exported_from_rank_only = _constant_array(
            name="exported_pads",
            initializers={},
            numpy_helper=numpy_helper,
            producer=producer,
            helper=FakeHelper,
            np_module=np,
            shapes={},
            ranks={"activation": 5},
        )
        self.assertEqual(np.asarray(exported_from_rank_only).shape, (10,))
        failures = {}
        self.assertIsNone(
            _constant_array(
                name="runtime_pads",
                initializers={},
                numpy_helper=numpy_helper,
                producer={},
                helper=FakeHelper,
                np_module=np,
                shapes={},
                failures=failures,
            )
        )
        self.assertEqual(failures["runtime_pads"]["reason"], "no_constant_producer")

    def test_profile_schema_v2_classifies_each_fusion_plugin(self):
        self.assertEqual(TRT_LAYER_PROFILE_SCHEMA_VERSION, 2)
        self.assertEqual(SUPPORTED_TRT_LAYER_PROFILE_SCHEMA_VERSIONS, {1, 2})
        expected = {
            "fused_input_pack_quant",
            "fused_cache_update",
            "fused_conv1_norm_silu",
            "fused_conv2_residual",
        }
        self.assertTrue(expected <= set(PROFILE_CATEGORIES))
        catalog = build_physical_layer_catalog(
            engine_kind="initial",
            plan_sha256="a" * 64,
            inspector={
                "Layers": [
                    {
                        "Name": "fusion/input_pack_quant/call SfWanCausalPackQuantPlugin",
                        "LayerType": "PluginV3",
                    },
                    {
                        "Name": "fusion/cache_update/call SfWanCacheUpdatePlugin",
                        "LayerType": "PluginV3",
                    },
                    {
                        "Name": "fusion/conv1_norm_silu/call SfWanInt8EpiloguePlugin",
                        "LayerType": "PluginV3",
                    },
                    {
                        "Name": "fusion/conv2_residual/call SfWanInt8EpiloguePlugin",
                        "LayerType": "PluginV3",
                    },
                ]
            },
            precision="fp16",
        )
        self.assertEqual({entry["category"] for entry in catalog["layers"]}, expected)

        dual_catalog = build_physical_layer_catalog(
            engine_kind="steady",
            plan_sha256="b" * 64,
            inspector={
                "Layers": [
                    {
                        "Name": (
                            "fusion/input_pack_quant_cache/call "
                            "SfWanCausalPackQuantPlugin"
                        ),
                        "LayerType": "PluginV3",
                    }
                ]
            },
            precision="fp16",
        )
        self.assertEqual(
            dual_catalog["layers"][0]["category"], "fused_input_pack_quant"
        )

    def test_plugin_sources_pin_sm87_int8_chw32_and_fp16_cache(self):
        root = Path(__file__).resolve().parents[3]
        source_root = (
            root
            / "python"
            / "sglang"
            / "multimodal_gen"
            / "experimental"
            / "jetson_sfwan"
            / "trt_plugins"
        )
        cmake = (source_root / "CMakeLists.txt").read_text(encoding="utf-8")
        plugin = (source_root / "sfwan_vae_plugin.cpp").read_text(encoding="utf-8")
        kernels = (source_root / "sfwan_vae_kernels.cu").read_text(encoding="utf-8")
        self.assertIn('SFWAN_CUDA_ARCHITECTURES "87"', cmake)
        self.assertIn("TensorFormat::kCDHW32", plugin)
        self.assertIn("DataType::kINT8", plugin)
        self.assertIn("DataType::kHALF", plugin)
        self.assertIn("emit_cache", plugin)
        self.assertIn("mEmitCache != 0 ? 2 : 1", plugin)
        self.assertIn("SfWanCacheUpdatePlugin", plugin)
        self.assertIn("quantizeSigned", kernels)
        self.assertIn("cacheUpdateKernel", kernels)
        self.assertIn("normSiluEpilogueKernel", kernels)

    def test_production_compare_excludes_warmup_and_rejects_mismatches(self):
        summaries = {
            "fp32": self._production_summary(
                backend="pytorch",
                precision="fp32",
                variant="baseline",
                base_chunk_ms=100.0,
            ),
            "fp16_trt": self._production_summary(
                backend="tensorrt",
                precision="fp16",
                variant="baseline",
                base_chunk_ms=50.0,
            ),
            "int8_v5": self._production_summary(
                backend="tensorrt",
                precision="int8",
                variant="baseline",
                base_chunk_ms=40.0,
            ),
            "int8_fusion_v1": self._production_summary(
                backend="tensorrt",
                precision="int8",
                variant="fusion_v1",
                base_chunk_ms=30.0,
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for label, summary in summaries.items():
                path = root / f"{label}.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                inputs[label] = (path, summary)
            result = compare_profile_summaries(
                inputs,
                expected_warmup=1,
                expected_repeat=2,
            )
        self.assertEqual(result["results"]["fp32"]["whole_request"]["sample_count"], 2)
        self.assertLess(result["results"]["fp32"]["whole_request"]["mean_ms"], 10**9)
        self.assertGreater(
            result["results"]["int8_fusion_v1"]["speedup_vs_fp16_trt"], 1.0
        )
        self.assertIn("int8_fusion_v1", render_trt_perf_markdown(result))

        summaries["int8_v5"]["measurement_context"]["server"][
            "trt_layer_profile_enabled"
        ] = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for label, summary in summaries.items():
                path = root / f"{label}.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                inputs[label] = (path, summary)
            with self.assertRaisesRegex(ValueError, "must be disabled"):
                compare_profile_summaries(
                    inputs,
                    expected_warmup=1,
                    expected_repeat=2,
                )


if __name__ == "__main__":
    unittest.main(verbosity=3)
