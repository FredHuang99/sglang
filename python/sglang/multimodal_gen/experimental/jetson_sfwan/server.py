"""FastAPI entrypoint for monolithic, DiT, and VAE SFWan roles."""

from __future__ import annotations

import argparse
import asyncio
import functools
import math
import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import msgspec
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from .engine import (
    AsyncChunkSender,
    JobRecord,
    SingleWorkerEngine,
    VaeJobRecord,
    chunk_digest,
)
from .model import (
    LatentChunk,
    ModelLoadConfig,
    SfWanDitModel,
    SfWanMonolithicModel,
    SfWanVaeModel,
    NativeInt8V2Level,
    VaePrecision,
    VaeTrtVariant,
    _uses_trt_vae,
)
from .protocol import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MODEL_PATH,
    DitProfileRequest,
    EngineStatus,
    GenerationRequest,
    HealthResponse,
    JobState,
    JobStatus,
    LatentJobRegistrationResponse,
    LatentJobSpec,
    LatentTransport,
    MAX_SAFETENSORS_HEADER_BYTES,
    SafetensorsLatentPayload,
    SharedMemoryChunkReady,
    SharedMemoryDescriptor,
    SubmissionResponse,
    decoded_frames_for_chunk,
    materialize_latent_tensor_for_cpu,
    parse_latent_safetensors_payload,
)
from .transport import (
    PinnedH2DLoader,
    SharedMemoryChunkRef,
    SharedMemoryChunkSender,
    SharedMemoryRegion,
    StagedDeviceChunk,
    stage_shared_chunk_to_device,
)

Role = Literal["monolithic", "dit", "vae"]
ModelFactory = Callable[[ModelLoadConfig], Any]


class ServerConfig(msgspec.Struct, frozen=True, kw_only=True):
    role: Role
    model_path: str = DEFAULT_MODEL_PATH
    host: str = "0.0.0.0"
    port: int = 30000
    public_url: str | None = None
    vae_url: str | None = None
    output_dir: str = "sfwan_outputs"
    max_pixels: int = DEFAULT_MAX_PIXELS
    transfer_queue_depth: int = 2
    chunk_timeout_seconds: float = 600.0
    device_index: int = 0
    vae_precision: VaePrecision = "fp32"
    vae_engine_dir: str | None = None
    vae_trt_variant: VaeTrtVariant = "baseline"
    native_int8_v2_level: NativeInt8V2Level = "p1"
    text_encoder_cpu_offload: bool = True
    dit_cpu_offload: bool = False
    vae_cpu_offload: bool = False
    latent_transport: LatentTransport = "http"
    enable_profile: bool = False
    enable_trt_layer_profile: bool = False
    enable_native_int8_kernel_profile: bool = False
    enable_nvtx: bool = False


def _write_mp4(
    *,
    frame_chunks: list[Any],
    output_path: Path,
    fps: int,
) -> float:
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        quality=5,
    )
    try:
        for chunk in frame_chunks:
            for frame in chunk:
                writer.append_data(frame)
    finally:
        writer.close()
    return (time.perf_counter() - start) * 1000


class SfWanRuntime:
    """Own the one model thread, one FCFS worker, and optional chunk sender."""

    def __init__(
        self,
        *,
        config: ServerConfig,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if config.transfer_queue_depth <= 0:
            raise ValueError("--transfer-queue-depth must be positive")
        if config.chunk_timeout_seconds <= 0:
            raise ValueError("--chunk-timeout-seconds must be positive")
        if config.enable_trt_layer_profile:
            if not config.enable_profile:
                raise ValueError("--enable-trt-layer-profile requires --enable-profile")
            if config.role != "vae":
                raise ValueError(
                    "--enable-trt-layer-profile is valid only for --role vae"
                )
            if not _uses_trt_vae(config.vae_precision):
                raise ValueError(
                    "--enable-trt-layer-profile requires --vae-precision "
                    "fp16_trt or int8_trt"
                )
        if config.enable_native_int8_kernel_profile:
            if not config.enable_profile:
                raise ValueError(
                    "--enable-native-int8-kernel-profile requires --enable-profile"
                )
            if config.role != "vae":
                raise ValueError(
                    "--enable-native-int8-kernel-profile is valid only for "
                    "--role vae"
                )
            if config.vae_precision != "int8_trt":
                raise ValueError(
                    "--enable-native-int8-kernel-profile requires "
                    "--vae-precision int8_trt"
                )
            if config.vae_trt_variant not in {"native_int8_v1", "native_int8_v2"}:
                raise ValueError(
                    "--enable-native-int8-kernel-profile requires "
                    "--vae-trt-variant native_int8_v1 or native_int8_v2"
                )
        if config.role == "monolithic" and config.vae_url is not None:
            raise ValueError("--vae-url is not valid for the monolithic role")
        if config.role in {"monolithic", "vae"}:
            if _uses_trt_vae(config.vae_precision):
                if config.vae_engine_dir is None:
                    raise ValueError(
                        f"--vae-precision {config.vae_precision} requires "
                        "--vae-engine-dir"
                    )
                if config.vae_cpu_offload:
                    raise ValueError(
                        "TensorRT VAE precision does not support --vae-cpu-offload"
                    )
            elif config.vae_engine_dir is not None:
                raise ValueError(
                    "--vae-engine-dir is valid only with fp16_trt or int8_trt"
                )
        if config.vae_trt_variant in {
            "fusion_v1",
            "fusion_v2",
            "native_int8_v1",
            "native_int8_v2",
        }:
            if config.vae_precision != "int8_trt":
                raise ValueError(
                    f"--vae-trt-variant {config.vae_trt_variant} requires "
                    "--vae-precision int8_trt"
                )
            if config.role not in {"monolithic", "vae"}:
                raise ValueError(
                    f"--vae-trt-variant {config.vae_trt_variant} is valid only "
                    "for VAE execution"
                )
        elif config.vae_trt_variant != "baseline":
            raise ValueError(
                f"unsupported TensorRT VAE variant: {config.vae_trt_variant}"
            )
        if (
            config.role == "vae"
            and config.latent_transport == "shm"
            and os.name != "posix"
        ):
            raise ValueError("--latent-transport shm requires a Linux CUDA host")
        self.config = config
        self._model_factory = model_factory
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"sfwan-{config.role}-model",
        )
        self._transfer_executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sfwan-vae-transfer",
            )
            if config.role == "vae"
            else None
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self.model: Any = None
        self.model_loaded = False
        self.model_load_ms: float | None = None
        self.jobs: dict[str, JobRecord] = {}
        self.engine: SingleWorkerEngine | None = None
        self.sender: Any = None
        self._h2d_loader: PinnedH2DLoader | None = None
        self._shm_regions: dict[str, SharedMemoryRegion] = {}
        self._shm_descriptors: dict[str, SharedMemoryDescriptor] = {}
        self._shm_h2d_stream: Any = None
        self._closed = False

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        load_config = ModelLoadConfig(
            model_path=self.config.model_path,
            device_index=self.config.device_index,
            vae_precision=self.config.vae_precision,
            vae_engine_dir=self.config.vae_engine_dir,
            vae_trt_variant=self.config.vae_trt_variant,
            native_int8_v2_level=self.config.native_int8_v2_level,
            text_encoder_cpu_offload=self.config.text_encoder_cpu_offload,
            dit_cpu_offload=self.config.dit_cpu_offload,
            vae_cpu_offload=self.config.vae_cpu_offload,
            enable_profile=self.config.enable_profile,
            enable_trt_layer_profile=self.config.enable_trt_layer_profile,
            enable_native_int8_kernel_profile=(
                self.config.enable_native_int8_kernel_profile
            ),
            enable_nvtx=self.config.enable_nvtx,
        )
        factory = self._model_factory or self._default_model_factory()
        load_start = time.perf_counter()
        self.model = await self._loop.run_in_executor(
            self._executor,
            functools.partial(factory, load_config),
        )
        self.model_load_ms = (time.perf_counter() - load_start) * 1000
        self.model_loaded = True

        handler = {
            "monolithic": self._handle_monolithic_job,
            "dit": self._handle_dit_job,
            "vae": self._handle_vae_job,
        }[self.config.role]
        self.engine = SingleWorkerEngine(
            role=self.config.role,
            handler=handler,
        )
        await self.engine.start()

        if self.config.role == "dit" and self.config.vae_url is not None:
            sender_class = (
                AsyncChunkSender
                if self.config.latent_transport == "http"
                else SharedMemoryChunkSender
            )
            self.sender = sender_class(
                vae_url=self.config.vae_url,
                queue_depth=self.config.transfer_queue_depth,
                enable_profile=self.config.enable_profile,
                event_callback=self._add_event,
                failure_callback=self._mark_transfer_failed,
            )
            await self.sender.start()
        elif self.config.role == "vae":
            model_device = getattr(self.model, "device", None)
            if getattr(model_device, "type", None) == "cuda":
                if self._transfer_executor is None:
                    raise RuntimeError("VAE transfer executor is unavailable")
                self._h2d_loader = PinnedH2DLoader(
                    device=model_device,
                    depth=self.config.transfer_queue_depth,
                    executor=self._transfer_executor,
                    enable_profile=self.config.enable_profile,
                )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.engine is not None:
            await self.engine.close()
            self.engine = None
        if self.sender is not None:
            await self.sender.close()
            self.sender = None
        if self._h2d_loader is not None:
            self._h2d_loader.close()
            self._h2d_loader = None
        for region in list(self._shm_regions.values()):
            region.close()
        self._shm_regions.clear()
        self._shm_descriptors.clear()
        self._shm_h2d_stream = None
        model = self.model
        self.model = None
        self.model_loaded = False
        model_close = getattr(model, "close", None)
        if callable(model_close):
            await self._loop.run_in_executor(self._executor, model_close)
        await asyncio.to_thread(self._executor.shutdown, True)
        if self._transfer_executor is not None:
            await asyncio.to_thread(self._transfer_executor.shutdown, True)
            self._transfer_executor = None

    def _default_model_factory(self) -> ModelFactory:
        model_class = {
            "monolithic": SfWanMonolithicModel,
            "dit": SfWanDitModel,
            "vae": SfWanVaeModel,
        }[self.config.role]

        def _factory(load_config: ModelLoadConfig) -> Any:
            return model_class(load_config=load_config)

        return _factory

    async def submit_generation(
        self,
        request: GenerationRequest,
    ) -> JobRecord:
        if self.config.role == "vae":
            raise ValueError("the VAE role does not accept text generation requests")
        if self.config.role == "dit" and self.sender is None:
            raise ValueError(
                "the DiT role requires --vae-url for normal generation; "
                "DiT-only profiling requires --enable-profile and "
                "POST /v1/dit-profiles"
            )
        request.validate_server_limits(self.config.max_pixels)
        if (
            self.config.role == "monolithic"
            and _uses_trt_vae(self.config.vae_precision)
            and (request.height, request.width) != (480, 832)
        ):
            raise ValueError("TensorRT VAE V1 supports only 480x832 generation")
        request_id = str(uuid.uuid4())
        record = JobRecord(
            request_id=request_id,
            role=self.config.role,
            payload=request,
        )
        for warning in request.warnings():
            record.add_event("warning", message=warning)
        record.metrics["model_load_ms"] = self.model_load_ms
        self.jobs[request_id] = record
        if self.engine is None:
            raise RuntimeError("engine has not started")
        await self.engine.submit(record)
        return record

    async def submit_dit_profile(
        self,
        request: DitProfileRequest,
    ) -> JobRecord:
        if self.config.role != "dit":
            raise ValueError("DiT profiles are accepted only by the DiT role")
        if not self.config.enable_profile:
            raise ValueError("DiT profiling requires the server flag --enable-profile")
        request.validate_server_limits(self.config.max_pixels)
        request_id = f"dit-profile-{uuid.uuid4()}"
        record = JobRecord(
            request_id=request_id,
            role="dit",
            payload=request,
        )
        record.metrics.update(
            {
                "model_load_ms": self.model_load_ms,
                "profile": "dit",
                "profile_warmup": request.profile_warmup,
                "profile_iteration": request.iteration,
            }
        )
        for warning in request.warnings():
            record.add_event("warning", message=warning)
        self.jobs[request_id] = record
        if self.engine is None:
            raise RuntimeError("engine has not started")
        await self.engine.submit(record)
        return record

    async def register_latent_job(
        self,
        spec: LatentJobSpec,
    ) -> tuple[VaeJobRecord, bool, SharedMemoryDescriptor | None]:
        if self.config.role != "vae":
            raise ValueError("latent jobs are accepted only by the VAE role")
        if spec.source == "profile" and not self.config.enable_profile:
            raise ValueError("VAE profiling requires the server flag --enable-profile")
        if (
            self.config.enable_trt_layer_profile
            or self.config.enable_native_int8_kernel_profile
        ) and spec.source != "profile":
            raise ValueError(
                "a diagnostic TensorRT VAE profile server accepts "
                "source=profile jobs only"
            )
        if spec.transport != self.config.latent_transport:
            raise ValueError(
                f"VAE server accepts transport={self.config.latent_transport}, "
                f"not {spec.transport}"
            )
        if (
            self.config.max_pixels > 0
            and spec.height * spec.width > self.config.max_pixels
        ):
            raise ValueError("latent job exceeds server max_pixels")
        if _uses_trt_vae(self.config.vae_precision) and (
            spec.height,
            spec.width,
        ) != (480, 832):
            raise ValueError("TensorRT VAE V1 accepts only 480x832 latent jobs")

        existing = self.jobs.get(spec.request_id)
        if existing is not None:
            if not isinstance(existing, VaeJobRecord) or existing.spec != spec:
                raise ValueError(
                    f"request_id={spec.request_id} already exists with "
                    "different metadata"
                )
            region = self._shm_regions.get(spec.request_id)
            return (
                existing,
                False,
                None if region is None else region.descriptor,
            )

        region = None
        if spec.transport == "shm":
            if self._h2d_loader is None:
                raise ValueError("shared-memory VAE transport requires a CUDA model")
            if self._transfer_executor is None:
                raise RuntimeError("VAE transfer executor is unavailable")
            loop = asyncio.get_running_loop()
            region = await loop.run_in_executor(
                self._transfer_executor,
                functools.partial(SharedMemoryRegion.create, spec=spec),
            )
        record = VaeJobRecord(spec=spec)
        record.metrics["model_load_ms"] = self.model_load_ms
        record.metrics["latent_transport"] = spec.transport
        self.jobs[spec.request_id] = record
        if region is not None:
            self._shm_regions[spec.request_id] = region
            self._shm_descriptors[spec.request_id] = region.descriptor
        if self.engine is None:
            self.jobs.pop(spec.request_id, None)
            if region is not None:
                self._shm_regions.pop(spec.request_id, None)
                self._shm_descriptors.pop(spec.request_id, None)
                region.close()
            raise RuntimeError("engine has not started")
        try:
            await self.engine.submit(record)
        except BaseException:
            self.jobs.pop(spec.request_id, None)
            if region is not None:
                self._shm_regions.pop(spec.request_id, None)
                self._shm_descriptors.pop(spec.request_id, None)
                region.close()
            raise
        return record, True, None if region is None else region.descriptor

    async def put_latent_chunk(
        self,
        *,
        request_id: str,
        chunk_index: int,
        payload: bytes,
    ) -> bool:
        record = self.jobs.get(request_id)
        if not isinstance(record, VaeJobRecord):
            raise KeyError(request_id)
        if record.spec.transport != "http":
            raise ValueError("safetensors PUT is valid only for HTTP latent jobs")
        expected_numel = math.prod(record.spec.latent_chunk_shape)
        max_payload_bytes = expected_numel * 2 + 8 + MAX_SAFETENSORS_HEADER_BYTES
        if len(payload) > max_payload_bytes:
            raise ValueError(
                f"payload has {len(payload)} bytes; expected at most "
                f"{max_payload_bytes}"
            )
        parse_start = time.perf_counter() if self.config.enable_profile else None
        parsed = await asyncio.to_thread(
            parse_latent_safetensors_payload,
            payload,
            expected_shape=record.spec.latent_chunk_shape,
        )
        parse_ms = (
            (time.perf_counter() - parse_start) * 1000
            if parse_start is not None
            else None
        )
        ingress_metrics: dict[str, Any] = {"payload_bytes": len(payload)}
        if parse_ms is not None:
            ingress_metrics["safetensors_parse_ms"] = parse_ms
        return await record.put_chunk(
            chunk_index=chunk_index,
            data=parsed,
            digest=chunk_digest(payload),
            ingress_metrics=ingress_metrics,
        )

    async def ready_shared_memory_chunk(
        self,
        *,
        request_id: str,
        chunk_index: int,
        ready: SharedMemoryChunkReady,
    ) -> bool:
        record = self.jobs.get(request_id)
        if not isinstance(record, VaeJobRecord):
            raise KeyError(request_id)
        if record.spec.transport != "shm":
            raise ValueError("shared-memory ready is valid only for SHM latent jobs")
        descriptor = self._shm_descriptors.get(request_id)
        if descriptor is None:
            raise RuntimeError("shared-memory descriptor is unavailable")
        if ready.lease_token != descriptor.lease_token:
            raise ValueError("shared-memory lease token does not match")
        existing_digest = await record.existing_chunk_digest(chunk_index)
        if existing_digest is not None:
            if existing_digest != ready.digest:
                raise ValueError(
                    f"chunk {chunk_index} has a shared-memory digest conflict"
                )
            return False

        region = self._shm_regions.get(request_id)
        if region is None:
            raise RuntimeError("shared-memory mapping is unavailable")

        digest_start = time.perf_counter() if self.config.enable_profile else None
        digest = await asyncio.to_thread(region.digest, chunk_index)
        digest_ms = (
            (time.perf_counter() - digest_start) * 1000
            if digest_start is not None
            else None
        )
        if digest != ready.digest:
            raise ValueError("shared-memory chunk digest does not match")
        ingress_metrics: dict[str, Any] = {"shared_memory_ready": True}
        if digest_ms is not None:
            ingress_metrics["shared_digest_ms"] = digest_ms
        return await record.put_chunk(
            chunk_index=chunk_index,
            data=SharedMemoryChunkRef(
                region=region,
                chunk_index=chunk_index,
            ),
            digest=digest,
            ingress_metrics=ingress_metrics,
        )

    async def cancel_latent_job(self, request_id: str) -> None:
        record = self.jobs.get(request_id)
        if not isinstance(record, VaeJobRecord):
            raise KeyError(request_id)
        was_running = record.state == JobState.RUNNING
        record.mark_cancelled("cancelled by upstream DiT server")
        await record.clear_chunks_and_wake_waiters()
        # A running handler may have an H2D already in flight. Its finally
        # block cancels/joins the prefetch before unregistering the mapping.
        if not was_running:
            self._cleanup_shm_region(request_id)

    def job_status(self, request_id: str) -> JobStatus:
        try:
            return self.jobs[request_id].to_status()
        except KeyError as exc:
            raise KeyError(request_id) from exc

    def engine_status(self) -> EngineStatus:
        contract = dict(getattr(self.model, "contract", {}))
        contract["profile_enabled"] = self.config.enable_profile
        contract["trt_layer_profile_enabled"] = self.config.enable_trt_layer_profile
        contract["native_int8_kernel_profile_enabled"] = (
            self.config.enable_native_int8_kernel_profile
        )
        if self.engine is None:
            return EngineStatus(
                role=self.config.role,
                model_loaded=self.model_loaded,
                waiting_count=0,
                waiting_ids=[],
                running_ids=[],
                contract=contract,
            )
        snapshot = self.engine.snapshot()
        return snapshot.model_copy(
            update={
                "model_loaded": self.model_loaded,
                "contract": contract,
            }
        )

    def _cleanup_shm_region(self, request_id: str) -> None:
        region = self._shm_regions.pop(request_id, None)
        if region is not None:
            region.close()

    async def _add_event(
        self,
        request_id: str,
        kind: str,
        details: dict[str, Any],
    ) -> None:
        record = self.jobs.get(request_id)
        if record is not None:
            record.add_event(kind, **details)

    async def _mark_transfer_failed(
        self,
        request_id: str,
        error: str,
    ) -> None:
        record = self.jobs.get(request_id)
        if record is not None:
            record.mark_failed(f"latent transfer failed: {error}")

    def _add_event_from_model_thread(
        self,
        *,
        request_id: str,
        kind: str,
        details: dict[str, Any],
    ) -> None:
        if self._loop is None:
            raise RuntimeError("runtime event loop is unavailable")
        future = asyncio.run_coroutine_threadsafe(
            self._add_event(request_id, kind, details),
            self._loop,
        )
        future.result()

    async def _handle_monolithic_job(self, record: JobRecord) -> None:
        if not isinstance(record.payload, GenerationRequest):
            raise TypeError("monolithic engine received a non-generation request")
        if self._loop is None:
            raise RuntimeError("runtime event loop is unavailable")

        def _on_chunk(decoded: Any) -> None:
            if record.state in {JobState.CANCELLED, JobState.FAILED}:
                raise RuntimeError(record.error or "generation was cancelled")
            details = {
                "chunk_index": decoded.chunk_index,
                "decoded_rgb_frames": decoded.frame_count,
            }
            if self.config.enable_profile:
                details["metrics"] = decoded.metrics
            self._add_event_from_model_thread(
                request_id=record.request_id,
                kind="monolithic_chunk_completed",
                details=details,
            )

        result = await self._loop.run_in_executor(
            self._executor,
            functools.partial(
                self.model.generate,
                request=record.payload,
                chunk_callback=_on_chunk,
            ),
        )
        frame_chunks = [chunk.frames for chunk in result.chunks]
        self._validate_decoded_frame_count(
            request_id=record.request_id,
            frame_chunks=frame_chunks,
            expected_frames=record.payload.resolved_num_frames,
        )
        output_path = self._output_path(record.request_id)
        encode_ms = await self._loop.run_in_executor(
            self._executor,
            functools.partial(
                _write_mp4,
                frame_chunks=frame_chunks,
                output_path=output_path,
                fps=record.payload.fps,
            ),
        )
        record.output_path = str(output_path)
        record.metrics.update(result.metrics)
        record.metrics["mp4_encode_ms"] = encode_ms
        record.add_event("mp4_completed", encode_ms=encode_ms)

    async def _handle_dit_job(self, record: JobRecord) -> None:
        if not isinstance(record.payload, GenerationRequest):
            raise TypeError("DiT engine received a non-generation request")
        if self._loop is None:
            raise RuntimeError("runtime event loop is unavailable")

        if isinstance(record.payload, DitProfileRequest):

            def _discard_after_clean_kv(latent_chunk: LatentChunk) -> None:
                details = {"chunk_index": latent_chunk.chunk_index}
                if self.config.enable_profile:
                    details["metrics"] = latent_chunk.metrics
                self._add_event_from_model_thread(
                    request_id=record.request_id,
                    kind="dit_profile_chunk_completed",
                    details=details,
                )

            metrics = await self._loop.run_in_executor(
                self._executor,
                functools.partial(
                    self.model.generate,
                    request=record.payload,
                    chunk_callback=_discard_after_clean_kv,
                ),
            )
            execution = metrics.get("profile_execution")
            if not isinstance(execution, dict):
                raise RuntimeError(
                    "profile-enabled DiT returned no profile_execution metrics"
                )
            execution_chunks = execution.get("chunks")
            if not isinstance(execution_chunks, list) or [
                chunk.get("chunk_index") for chunk in execution_chunks
            ] != list(range(record.payload.total_chunks)):
                raise RuntimeError(
                    "DiT profile execution did not report every chunk in order"
                )
            if any(
                len(chunk.get("denoise_steps", ())) != 4
                or "clean_kv_cuda_ms" not in chunk
                for chunk in execution_chunks
            ):
                raise RuntimeError(
                    "each DiT profile chunk must report four DMD steps "
                    "and one clean-KV timing"
                )
            record.metrics.update(metrics)
            record.add_event("dit_profile_completed")
            return

        if self.sender is None:
            raise RuntimeError("DiT chunk sender is unavailable")

        spec = LatentJobSpec.from_generation(
            record.request_id,
            record.payload,
            transport=self.config.latent_transport,
        )

        def _after_clean_kv(latent_chunk: LatentChunk) -> None:
            if record.state in {JobState.CANCELLED, JobState.FAILED}:
                raise RuntimeError(record.error or "latent transfer failed")
            details = {"chunk_index": latent_chunk.chunk_index}
            if self.config.enable_profile:
                details["metrics"] = latent_chunk.metrics
            self._add_event_from_model_thread(
                request_id=record.request_id,
                kind="dit_chunk_clean_kv_completed",
                details=details,
            )
            pending = self.sender.stage_tensor(
                request_id=record.request_id,
                chunk_index=latent_chunk.chunk_index,
                tensor=latent_chunk.tensor,
            )
            enqueue_future = asyncio.run_coroutine_threadsafe(
                self.sender.enqueue(pending),
                self._loop,
            )
            enqueue_future.result()

        try:
            await self.sender.register_job(spec)
            metrics = await self._loop.run_in_executor(
                self._executor,
                functools.partial(
                    self.model.generate,
                    request=record.payload,
                    chunk_callback=_after_clean_kv,
                ),
            )
            await self.sender.wait_for_job(
                record.request_id,
                is_cancelled=lambda: (
                    record.state in {JobState.CANCELLED, JobState.FAILED}
                ),
            )
        except Exception:
            await self.sender.cancel_remote_job(record.request_id)
            raise
        finally:
            self.sender.release_job(record.request_id)
        record.metrics.update(metrics)
        record.add_event("all_latent_chunks_accepted")

    async def _handle_vae_job(self, record: JobRecord) -> None:
        if not isinstance(record, VaeJobRecord):
            raise TypeError("VAE engine received a non-latent job")
        if self._loop is None:
            raise RuntimeError("runtime event loop is unavailable")

        profile_enabled = self.config.enable_profile
        frame_chunks: list[Any] = []
        chunk_metrics = []
        execution_chunks = []
        decoded_frame_total = 0
        vae_decode_total_ms = 0.0 if profile_enabled else None
        running_start = time.perf_counter() if profile_enabled else None
        previous_chunk_completed_time = running_start
        reset_attempted = False
        reset_completed = False
        reset_ms = None
        prepared_task: asyncio.Task[tuple[Any, dict[str, Any], float | None]] | None = (
            None
        )

        async def _stage_without_cancellation_leak(awaitable: Any) -> Any:
            stage_task = asyncio.ensure_future(awaitable)
            try:
                return await asyncio.shield(stage_task)
            except asyncio.CancelledError:
                # An executor/CUDA copy cannot be cancelled once launched. Wait
                # for it and return the bounded slot before propagating cancel.
                staged_after_cancel = await stage_task
                if isinstance(staged_after_cancel, StagedDeviceChunk):
                    staged_after_cancel.release()
                raise

        async def _prepare_chunk(
            chunk_index: int,
        ) -> tuple[Any, dict[str, Any], float | None]:
            wait_start = time.perf_counter() if profile_enabled else None
            received = await record.wait_for_chunk(
                chunk_index=chunk_index,
                timeout_seconds=self.config.chunk_timeout_seconds,
            )
            chunk_wait_ms = (
                (time.perf_counter() - wait_start) * 1000
                if wait_start is not None
                else None
            )
            staged = received.data
            ingress_metrics = dict(received.ingress_metrics)
            try:
                if record.spec.transport == "http":
                    if self._h2d_loader is not None and not isinstance(
                        staged, StagedDeviceChunk
                    ):
                        if not isinstance(staged, SafetensorsLatentPayload):
                            raise TypeError(
                                "HTTP ingress did not contain a safetensors payload"
                            )
                        staged = await _stage_without_cancellation_leak(
                            self._h2d_loader.stage(staged)
                        )
                        ingress_metrics.update(staged.ingress_metrics)
                    elif self._h2d_loader is None and isinstance(
                        staged, SafetensorsLatentPayload
                    ):
                        if self._transfer_executor is None:
                            raise RuntimeError("VAE transfer executor is unavailable")
                        staged = await self._loop.run_in_executor(
                            self._transfer_executor,
                            materialize_latent_tensor_for_cpu,
                            staged,
                        )
                elif record.spec.transport == "shm":
                    if not isinstance(staged, SharedMemoryChunkRef):
                        raise TypeError("SHM ingress did not contain a shared slot")
                    if self._h2d_loader is None:
                        raise RuntimeError("shared-memory H2D requires a CUDA VAE")
                    if self._transfer_executor is None:
                        raise RuntimeError("VAE transfer executor is unavailable")

                    def _stage_shared() -> StagedDeviceChunk:
                        device_chunk, stream = stage_shared_chunk_to_device(
                            region=staged.region,
                            chunk_index=staged.chunk_index,
                            device=self._h2d_loader.device,
                            copy_stream=self._shm_h2d_stream,
                            enable_profile=profile_enabled,
                        )
                        self._shm_h2d_stream = stream
                        return device_chunk

                    staged = await _stage_without_cancellation_leak(
                        self._loop.run_in_executor(
                            self._transfer_executor,
                            _stage_shared,
                        )
                    )
                    ingress_metrics.update(staged.ingress_metrics)
                return staged, ingress_metrics, chunk_wait_ms
            except BaseException:
                if isinstance(staged, StagedDeviceChunk):
                    staged.release()
                raise

        async def _release_prefetched_task() -> None:
            nonlocal prepared_task
            if prepared_task is None:
                return
            task = prepared_task
            prepared_task = None
            if not task.done():
                task.cancel()
            try:
                staged, _metrics, _wait_ms = await task
            except BaseException:
                return
            if isinstance(staged, StagedDeviceChunk):
                staged.release()

        try:
            reset_start = time.perf_counter() if profile_enabled else None
            reset_attempted = True
            await self._loop.run_in_executor(
                self._executor,
                self.model.reset_request,
            )
            reset_completed = True
            if reset_start is not None:
                reset_ms = (time.perf_counter() - reset_start) * 1000
            prepared_task = asyncio.create_task(
                _prepare_chunk(0),
                name=f"sfwan-vae-{record.request_id}-chunk-0",
            )
            for chunk_index in range(int(record.spec.total_chunks)):
                if prepared_task is None:
                    raise RuntimeError("VAE chunk prefetch task is unavailable")
                staged, ingress_metrics, chunk_wait_ms = await prepared_task
                prepared_task = None
                next_chunk_index = chunk_index + 1
                if next_chunk_index < int(record.spec.total_chunks):
                    prepared_task = asyncio.create_task(
                        _prepare_chunk(next_chunk_index),
                        name=(
                            f"sfwan-vae-{record.request_id}-chunk-{next_chunk_index}"
                        ),
                    )
                start_details: dict[str, Any] = {"chunk_index": chunk_index}
                if chunk_wait_ms is not None:
                    start_details["chunk_wait_ms"] = chunk_wait_ms
                record.add_event("vae_chunk_started", **start_details)

                def _decode_staged() -> Any:
                    latents = (
                        staged.wait_on_current_stream()
                        if isinstance(staged, StagedDeviceChunk)
                        else staged
                    )
                    return self.model.decode_chunk(
                        chunk_index=chunk_index,
                        latents=latents,
                        return_frames=not record.spec.discard_output,
                    )

                try:
                    decode_start = time.perf_counter() if profile_enabled else None
                    decoded = await self._loop.run_in_executor(
                        self._executor,
                        _decode_staged,
                    )
                    decode_wall_ms = (
                        (time.perf_counter() - decode_start) * 1000
                        if decode_start is not None
                        else None
                    )
                finally:
                    if isinstance(staged, StagedDeviceChunk):
                        ingress_metrics.update(staged.completed_metrics())
                        staged.release()
                if record.state in {JobState.CANCELLED, JobState.FAILED}:
                    raise RuntimeError(record.error or "VAE job was cancelled")
                if profile_enabled:
                    assert vae_decode_total_ms is not None
                    assert decode_wall_ms is not None
                    vae_decode_total_ms += decode_wall_ms
                actual_frames = int(decoded.frame_count)
                expected_frames = decoded_frames_for_chunk(chunk_index)
                if actual_frames != expected_frames:
                    raise RuntimeError(
                        f"chunk {chunk_index} decoded {actual_frames} frames; "
                        f"expected {expected_frames}"
                    )
                decoded_frame_total += actual_frames
                if not record.spec.discard_output:
                    if decoded.frames is None:
                        raise RuntimeError("VAE did not return frames for saved output")
                    frame_chunks.append(decoded.frames)
                metrics: dict[str, Any] = {
                    "chunk_index": chunk_index,
                    "decoded_rgb_frames": actual_frames,
                }
                if profile_enabled:
                    assert running_start is not None
                    assert previous_chunk_completed_time is not None
                    assert chunk_wait_ms is not None
                    assert decode_wall_ms is not None
                    chunk_completed_time = time.perf_counter()
                    metrics.update(decoded.metrics)
                    metrics["chunk_wait_ms"] = chunk_wait_ms
                    metrics["decode_wall_ms"] = decode_wall_ms
                    metrics["chunk_service_interval_ms"] = (
                        chunk_completed_time - previous_chunk_completed_time
                    ) * 1000
                    metrics["running_elapsed_ms"] = (
                        chunk_completed_time - running_start
                    ) * 1000
                    metrics["transfer"] = ingress_metrics
                    previous_chunk_completed_time = chunk_completed_time
                    chunk_execution = decoded.metrics.get("profile_execution")
                    if chunk_execution is None:
                        raise RuntimeError(
                            "profile-enabled VAE decode returned no "
                            "profile_execution metrics"
                        )
                    if int(chunk_execution.get("chunk_index", -1)) != chunk_index:
                        raise RuntimeError(
                            "VAE profile execution chunk index does not match "
                            f"the FCFS decode index {chunk_index}"
                        )
                    execution_chunks.append(dict(chunk_execution))
                chunk_metrics.append(metrics)
                completed_details: dict[str, Any] = {
                    "chunk_index": chunk_index,
                    "decoded_rgb_frames": actual_frames,
                }
                if profile_enabled:
                    completed_details["metrics"] = metrics
                record.add_event(
                    "vae_chunk_completed",
                    **completed_details,
                )
        finally:
            try:
                await _release_prefetched_task()
                finish_ms = None
                if reset_attempted:
                    finish_start = time.perf_counter() if profile_enabled else None
                    await self._loop.run_in_executor(
                        self._executor,
                        self.model.finish_request,
                    )
                    if finish_start is not None:
                        finish_ms = (time.perf_counter() - finish_start) * 1000
            finally:
                await record.clear_chunks_and_wake_waiters()
                self._cleanup_shm_region(record.request_id)
                record.metrics.update(
                    {
                        "source": record.spec.source,
                        "profile_warmup": record.spec.profile_warmup,
                        "decoded_rgb_frames": decoded_frame_total,
                        "num_chunks": int(record.spec.total_chunks),
                    }
                )
                if profile_enabled:
                    assert running_start is not None
                    running_total_ms = (time.perf_counter() - running_start) * 1000
                    profile_execution = {
                        "component": "vae",
                        "num_chunks": int(record.spec.total_chunks),
                        "chunks": execution_chunks,
                        "vae_execution_cuda_ms": sum(
                            float(chunk["chunk_execution_cuda_ms"])
                            for chunk in execution_chunks
                        ),
                    }
                    if self.config.enable_trt_layer_profile:
                        layer_profile_metadata = getattr(
                            self.model,
                            "trt_layer_profile_metadata",
                            None,
                        )
                        if layer_profile_metadata is None:
                            raise RuntimeError(
                                "TensorRT layer profiling completed without both "
                                "initial and steady physical-layer catalogs"
                            )
                        profile_execution["trt_layer_profile_metadata"] = (
                            layer_profile_metadata
                        )
                    record.metrics.update(
                        {
                            "vae_reset_ms": (reset_ms if reset_completed else None),
                            "vae_finish_ms": (finish_ms if reset_attempted else None),
                            "vae_decode_total_ms": vae_decode_total_ms,
                            "vae_running_total_ms": running_total_ms,
                            "first_9_frames_ms": (
                                chunk_metrics[0]["chunk_service_interval_ms"]
                                if chunk_metrics
                                else None
                            ),
                            "first_9_frames_decoder_ms": (
                                chunk_metrics[0]["decode_wall_ms"]
                                if chunk_metrics
                                else None
                            ),
                            "steady_chunk_ms": [
                                metrics["chunk_service_interval_ms"]
                                for metrics in chunk_metrics[1:]
                            ],
                            "steady_chunk_decoder_ms": [
                                metrics["decode_wall_ms"]
                                for metrics in chunk_metrics[1:]
                            ],
                            "chunks": chunk_metrics,
                            "profile_execution": profile_execution,
                        }
                    )

        if decoded_frame_total != record.spec.num_frames:
            raise RuntimeError(
                f"request {record.request_id} produced "
                f"{decoded_frame_total} frames; expected {record.spec.num_frames}"
            )
        if not record.spec.discard_output:
            output_path = self._output_path(record.request_id)
            encode_ms = await self._loop.run_in_executor(
                self._executor,
                functools.partial(
                    _write_mp4,
                    frame_chunks=frame_chunks,
                    output_path=output_path,
                    fps=record.spec.fps,
                ),
            )
            record.output_path = str(output_path)
            record.metrics["mp4_encode_ms"] = encode_ms
            record.add_event("mp4_completed", encode_ms=encode_ms)

    @staticmethod
    def _validate_decoded_frame_count(
        *,
        request_id: str,
        frame_chunks: list[Any],
        expected_frames: int,
    ) -> None:
        actual_frames = sum(int(frames.shape[0]) for frames in frame_chunks)
        if actual_frames != expected_frames:
            raise RuntimeError(
                f"request {request_id} produced {actual_frames} frames; "
                f"expected {expected_frames}"
            )

    def _output_path(self, request_id: str) -> Path:
        return Path(self.config.output_dir).resolve() / f"{request_id}.mp4"


def create_app(
    *,
    config: ServerConfig,
    model_factory: ModelFactory | None = None,
) -> FastAPI:
    runtime = SfWanRuntime(config=config, model_factory=model_factory)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            await runtime.start()
            yield
        finally:
            await runtime.close()

    app = FastAPI(
        title=f"SFWan2.1 minimal {config.role} server",
        lifespan=_lifespan,
    )
    app.state.sfwan_runtime = runtime

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            role=config.role,
            model_loaded=runtime.model_loaded,
            model_path=config.model_path,
            model_load_ms=runtime.model_load_ms,
        )

    @app.get("/v1/engine", response_model=EngineStatus)
    async def engine_status() -> EngineStatus:
        return runtime.engine_status()

    @app.post(
        "/v1/generations",
        response_model=SubmissionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_generation(
        generation: GenerationRequest,
        http_request: Request,
    ) -> SubmissionResponse:
        try:
            record = await runtime.submit_generation(generation)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        base_url = (config.public_url or str(http_request.base_url)).rstrip("/")
        status_url = f"{base_url}/v1/jobs/{record.request_id}"
        result_url = f"{base_url}/v1/jobs/{record.request_id}/result"
        vae_status_url = None
        vae_result_url = None
        if config.role == "dit" and config.vae_url is not None:
            vae_base = config.vae_url.rstrip("/")
            vae_status_url = f"{vae_base}/v1/jobs/{record.request_id}"
            vae_result_url = f"{vae_status_url}/result"
        return SubmissionResponse(
            request_id=record.request_id,
            status=record.state,
            status_url=status_url,
            result_url=result_url,
            vae_status_url=vae_status_url,
            vae_result_url=vae_result_url,
            warnings=generation.warnings(),
        )

    @app.post(
        "/v1/dit-profiles",
        response_model=SubmissionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_dit_profile(
        profile: DitProfileRequest,
        http_request: Request,
    ) -> SubmissionResponse:
        try:
            record = await runtime.submit_dit_profile(profile)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        base_url = (config.public_url or str(http_request.base_url)).rstrip("/")
        return SubmissionResponse(
            request_id=record.request_id,
            status=record.state,
            status_url=f"{base_url}/v1/jobs/{record.request_id}",
            result_url=f"{base_url}/v1/jobs/{record.request_id}/result",
            warnings=profile.warnings(),
        )

    @app.post(
        "/v1/latent-jobs",
        response_model=LatentJobRegistrationResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def register_latent_job(
        spec: LatentJobSpec,
    ) -> LatentJobRegistrationResponse:
        try:
            record, created, descriptor = await runtime.register_latent_job(spec)
        except ValueError as exc:
            code = 409 if "already exists" in str(exc) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc
        return LatentJobRegistrationResponse(
            request_id=record.request_id,
            created=created,
            status=record.state,
            transport=spec.transport,
            shm=descriptor,
        )

    @app.put(
        "/v1/latent-jobs/{request_id}/chunks/{chunk_index}",
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def put_latent_chunk(
        request_id: str,
        chunk_index: int,
        http_request: Request,
    ) -> dict[str, Any]:
        content_type = http_request.headers.get("content-type", "")
        if not content_type.startswith("application/x-safetensors"):
            raise HTTPException(
                status_code=415,
                detail="content-type must be application/x-safetensors",
            )
        payload = await http_request.body()
        try:
            accepted = await runtime.put_latent_chunk(
                request_id=request_id,
                chunk_index=chunk_index,
                payload=payload,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="latent job not found") from exc
        except ValueError as exc:
            code = 409 if "different content" in str(exc) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc
        return {
            "request_id": request_id,
            "chunk_index": chunk_index,
            "accepted": accepted,
        }

    @app.post(
        "/v1/latent-jobs/{request_id}/chunks/{chunk_index}/ready",
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def ready_shared_memory_chunk(
        request_id: str,
        chunk_index: int,
        ready: SharedMemoryChunkReady,
    ) -> dict[str, Any]:
        try:
            accepted = await runtime.ready_shared_memory_chunk(
                request_id=request_id,
                chunk_index=chunk_index,
                ready=ready,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="latent job not found") from exc
        except ValueError as exc:
            code = 409 if "digest" in str(exc) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc
        return {
            "request_id": request_id,
            "chunk_index": chunk_index,
            "accepted": accepted,
        }

    @app.delete(
        "/v1/latent-jobs/{request_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def cancel_latent_job(request_id: str) -> Response:
        try:
            await runtime.cancel_latent_job(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="latent job not found") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/v1/jobs/{request_id}", response_model=JobStatus)
    async def get_job(request_id: str) -> JobStatus:
        try:
            return runtime.job_status(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/v1/jobs/{request_id}/result")
    async def get_result(request_id: str) -> FileResponse:
        record = runtime.jobs.get(request_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        if record.output_path is None:
            raise HTTPException(
                status_code=409,
                detail="result is not available",
            )
        path = Path(record.output_path)
        if not path.is_file():
            raise HTTPException(status_code=500, detail="result file is missing")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=path.name,
        )

    return app


def _parse_args() -> ServerConfig:
    from sglang.multimodal_gen.utils import StoreBoolean

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        required=True,
        choices=("monolithic", "dit", "vae"),
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--public-url")
    parser.add_argument("--vae-url")
    parser.add_argument("--output-dir", default="sfwan_outputs")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--transfer-depth",
        "--transfer-queue-depth",
        dest="transfer_queue_depth",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--chunk-timeout",
        "--chunk-timeout-seconds",
        dest="chunk_timeout_seconds",
        type=float,
        default=600.0,
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--latent-transport",
        choices=("http", "shm"),
        default="http",
    )
    parser.add_argument(
        "--vae-precision",
        choices=("fp32", "fp16", "fp16_trt", "int8_trt"),
        default="fp32",
    )
    parser.add_argument("--vae-engine-dir")
    parser.add_argument(
        "--vae-trt-variant",
        choices=(
            "baseline",
            "fusion_v1",
            "fusion_v2",
            "native_int8_v1",
            "native_int8_v2",
        ),
        default="baseline",
        help="isolated TensorRT VAE experiment variant; baseline is unchanged",
    )
    parser.add_argument(
        "--native-int8-v2-level",
        choices=("p1", "p2", "p3"),
        default="p1",
        help="Native INT8 V2 optimization level; ignored by other variants",
    )
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action=StoreBoolean,
        default=True,
    )
    parser.add_argument(
        "--dit-cpu-offload",
        action=StoreBoolean,
        default=False,
    )
    parser.add_argument(
        "--vae-cpu-offload",
        action=StoreBoolean,
        default=False,
    )
    parser.add_argument("--enable-profile", action="store_true")
    parser.add_argument(
        "--enable-trt-layer-profile",
        action="store_true",
        help=(
            "enable diagnostic TensorRT IProfiler callbacks for a dedicated "
            "VAE profile server"
        ),
    )
    parser.add_argument(
        "--enable-native-int8-kernel-profile",
        action="store_true",
        help=(
            "collect diagnostic native residual-block kernel timings for a "
            "dedicated VAE profile server"
        ),
    )
    parser.add_argument("--enable-nvtx", action="store_true")
    args = parser.parse_args()
    return ServerConfig(
        role=args.role,
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        public_url=args.public_url,
        vae_url=args.vae_url,
        output_dir=args.output_dir,
        max_pixels=args.max_pixels,
        transfer_queue_depth=args.transfer_queue_depth,
        chunk_timeout_seconds=args.chunk_timeout_seconds,
        device_index=args.device_index,
        vae_precision=args.vae_precision,
        vae_engine_dir=args.vae_engine_dir,
        vae_trt_variant=args.vae_trt_variant,
        native_int8_v2_level=args.native_int8_v2_level,
        text_encoder_cpu_offload=args.text_encoder_cpu_offload,
        dit_cpu_offload=args.dit_cpu_offload,
        vae_cpu_offload=args.vae_cpu_offload,
        latent_transport=args.latent_transport,
        enable_profile=args.enable_profile,
        enable_trt_layer_profile=args.enable_trt_layer_profile,
        enable_native_int8_kernel_profile=(
            args.enable_native_int8_kernel_profile
        ),
        enable_nvtx=args.enable_nvtx,
    )


def main() -> None:
    import uvicorn

    config = _parse_args()
    uvicorn.run(
        create_app(config=config),
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
