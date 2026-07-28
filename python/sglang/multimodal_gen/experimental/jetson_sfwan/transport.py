"""Pinned HTTP ingress and same-host shared-memory latent transport.

CUDA imports stay inside runtime methods so CPU-only protocol tests can import
this module without initializing a device.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mmap
import os
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import Executor
from multiprocessing import shared_memory
from typing import Any

import msgspec

from .protocol import (
    LatentJobRegistrationResponse,
    LatentJobSpec,
    SAFETENSORS_WIRE_DTYPE,
    SafetensorsLatentPayload,
    SharedMemoryChunkReady,
    SharedMemoryDescriptor,
)

EventCallback = Callable[[str, str, dict[str, Any]], Awaitable[None]]
FailureCallback = Callable[[str, str], Awaitable[None]]
_SHM_HEADER_MAGIC = b"SFWAN-SHM-V1\0"


def _align_up(value: int, alignment: int) -> int:
    if value <= 0 or alignment <= 0:
        raise ValueError("value and alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def build_shared_memory_descriptor(
    *,
    name: str,
    lease_token: str,
    shape: tuple[int, int, int, int, int],
    total_chunks: int,
    page_size: int = mmap.PAGESIZE,
) -> SharedMemoryDescriptor:
    """Build the fixed request-level layout without allocating memory."""

    tensor_nbytes = 2
    for dimension in shape:
        tensor_nbytes *= dimension
    data_offset = page_size
    chunk_stride = _align_up(tensor_nbytes, page_size)
    total_bytes = data_offset + chunk_stride * total_chunks
    return SharedMemoryDescriptor(
        name=name,
        lease_token=lease_token,
        total_bytes=total_bytes,
        data_offset=data_offset,
        chunk_stride=chunk_stride,
        tensor_nbytes=tensor_nbytes,
        shape=shape,
        total_chunks=total_chunks,
    )


def _shared_memory_header(descriptor: SharedMemoryDescriptor) -> bytes:
    canonical_descriptor = json.dumps(
        descriptor.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _SHM_HEADER_MAGIC + hashlib.sha256(canonical_descriptor).digest()


def _check_cuda_result(result: Any, operation: str) -> None:
    result_code = result[0] if isinstance(result, tuple) else result
    if int(result_code) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {int(result_code)}")


def _cuda_memcpy_async(
    *,
    destination: int,
    source: int,
    nbytes: int,
    kind_name: str,
    stream: Any,
) -> None:
    import torch

    cudart = torch.cuda.cudart()
    kind = getattr(cudart.cudaMemcpyKind, kind_name)
    result = cudart.cudaMemcpyAsync(
        destination,
        source,
        nbytes,
        kind,
        stream.cuda_stream,
    )
    _check_cuda_result(result, "cudaMemcpyAsync")


def _copy_payload_data_to_pinned(
    payload: SafetensorsLatentPayload,
    pinned: Any,
) -> None:
    """Copy raw safetensors tensor bytes directly into existing CPU storage."""

    import torch

    if pinned.device.type != "cpu":
        raise ValueError("HTTP latent staging destination must be on CPU")
    if pinned.dtype != torch.bfloat16:
        raise ValueError("HTTP latent staging destination must be bfloat16")
    if tuple(pinned.shape) != payload.shape:
        raise ValueError("HTTP latent payload and pinned destination shapes differ")
    if not pinned.is_contiguous():
        raise ValueError("HTTP latent staging destination must be contiguous")
    if payload.dtype != SAFETENSORS_WIRE_DTYPE:
        raise ValueError("HTTP latent payload must use BF16 safetensors storage")

    destination = memoryview(pinned.view(torch.uint8).reshape(-1).numpy())
    source = payload.data_view()
    try:
        if destination.nbytes != source.nbytes:
            raise ValueError(
                "HTTP latent payload byte length does not match destination"
            )
        destination[:] = source
    finally:
        source.release()
        destination.release()


class StagedDeviceChunk:
    """GPU tensor plus a producer-stream event and bounded-buffer lifetime."""

    def __init__(
        self,
        *,
        tensor: Any,
        ready_event: Any = None,
        start_event: Any = None,
        ingress_metrics: dict[str, Any] | None = None,
        release_callback: Callable[[], None] | None = None,
    ) -> None:
        self.tensor = tensor
        self.ready_event = ready_event
        self.start_event = start_event
        self.ingress_metrics = dict(ingress_metrics or {})
        self._release_callback = release_callback
        self._released = False

    def wait_on_current_stream(self) -> Any:
        if self.ready_event is not None:
            import torch

            compute_stream = torch.cuda.current_stream(self.tensor.device)
            compute_stream.wait_event(self.ready_event)
            # The tensor was populated on a copy stream but is consumed on the
            # model thread's compute stream. Tell the caching allocator about
            # that use so dropping this wrapper after decode() returns cannot
            # recycle the allocation while queued decoder kernels still read it.
            self.tensor.record_stream(compute_stream)
        return self.tensor

    def completed_metrics(self) -> dict[str, Any]:
        metrics = dict(self.ingress_metrics)
        if self.start_event is not None and self.ready_event is not None:
            self.ready_event.synchronize()
            metrics["h2d_ms"] = float(self.start_event.elapsed_time(self.ready_event))
        return metrics

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.ready_event is not None:
            self.ready_event.synchronize()
        if self._release_callback is not None:
            self._release_callback()


class SharedMemoryChunkRef(msgspec.Struct, frozen=True, kw_only=True):
    """A ready shared slot; H2D is deferred until its FCFS job is running."""

    region: Any
    chunk_index: int


class PinnedH2DLoader:
    """Bounded safetensors-body -> pinned -> GPU prefetch path for HTTP chunks."""

    def __init__(
        self,
        *,
        device: Any,
        depth: int,
        executor: Executor,
        enable_profile: bool = False,
    ) -> None:
        if depth <= 0:
            raise ValueError("H2D depth must be positive")
        self.device = device
        self._executor = executor
        self._enable_profile = enable_profile
        self._slots = asyncio.Semaphore(depth)
        self._pool: dict[tuple[tuple[int, ...], Any], list[Any]] = {}
        self._pool_lock = threading.Lock()
        self._copy_stream: Any = None
        self._closed = False

    async def stage(
        self,
        payload: SafetensorsLatentPayload,
    ) -> StagedDeviceChunk:
        if self._closed:
            raise RuntimeError("H2D loader is closed")
        await self._slots.acquire()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor,
                self._stage_sync,
                payload,
                loop,
            )
        except BaseException:
            self._slots.release()
            raise

    def _stage_sync(
        self,
        payload: SafetensorsLatentPayload,
        loop: asyncio.AbstractEventLoop,
    ) -> StagedDeviceChunk:
        import torch

        if not isinstance(payload, SafetensorsLatentPayload):
            raise TypeError("HTTP H2D staging requires a safetensors payload view")
        key = (payload.shape, torch.bfloat16)
        with self._pool_lock:
            available = self._pool.get(key)
            pinned = available.pop() if available else None
        if pinned is None:
            pinned = torch.empty(
                payload.shape,
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )

        copy_start = time.perf_counter() if self._enable_profile else None
        _copy_payload_data_to_pinned(payload, pinned)
        pageable_to_pinned_ms = (
            (time.perf_counter() - copy_start) * 1000
            if copy_start is not None
            else None
        )
        device_tensor = torch.empty_like(pinned, device=self.device)
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.device)
        start_event = (
            torch.cuda.Event(enable_timing=True) if self._enable_profile else None
        )
        ready_event = torch.cuda.Event(enable_timing=self._enable_profile)
        with torch.cuda.stream(self._copy_stream):
            if start_event is not None:
                start_event.record(self._copy_stream)
            device_tensor.copy_(pinned, non_blocking=True)
            ready_event.record(self._copy_stream)

        def _release() -> None:
            with self._pool_lock:
                self._pool.setdefault(key, []).append(pinned)
            loop.call_soon_threadsafe(self._slots.release)

        ingress_metrics = {}
        if pageable_to_pinned_ms is not None:
            ingress_metrics["pageable_to_pinned_ms"] = pageable_to_pinned_ms
        return StagedDeviceChunk(
            tensor=device_tensor,
            start_event=start_event,
            ready_event=ready_event,
            ingress_metrics=ingress_metrics,
            release_callback=_release,
        )

    def close(self) -> None:
        self._closed = True
        with self._pool_lock:
            self._pool.clear()
        self._copy_stream = None


class SharedMemoryRegion:
    """One process's mapping and CUDA registration of a request-level region."""

    def __init__(
        self,
        *,
        descriptor: SharedMemoryDescriptor,
        mapping: shared_memory.SharedMemory,
        owner: bool,
    ) -> None:
        import torch

        self.descriptor = descriptor
        self.mapping = mapping
        self.owner = owner
        self._closed = False
        self._lock = threading.RLock()
        header = _shared_memory_header(descriptor)
        if len(header) > descriptor.data_offset:
            raise ValueError("shared-memory descriptor leaves no room for its header")
        header_page = mapping.buf[: descriptor.data_offset]
        try:
            if owner:
                header_page[:] = b"\0" * descriptor.data_offset
                header_page[: len(header)] = header
            elif bytes(header_page[: len(header)]) != header:
                raise ValueError(
                    "shared-memory header does not match its descriptor or lease"
                )
        finally:
            del header_page
        self._bytes = torch.frombuffer(mapping.buf, dtype=torch.uint8)
        cudart = torch.cuda.cudart()
        try:
            result = cudart.cudaHostRegister(
                self._bytes.data_ptr(),
                descriptor.total_bytes,
                0,
            )
            _check_cuda_result(result, "cudaHostRegister")
        except BaseException:
            self._bytes = None
            raise

    @classmethod
    def create(
        cls,
        *,
        spec: LatentJobSpec,
    ) -> "SharedMemoryRegion":
        if os.name != "posix":
            raise RuntimeError("shared-memory latent transport requires Linux")
        token = uuid.uuid4().hex
        name = f"sfwan_{spec.request_id[:40]}_{token[:12]}"
        descriptor = build_shared_memory_descriptor(
            name=name,
            lease_token=token,
            shape=spec.latent_chunk_shape,
            total_chunks=int(spec.total_chunks),
        )
        mapping = shared_memory.SharedMemory(
            name=descriptor.name,
            create=True,
            size=descriptor.total_bytes,
        )
        try:
            return cls(descriptor=descriptor, mapping=mapping, owner=True)
        except BaseException:
            mapping.close()
            mapping.unlink()
            raise

    @classmethod
    def attach(
        cls,
        descriptor: SharedMemoryDescriptor,
    ) -> "SharedMemoryRegion":
        if os.name != "posix":
            raise RuntimeError("shared-memory latent transport requires Linux")
        mapping = shared_memory.SharedMemory(
            name=descriptor.name,
            create=False,
        )
        if mapping.size < descriptor.total_bytes:
            mapping.close()
            raise ValueError("shared-memory mapping is smaller than its descriptor")
        try:
            return cls(descriptor=descriptor, mapping=mapping, owner=False)
        except BaseException:
            mapping.close()
            raise

    def _offset(self, chunk_index: int) -> int:
        if chunk_index < 0 or chunk_index >= self.descriptor.total_chunks:
            raise ValueError("shared-memory chunk index is outside the request")
        return self.descriptor.data_offset + chunk_index * self.descriptor.chunk_stride

    def host_tensor(self, chunk_index: int) -> Any:
        import torch

        offset = self._offset(chunk_index)
        raw = self._bytes.narrow(0, offset, self.descriptor.tensor_nbytes)
        return raw.view(torch.bfloat16).view(self.descriptor.shape)

    def digest(self, chunk_index: int) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("shared-memory mapping is closed")
            offset = self._offset(chunk_index)
            view = self.mapping.buf[offset : offset + self.descriptor.tensor_nbytes]
            digest = hashlib.sha256(view).hexdigest()
            del view
            return digest

    def copy_from_cuda(
        self,
        *,
        chunk_index: int,
        source: Any,
        stream: Any,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("shared-memory mapping is closed")
            host = self.host_tensor(chunk_index)
            _cuda_memcpy_async(
                destination=host.data_ptr(),
                source=source.data_ptr(),
                nbytes=self.descriptor.tensor_nbytes,
                kind_name="cudaMemcpyDeviceToHost",
                stream=stream,
            )

    def copy_to_cuda(
        self,
        *,
        chunk_index: int,
        destination: Any,
        stream: Any,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("shared-memory mapping is closed")
            host = self.host_tensor(chunk_index)
            _cuda_memcpy_async(
                destination=destination.data_ptr(),
                source=host.data_ptr(),
                nbytes=self.descriptor.tensor_nbytes,
                kind_name="cudaMemcpyHostToDevice",
                stream=stream,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            import torch

            cudart = torch.cuda.cudart()
            unregister_error = None
            try:
                result = cudart.cudaHostUnregister(self._bytes.data_ptr())
                _check_cuda_result(result, "cudaHostUnregister")
            except Exception as exc:
                unregister_error = exc
            finally:
                self._bytes = None
                self.mapping.close()
                if self.owner:
                    try:
                        self.mapping.unlink()
                    except FileNotFoundError:
                        pass
            if unregister_error is not None:
                raise unregister_error


def stage_shared_chunk_to_device(
    *,
    region: SharedMemoryRegion,
    chunk_index: int,
    device: Any,
    copy_stream: Any = None,
    enable_profile: bool = False,
) -> tuple[StagedDeviceChunk, Any]:
    """Launch registered-host H2D and return the reusable stream."""

    import torch

    stream = copy_stream or torch.cuda.Stream(device=device)
    destination = torch.empty(
        region.descriptor.shape,
        dtype=torch.bfloat16,
        device=device,
    )
    start_event = torch.cuda.Event(enable_timing=True) if enable_profile else None
    ready_event = torch.cuda.Event(enable_timing=enable_profile)
    with torch.cuda.stream(stream):
        if start_event is not None:
            start_event.record(stream)
        region.copy_to_cuda(
            chunk_index=chunk_index,
            destination=destination,
            stream=stream,
        )
        ready_event.record(stream)
    return (
        StagedDeviceChunk(
            tensor=destination,
            start_event=start_event,
            ready_event=ready_event,
            ingress_metrics={"shared_memory": True},
        ),
        stream,
    )


class _SharedPendingChunk(msgspec.Struct, frozen=True, kw_only=True):
    request_id: str
    chunk_index: int
    source_tensor: Any
    copy_start_event: Any
    ready_event: Any


class _SharedTransferState:
    def __init__(
        self,
        *,
        spec: LatentJobSpec,
        region: SharedMemoryRegion,
    ) -> None:
        self.spec = spec
        self.region = region
        self.accepted_chunks = 0
        self.error: str | None = None
        self.done = asyncio.Event()
        self.pending_copy_events: dict[int, Any] = {}
        # Custom cudaMemcpyAsync does not teach PyTorch's allocator about the
        # copy stream. Keep every GPU source alive explicitly until its local
        # D2H event has completed, including cancellation/close paths.
        self.pending_sources: dict[int, Any] = {}


class SharedMemoryChunkSender:
    """DiT-side D2H and control-only sender for same-host deployments."""

    _RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)

    def __init__(
        self,
        *,
        vae_url: str,
        queue_depth: int,
        enable_profile: bool = False,
        event_callback: EventCallback,
        failure_callback: FailureCallback,
        client: Any = None,
    ) -> None:
        if queue_depth <= 0:
            raise ValueError("queue depth must be positive")
        self.vae_url = vae_url.rstrip("/")
        self._enable_profile = enable_profile
        self._queue: asyncio.Queue[_SharedPendingChunk] = asyncio.Queue(
            maxsize=queue_depth
        )
        self._event_callback = event_callback
        self._failure_callback = failure_callback
        self._client = client
        self._owns_client = client is None
        self._sender_task: asyncio.Task[None] | None = None
        self._states: dict[str, _SharedTransferState] = {}
        self._copy_streams: dict[int, Any] = {}

    async def start(self) -> None:
        if os.name != "posix":
            raise RuntimeError("shared-memory latent transport requires Linux")
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        self._sender_task = asyncio.create_task(
            self._sender_loop(),
            name="sfwan-shm-control-sender",
        )

    async def close(self) -> None:
        if self._sender_task is not None:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None
        for request_id in list(self._states):
            self.release_job(request_id)
        self._states.clear()
        self._copy_streams.clear()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None

    async def register_job(self, spec: LatentJobSpec) -> None:
        if spec.transport != "shm":
            raise ValueError("SHM chunk sender requires transport='shm'")
        response = await self._request_with_retries(
            method="post",
            url=f"{self.vae_url}/v1/latent-jobs",
            request_id=spec.request_id,
            json=spec.model_dump(mode="json"),
        )
        registration = LatentJobRegistrationResponse.model_validate(response.json())
        if registration.shm is None:
            raise RuntimeError("VAE server did not return a shared-memory descriptor")
        descriptor = registration.shm
        if tuple(descriptor.shape) != tuple(spec.latent_chunk_shape):
            raise RuntimeError("VAE shared-memory shape does not match the latent job")
        if descriptor.total_chunks != int(spec.total_chunks):
            raise RuntimeError(
                "VAE shared-memory chunk count does not match the latent job"
            )
        region = await asyncio.to_thread(
            SharedMemoryRegion.attach,
            descriptor,
        )
        self._states[spec.request_id] = _SharedTransferState(
            spec=spec,
            region=region,
        )
        await self._event_callback(
            spec.request_id,
            "vae_job_registered",
            {
                "vae_url": self.vae_url,
                "transport": "shm",
                "shared_bytes": descriptor.total_bytes,
            },
        )

    def stage_tensor(
        self,
        *,
        request_id: str,
        chunk_index: int,
        tensor: Any,
    ) -> _SharedPendingChunk:
        import torch

        state = self._states.get(request_id)
        if state is None:
            raise RuntimeError("shared-memory job is not registered")
        source = tensor.detach().to(dtype=torch.bfloat16).contiguous()
        if source.device.type != "cuda":
            raise ValueError("SHM D2H source must be a CUDA tensor")
        if tuple(source.shape) != state.region.descriptor.shape:
            raise ValueError("SHM D2H tensor shape does not match descriptor")
        device_index = source.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream = self._copy_streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=device_index)
            self._copy_streams[device_index] = stream
        stream.wait_stream(torch.cuda.current_stream(device_index))
        start_event = (
            torch.cuda.Event(enable_timing=True) if self._enable_profile else None
        )
        ready_event = torch.cuda.Event(enable_timing=self._enable_profile)
        with torch.cuda.stream(stream):
            if start_event is not None:
                start_event.record(stream)
            state.region.copy_from_cuda(
                chunk_index=chunk_index,
                source=source,
                stream=stream,
            )
            ready_event.record(stream)
        state.pending_copy_events[chunk_index] = ready_event
        state.pending_sources[chunk_index] = source
        return _SharedPendingChunk(
            request_id=request_id,
            chunk_index=chunk_index,
            source_tensor=source,
            copy_start_event=start_event,
            ready_event=ready_event,
        )

    async def enqueue(self, chunk: _SharedPendingChunk) -> None:
        await self._queue.put(chunk)
        await self._event_callback(
            chunk.request_id,
            "transfer_enqueued",
            {
                "chunk_index": chunk.chunk_index,
                "transport": "shm",
                "sender_queue_depth": self._queue.qsize(),
            },
        )

    async def wait_for_job(
        self,
        request_id: str,
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        state = self._states[request_id]
        while not state.done.is_set():
            if is_cancelled is not None and is_cancelled():
                raise RuntimeError("shared-memory transfer was cancelled")
            try:
                await asyncio.wait_for(state.done.wait(), timeout=0.1)
            except TimeoutError:
                continue
        if state.error is not None:
            raise RuntimeError(state.error)

    async def cancel_remote_job(self, request_id: str) -> None:
        if self._client is None:
            return
        try:
            await self._client.delete(f"{self.vae_url}/v1/latent-jobs/{request_id}")
        except Exception:
            return

    def release_job(self, request_id: str) -> None:
        state = self._states.pop(request_id, None)
        if state is not None:
            for chunk_index, event in state.pending_copy_events.items():
                event.synchronize()
                state.pending_sources.pop(chunk_index, None)
            state.pending_copy_events.clear()
            state.pending_sources.clear()
            state.region.close()

    async def _sender_loop(self) -> None:
        while True:
            chunk = await self._queue.get()
            try:
                await self._send_chunk(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                state = self._states.get(chunk.request_id)
                if state is not None:
                    state.error = message
                    state.done.set()
                await self._failure_callback(chunk.request_id, message)
                await self.cancel_remote_job(chunk.request_id)
            finally:
                self._queue.task_done()

    async def _send_chunk(self, chunk: _SharedPendingChunk) -> None:
        state = self._states.get(chunk.request_id)
        if state is None or state.error is not None:
            return
        await asyncio.to_thread(chunk.ready_event.synchronize)
        state.pending_copy_events.pop(chunk.chunk_index, None)
        state.pending_sources.pop(chunk.chunk_index, None)
        d2h_ms = (
            float(chunk.copy_start_event.elapsed_time(chunk.ready_event))
            if self._enable_profile and chunk.copy_start_event is not None
            else None
        )
        digest_start = time.perf_counter() if self._enable_profile else None
        digest = await asyncio.to_thread(
            state.region.digest,
            chunk.chunk_index,
        )
        digest_ms = (
            (time.perf_counter() - digest_start) * 1000
            if digest_start is not None
            else None
        )
        ready = SharedMemoryChunkReady(
            lease_token=state.region.descriptor.lease_token,
            digest=digest,
        )
        send_start = time.perf_counter() if self._enable_profile else None
        await self._request_with_retries(
            method="post",
            url=(
                f"{self.vae_url}/v1/latent-jobs/{chunk.request_id}/"
                f"chunks/{chunk.chunk_index}/ready"
            ),
            request_id=chunk.request_id,
            json=ready.model_dump(mode="json"),
        )
        control_rtt_ms = (
            (time.perf_counter() - send_start) * 1000
            if send_start is not None
            else None
        )
        details: dict[str, Any] = {
            "chunk_index": chunk.chunk_index,
            "transport": "shm",
        }
        if self._enable_profile:
            details.update(
                {
                    "d2h_ms": d2h_ms,
                    "digest_ms": digest_ms,
                    "control_rtt_ms": control_rtt_ms,
                }
            )
        await self._event_callback(
            chunk.request_id,
            "transfer_accepted",
            details,
        )
        state.accepted_chunks += 1
        if state.accepted_chunks == int(state.spec.total_chunks):
            state.done.set()

    async def _request_with_retries(
        self,
        *,
        method: str,
        url: str,
        request_id: str,
        **kwargs: Any,
    ) -> Any:
        if self._client is None:
            raise RuntimeError("SHM sender has not been started")
        last_error: Exception | None = None
        attempts = len(self._RETRY_DELAYS_SECONDS) + 1
        for attempt in range(attempts):
            try:
                response = await getattr(self._client, method)(url, **kwargs)
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                if (
                    status_code is not None
                    and 400 <= status_code < 500
                    and status_code not in {408, 429}
                ):
                    raise
                if attempt == attempts - 1:
                    break
                delay = self._RETRY_DELAYS_SECONDS[attempt]
                await self._event_callback(
                    request_id,
                    "transfer_retry",
                    {
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "transport": "shm",
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"request to {url} failed after {attempts} attempts: {last_error}"
        )
