"""Small FCFS engines and the asynchronous DiT-to-VAE chunk sender."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import msgspec

from .protocol import (
    DitProfileRequest,
    EngineStatus,
    GenerationRequest,
    JobEvent,
    JobState,
    JobStatus,
    LatentJobSpec,
    TERMINAL_JOB_STATES,
    payload_digest,
    serialize_latent_tensor,
)

Role = Literal["monolithic", "dit", "vae"]
JobPayload = GenerationRequest | DitProfileRequest | LatentJobSpec
JobHandler = Callable[["JobRecord"], Awaitable[None]]
EventCallback = Callable[[str, str, dict[str, Any]], Awaitable[None]]
FailureCallback = Callable[[str, str], Awaitable[None]]


class ReceivedChunk(msgspec.Struct, frozen=True, kw_only=True):
    data: Any
    digest: str
    received_unix_time_ns: int
    ingress_metrics: dict[str, Any] = msgspec.field(default_factory=dict)


class PendingChunk(msgspec.Struct, frozen=True, kw_only=True):
    request_id: str
    chunk_index: int
    tensor: Any
    source_tensor: Any = None
    copy_start_event: Any = None
    ready_event: Any = None
    release_callback: Any = None


class JobRecord:
    """Mutable event-loop-owned state for one request."""

    def __init__(
        self,
        *,
        request_id: str,
        role: Role,
        payload: JobPayload,
    ) -> None:
        self.request_id = request_id
        self.role = role
        self.payload = payload
        self.queue_sequence = -1
        self.state = JobState.WAITING
        self.created_unix_time_ns = time.time_ns()
        self.started_unix_time_ns: int | None = None
        self.completed_unix_time_ns: int | None = None
        self.error: str | None = None
        self.output_path: str | None = None
        self.events: list[JobEvent] = []
        self.metrics: dict[str, Any] = {}
        self._next_event_sequence = 0
        self.add_event("accepted")

    def add_event(self, kind: str, **details: Any) -> None:
        self.events.append(
            JobEvent(
                sequence=self._next_event_sequence,
                kind=kind,
                details=details,
            )
        )
        self._next_event_sequence += 1

    def mark_running(self) -> None:
        self.state = JobState.RUNNING
        self.started_unix_time_ns = time.time_ns()
        queue_wait_ms = (
            self.started_unix_time_ns - self.created_unix_time_ns
        ) / 1_000_000
        self.metrics["queue_wait_ms"] = queue_wait_ms
        self.add_event(
            "started",
            queue_wait_ms=queue_wait_ms,
        )

    def mark_completed(self) -> None:
        self.state = JobState.COMPLETED
        self.completed_unix_time_ns = time.time_ns()
        total_ms = (self.completed_unix_time_ns - self.created_unix_time_ns) / 1_000_000
        self.metrics["total_ms"] = total_ms
        self.add_event(
            "completed",
            total_ms=total_ms,
        )

    def mark_failed(self, error: str) -> None:
        if self.state in TERMINAL_JOB_STATES:
            return
        self.state = JobState.FAILED
        self.error = error
        self.completed_unix_time_ns = time.time_ns()
        total_ms = (self.completed_unix_time_ns - self.created_unix_time_ns) / 1_000_000
        self.metrics["total_ms"] = total_ms
        self.add_event(
            "failed",
            error=error,
            total_ms=total_ms,
        )

    def mark_cancelled(self, reason: str) -> None:
        if self.state in TERMINAL_JOB_STATES:
            return
        self.state = JobState.CANCELLED
        self.error = reason
        self.completed_unix_time_ns = time.time_ns()
        total_ms = (self.completed_unix_time_ns - self.created_unix_time_ns) / 1_000_000
        self.metrics["total_ms"] = total_ms
        self.add_event(
            "cancelled",
            reason=reason,
            total_ms=total_ms,
        )

    def to_status(self) -> JobStatus:
        return JobStatus(
            request_id=self.request_id,
            role=self.role,
            state=self.state,
            queue_sequence=self.queue_sequence,
            created_unix_time_ns=self.created_unix_time_ns,
            started_unix_time_ns=self.started_unix_time_ns,
            completed_unix_time_ns=self.completed_unix_time_ns,
            error=self.error,
            output_available=self.output_path is not None,
            events=list(self.events),
            metrics=dict(self.metrics),
        )


class VaeJobRecord(JobRecord):
    """A request-level VAE job that may receive chunks out of order."""

    def __init__(self, *, spec: LatentJobSpec) -> None:
        super().__init__(request_id=spec.request_id, role="vae", payload=spec)
        self.spec = spec
        self._chunks: dict[int, ReceivedChunk] = {}
        self._chunk_digests: dict[int, str] = {}
        self._chunk_condition = asyncio.Condition()

    async def put_chunk(
        self,
        *,
        chunk_index: int,
        data: Any,
        digest: str,
        ingress_metrics: dict[str, Any] | None = None,
    ) -> bool:
        if chunk_index < 0 or chunk_index >= int(self.spec.total_chunks):
            raise ValueError(
                f"chunk_index={chunk_index} is outside "
                f"[0, {int(self.spec.total_chunks)})"
            )

        async with self._chunk_condition:
            existing_digest = self._chunk_digests.get(chunk_index)
            if existing_digest is not None:
                if existing_digest != digest:
                    raise ValueError(
                        f"chunk {chunk_index} was already uploaded with "
                        "different content"
                    )
                return False
            if self.state in TERMINAL_JOB_STATES:
                raise ValueError(
                    f"cannot upload a new chunk to a {self.state.value} latent job"
                )
            self._chunk_digests[chunk_index] = digest
            self._chunks[chunk_index] = ReceivedChunk(
                data=data,
                digest=digest,
                received_unix_time_ns=time.time_ns(),
                ingress_metrics=dict(ingress_metrics or {}),
            )
            self._chunk_condition.notify_all()
        self.add_event("chunk_received", chunk_index=chunk_index)
        return True

    async def wait_for_chunk(
        self,
        *,
        chunk_index: int,
        timeout_seconds: float,
    ) -> ReceivedChunk:
        async def _wait() -> ReceivedChunk:
            async with self._chunk_condition:
                await self._chunk_condition.wait_for(
                    lambda: (
                        chunk_index in self._chunks
                        or self.state in {JobState.CANCELLED, JobState.FAILED}
                    )
                )
                if self.state in {JobState.CANCELLED, JobState.FAILED}:
                    raise RuntimeError(self.error or "VAE job was cancelled")
                return self._chunks.pop(chunk_index)

        try:
            return await asyncio.wait_for(_wait(), timeout=timeout_seconds)
        except TimeoutError as exc:
            raise TimeoutError(
                f"timed out waiting {timeout_seconds}s for chunk {chunk_index}"
            ) from exc

    async def existing_chunk_digest(self, chunk_index: int) -> str | None:
        """Return accepted content identity even after the chunk was decoded."""

        async with self._chunk_condition:
            return self._chunk_digests.get(chunk_index)

    async def clear_chunks_and_wake_waiters(self) -> None:
        async with self._chunk_condition:
            for chunk in self._chunks.values():
                release = getattr(chunk.tensor, "release", None)
                if callable(release):
                    release()
            self._chunks.clear()
            self._chunk_condition.notify_all()


class SingleWorkerEngine:
    """One request-level FIFO queue with exactly one running slot."""

    def __init__(self, *, role: Role, handler: JobHandler) -> None:
        self.role = role
        self._handler = handler
        self._waiting: deque[JobRecord] = deque()
        self._running: JobRecord | None = None
        self._condition = asyncio.Condition()
        self._worker_task: asyncio.Task[None] | None = None
        self._next_queue_sequence = 0
        self._closing = False

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name=f"sfwan-{self.role}-fcfs-worker",
        )

    async def submit(self, record: JobRecord) -> None:
        if self._closing:
            raise RuntimeError("engine is closing")
        async with self._condition:
            record.queue_sequence = self._next_queue_sequence
            self._next_queue_sequence += 1
            record.add_event("queued", queue_sequence=record.queue_sequence)
            self._waiting.append(record)
            self._condition.notify()

    def snapshot(self) -> EngineStatus:
        running_ids = [] if self._running is None else [self._running.request_id]
        return EngineStatus(
            role=self.role,
            waiting_count=len(self._waiting),
            waiting_ids=[record.request_id for record in self._waiting],
            running_ids=running_ids,
        )

    async def close(self) -> None:
        self._closing = True
        waiting_records: list[JobRecord]
        async with self._condition:
            waiting_records = list(self._waiting)
            for record in waiting_records:
                record.mark_cancelled("server shutdown")
            self._waiting.clear()
            self._condition.notify_all()
        for record in waiting_records:
            if isinstance(record, VaeJobRecord):
                await record.clear_chunks_and_wake_waiters()
        if self._running is not None:
            self._running.mark_cancelled("server shutdown")
            if isinstance(self._running, VaeJobRecord):
                await self._running.clear_chunks_and_wake_waiters()
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None

    async def _next_record(self) -> JobRecord | None:
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._waiting) or self._closing)
            if self._closing:
                return None
            return self._waiting.popleft()

    async def _worker_loop(self) -> None:
        while True:
            record = await self._next_record()
            if record is None:
                return
            if record.state in TERMINAL_JOB_STATES:
                continue
            self._running = record
            record.mark_running()
            try:
                await self._handler(record)
                if record.state == JobState.RUNNING:
                    record.mark_completed()
            except asyncio.CancelledError:
                record.mark_cancelled("engine worker cancelled")
                raise
            except Exception as exc:
                record.mark_failed(f"{type(exc).__name__}: {exc}")
            finally:
                self._running = None


class _TransferState:
    def __init__(self, *, expected_chunks: int) -> None:
        self.expected_chunks = expected_chunks
        self.accepted_chunks = 0
        self.error: str | None = None
        self.done = asyncio.Event()


class AsyncChunkSender:
    """Bounded D2H + safetensors + HTTP sender used by the DiT role."""

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
            raise ValueError("queue_depth must be positive")
        self.vae_url = vae_url.rstrip("/")
        self._enable_profile = enable_profile
        self._queue: asyncio.Queue[PendingChunk] = asyncio.Queue(maxsize=queue_depth)
        self._event_callback = event_callback
        self._failure_callback = failure_callback
        self._client = client
        self._owns_client = client is None
        self._sender_task: asyncio.Task[None] | None = None
        self._states: dict[str, _TransferState] = {}
        self._copy_streams: dict[int, Any] = {}
        self._slot_limit = queue_depth
        self._slot_count = 0
        self._slot_pool: dict[tuple[tuple[int, ...], Any], list[Any]] = {}
        self._slot_condition = threading.Condition()

    async def start(self) -> None:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        self._sender_task = asyncio.create_task(
            self._sender_loop(),
            name="sfwan-latent-chunk-sender",
        )

    async def close(self) -> None:
        if self._sender_task is not None:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None
        while not self._queue.empty():
            chunk = self._queue.get_nowait()
            self._release_pending_chunk(chunk)
            self._queue.task_done()
        self._states.clear()
        self._copy_streams.clear()
        with self._slot_condition:
            self._slot_pool.clear()
            self._slot_count = 0
            self._slot_condition.notify_all()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None

    async def register_job(self, spec: LatentJobSpec) -> None:
        if spec.transport != "http":
            raise ValueError("HTTP chunk sender requires transport='http'")
        self._states[spec.request_id] = _TransferState(
            expected_chunks=int(spec.total_chunks)
        )
        await self._request_with_retries(
            method="post",
            url=f"{self.vae_url}/v1/latent-jobs",
            request_id=spec.request_id,
            json=spec.model_dump(mode="json"),
        )
        await self._event_callback(
            spec.request_id,
            "vae_job_registered",
            {"vae_url": self.vae_url},
        )

    def stage_tensor(
        self,
        *,
        request_id: str,
        chunk_index: int,
        tensor: Any,
    ) -> PendingChunk:
        """Launch a non-blocking CUDA-to-pinned-host copy when applicable."""

        import torch

        source = tensor.detach()
        if source.device.type != "cuda":
            return PendingChunk(
                request_id=request_id,
                chunk_index=chunk_index,
                tensor=source.to(device="cpu", dtype=torch.bfloat16).contiguous(),
            )

        device_index = source.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        copy_stream = self._copy_streams.get(device_index)
        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=device_index)
            self._copy_streams[device_index] = copy_stream

        slot_key = (tuple(source.shape), torch.bfloat16)
        with self._slot_condition:
            while True:
                available = self._slot_pool.get(slot_key)
                if available:
                    host_tensor = available.pop()
                    break
                if self._slot_count < self._slot_limit:
                    host_tensor = torch.empty(
                        tuple(source.shape),
                        dtype=torch.bfloat16,
                        device="cpu",
                        pin_memory=True,
                    )
                    self._slot_count += 1
                    break
                self._slot_condition.wait()

        current_stream = torch.cuda.current_stream(device_index)
        copy_stream.wait_stream(current_stream)
        start_event = (
            torch.cuda.Event(enable_timing=True) if self._enable_profile else None
        )
        ready_event = torch.cuda.Event(enable_timing=self._enable_profile)
        with torch.cuda.stream(copy_stream):
            if start_event is not None:
                start_event.record(copy_stream)
            host_tensor.copy_(source.to(dtype=torch.bfloat16), non_blocking=True)
            ready_event.record(copy_stream)

        release_lock = threading.Lock()
        released = False

        def _release_slot() -> None:
            nonlocal released
            with release_lock:
                if released:
                    return
                released = True
            # This also makes cancellation safe: a pinned slot is never reused
            # while its producer stream can still be writing to it.
            ready_event.synchronize()
            with self._slot_condition:
                self._slot_pool.setdefault(slot_key, []).append(host_tensor)
                self._slot_condition.notify()

        return PendingChunk(
            request_id=request_id,
            chunk_index=chunk_index,
            tensor=host_tensor,
            source_tensor=source,
            copy_start_event=start_event,
            ready_event=ready_event,
            release_callback=_release_slot,
        )

    async def enqueue(self, chunk: PendingChunk) -> None:
        await self._queue.put(chunk)
        await self._event_callback(
            chunk.request_id,
            "transfer_enqueued",
            {
                "chunk_index": chunk.chunk_index,
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
                raise RuntimeError("latent transfer was cancelled")
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
        self._states.pop(request_id, None)

    async def _sender_loop(self) -> None:
        while True:
            chunk = await self._queue.get()
            try:
                state = self._states.get(chunk.request_id)
                if state is None or state.error is not None:
                    self._release_pending_chunk(chunk)
                    continue
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

    async def _send_chunk(self, chunk: PendingChunk) -> None:
        try:
            d2h_ms = None
            if chunk.ready_event is not None:
                await asyncio.to_thread(chunk.ready_event.synchronize)
                if self._enable_profile and chunk.copy_start_event is not None:
                    d2h_ms = chunk.copy_start_event.elapsed_time(chunk.ready_event)

            encode_start = time.perf_counter() if self._enable_profile else None
            payload = await asyncio.to_thread(serialize_latent_tensor, chunk.tensor)
            encode_ms = (
                (time.perf_counter() - encode_start) * 1000
                if encode_start is not None
                else None
            )
        finally:
            self._release_pending_chunk(chunk)

        send_start = time.perf_counter() if self._enable_profile else None
        await self._request_with_retries(
            method="put",
            url=(
                f"{self.vae_url}/v1/latent-jobs/"
                f"{chunk.request_id}/chunks/{chunk.chunk_index}"
            ),
            request_id=chunk.request_id,
            content=payload,
            headers={"content-type": "application/x-safetensors"},
        )
        http_rtt_ms = (
            (time.perf_counter() - send_start) * 1000
            if send_start is not None
            else None
        )
        details: dict[str, Any] = {
            "chunk_index": chunk.chunk_index,
            "payload_bytes": len(payload),
        }
        if self._enable_profile:
            details.update(
                {
                    "d2h_ms": d2h_ms,
                    "serialize_ms": encode_ms,
                    "http_rtt_ms": http_rtt_ms,
                }
            )
        await self._event_callback(
            chunk.request_id,
            "transfer_accepted",
            details,
        )

        state = self._states.get(chunk.request_id)
        if state is None:
            return
        state.accepted_chunks += 1
        if state.accepted_chunks == state.expected_chunks:
            state.done.set()

    @staticmethod
    def _release_pending_chunk(chunk: PendingChunk) -> None:
        release = chunk.release_callback
        if callable(release):
            release()

    async def _request_with_retries(
        self,
        *,
        method: str,
        url: str,
        request_id: str,
        **kwargs: Any,
    ) -> Any:
        if self._client is None:
            raise RuntimeError("chunk sender has not been started")
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
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"request to {url} failed after {attempts} attempts: {last_error}"
        )


def chunk_digest(payload: bytes) -> str:
    """Named wrapper kept near the VAE chunk-bookkeeping implementation."""

    return payload_digest(payload)
