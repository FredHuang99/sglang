# SPDX-License-Identifier: Apache-2.0
"""Central request router for disaggregated diffusion pipelines."""

import json
import logging
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import zmq

from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    PoolDispatcher,
)
from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    RequestTracker,
    TransferPhase,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
    unpack_tensors,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferAllocMsg,
    TransferAbortMsg,
    TransferMsgType,
    decode_transfer_msg,
    encode_transfer_msg,
    is_transfer_message,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket
from sglang.multimodal_gen.runtime.utils.request_profiling import (
    RequestCsvProfiler,
    resolve_profile_dir,
)

logger = logging.getLogger(__name__)


@dataclass
class _EncoderTTAEntry:
    request_id: str
    client_identity: bytes
    payload: bytes


@dataclass
class _TransferRequestState:
    sender_role: str = ""
    sender_session_id: str = ""
    sender_pool_ptr: int = 0
    sender_slot_offset: int = 0
    sender_meta_pool_ptr: int = 0
    sender_meta_slot_offset: int = 0
    sender_control_endpoint: str = ""
    sender_host_id: str = ""
    data_size: int = 0
    meta_size: int = 0
    receiver_role: str = ""
    receiver_session_id: str = ""
    receiver_pool_ptr: int = 0
    receiver_slot_offset: int = 0
    receiver_slot_size: int = 0
    receiver_meta_pool_ptr: int = 0
    receiver_meta_slot_offset: int = 0
    receiver_meta_slot_size: int = 0
    receiver_control_endpoint: str = ""
    receiver_host_id: str = ""
    receiver_supports_local_copy: bool = False
    sender_instance: int = -1
    receiver_instance: int = -1
    prealloc_slot_id: int | None = None
    sender_slot_released: bool = False
    receiver_slot_released: bool = False
    receiver_prealloc_recycled: bool = False
    decoder_dispatch_enqueued: bool = False
    transfer_completion_processed: bool = False
    alloc_accepted: bool = False
    transfer_phase: TransferPhase = TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT
    handoff_started_at: float | None = None
    phase_started_at: float | None = None
    downstream_wait_since: float | None = None
    downstream_tta_enqueued: bool = False
    rejected_instances: dict[int, int] = field(default_factory=dict)
    send_attempts: int = 0
    max_send_retries: int = 2
    last_send_error: str | None = None
    sender_abort_sent: bool = False
    receiver_abort_sent: bool = False


@dataclass
class _RoleTTAEntry:
    request_id: str
    transfer_state: _TransferRequestState | None = None


class DiffusionServer:
    """Global pipeline orchestrator for N:M:K disaggregated diffusion.

    Capacity-aware dispatch with FreeBufferSlots per instance and TTA queues.
    """

    def __init__(
        self,
        frontend_endpoint: str,
        encoder_work_endpoints: list[str],
        denoiser_work_endpoints: list[str],
        decoder_work_endpoints: list[str],
        encoder_result_endpoint: str,
        denoiser_result_endpoint: str,
        decoder_result_endpoint: str,
        dispatch_policy_name: str = "round_robin",
        timeout_s: float = 600.0,
        downstream_wait_timeout_s: float = 120.0,
        max_slots_per_instance: int = 2,
        p2p_mode: bool = True,
        profile_enabled: bool = False,
        profile_output_dir: str | None = None,
        profile_run_id: str | None = None,
    ):
        self._frontend_endpoint = frontend_endpoint
        self._encoder_work_endpoints = encoder_work_endpoints
        self._denoiser_work_endpoints = denoiser_work_endpoints
        self._decoder_work_endpoints = decoder_work_endpoints
        self._encoder_result_endpoint = encoder_result_endpoint
        self._denoiser_result_endpoint = denoiser_result_endpoint
        self._decoder_result_endpoint = decoder_result_endpoint

        self._num_encoders = len(encoder_work_endpoints)
        self._num_denoisers = len(denoiser_work_endpoints)
        self._num_decoders = len(decoder_work_endpoints)
        self._timeout_s = timeout_s
        self._downstream_wait_timeout_s = downstream_wait_timeout_s
        self._alloc_result_timeout_s = min(self._downstream_wait_timeout_s, 5.0)

        self._tracker = RequestTracker()
        self._dispatcher = PoolDispatcher(
            num_encoders=self._num_encoders,
            num_denoisers=self._num_denoisers,
            num_decoders=self._num_decoders,
            policy_name=dispatch_policy_name,
            max_slots_per_instance=max_slots_per_instance,
        )

        self._context = zmq.Context(io_threads=2)
        self._context_destroyed = False
        self._running = False
        self._thread: threading.Thread | None = None

        self._pending: dict[str, bytes] = {}  # request_id -> client ZMQ identity
        self._lock = threading.Lock()
        self._control_push_sockets: dict[str, zmq.Socket] = {}

        # FreeBufferSlots per instance
        self._encoder_free_slots = [max_slots_per_instance] * self._num_encoders
        self._denoiser_free_slots = [max_slots_per_instance] * self._num_denoisers
        self._decoder_free_slots = [max_slots_per_instance] * self._num_decoders
        self._denoiser_capacity_epochs = [0] * self._num_denoisers
        self._decoder_capacity_epochs = [0] * self._num_decoders

        # TTA queues per role type
        self._encoder_tta: deque[_EncoderTTAEntry] = deque()
        self._denoiser_tta: deque[_RoleTTAEntry] = deque()
        self._decoder_tta: deque[_RoleTTAEntry] = deque()

        self._transfer_mode = p2p_mode
        self._transfer_state: dict[str, _TransferRequestState] = {}
        self._server_profile_writer = None
        self._profiled_request_ids: set[str] = set()
        if profile_enabled:
            profile_dir = resolve_profile_dir(
                profile_output_dir,
                profile_run_id,
                deployment_mode="disaggregation",
            )
            self._server_profile_writer = RequestCsvProfiler(f"{profile_dir}/server.csv")

        # Per-instance registration: instance_idx -> {session_id, pool_ptr, pool_size}
        self._encoder_peers: dict[int, dict] = {}
        self._denoiser_peers: dict[int, dict] = {}
        self._decoder_peers: dict[int, dict] = {}

    @staticmethod
    def _set_transfer_phase(
        p2p: _TransferRequestState, phase: TransferPhase, *, now: float | None = None
    ) -> None:
        timestamp = time.monotonic() if now is None else now
        p2p.transfer_phase = phase
        p2p.phase_started_at = timestamp
        if p2p.handoff_started_at is None:
            p2p.handoff_started_at = timestamp

    @property
    def tracker(self) -> RequestTracker:
        return self._tracker

    @property
    def dispatcher(self) -> PoolDispatcher:
        return self._dispatcher

    def _mark_profile_event(
        self, request_id: str, event_name: str, timestamp_s: float | None = None
    ) -> None:
        if self._server_profile_writer is None:
            return
        try:
            self._tracker.mark_event(request_id, event_name, timestamp_s)
        except ValueError:
            pass

    def _finalize_server_profile(
        self,
        request_id: str,
        *,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        if (
            self._server_profile_writer is None
            or request_id in self._profiled_request_ids
        ):
            return

        record = self._tracker.get(request_id)
        if record is None:
            return
        if "finish_time_s" not in record.event_timestamps and record.is_terminal():
            record.event_timestamps["finish_time_s"] = time.time()

        row = {
            "request_id": request_id,
            "status": status or record.state.value,
            "error": error if error is not None else record.error,
            "request_arrival_time_s": record.submit_time_s,
            "encoder_tta_enter_time_s": record.state_timestamps.get(
                RequestState.ENCODER_WAITING.value
            ),
            "encoder_tta_leave_time_s": record.state_timestamps.get(
                RequestState.ENCODER_RUNNING.value
            ),
            "denoiser_tta_enter_time_s": record.state_timestamps.get(
                RequestState.DENOISING_WAITING.value
            ),
            "denoiser_tta_leave_time_s": record.state_timestamps.get(
                RequestState.DENOISING_RUNNING.value
            ),
            "decoder_tta_enter_time_s": record.state_timestamps.get(
                RequestState.DECODER_WAITING.value
            ),
            "decoder_tta_leave_time_s": record.state_timestamps.get(
                RequestState.DECODER_RUNNING.value
            ),
            "completion_signal_time_s": record.event_timestamps.get(
                "completion_signal_time_s"
            ),
            "finish_time_s": record.event_timestamps.get("finish_time_s"),
            "encoder_instance": record.encoder_instance,
            "denoiser_instance": record.denoiser_instance,
            "decoder_instance": record.decoder_instance,
        }
        self._server_profile_writer.write_row(row)
        self._profiled_request_ids.add(request_id)

    def _close_control_push_sockets(self) -> None:
        for sock in self._control_push_sockets.values():
            try:
                sock.close(linger=0)
            except TypeError:
                sock.close()
            except Exception:
                logger.exception("DiffusionServer: failed to close control push socket")
        self._control_push_sockets.clear()

    def _destroy_context_if_needed(self) -> None:
        if self._context_destroyed:
            return
        self._context_destroyed = True
        try:
            self._context.destroy(linger=0)
        except Exception:
            logger.exception("DiffusionServer: failed to destroy ZMQ context")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._event_loop,
            name="DiffusionServer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "DiffusionServer started: frontend=%s, "
            "%d encoder(s), %d denoiser(s), %d decoder(s), policy=%s, "
            "capacity=(%d/%d/%d)",
            self._frontend_endpoint,
            self._num_encoders,
            self._num_denoisers,
            self._num_decoders,
            type(self._dispatcher.encoder_policy).__name__,
            self._encoder_free_slots[0] if self._encoder_free_slots else 0,
            self._denoiser_free_slots[0] if self._denoiser_free_slots else 0,
            self._decoder_free_slots[0] if self._decoder_free_slots else 0,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "DiffusionServer: stop timed out while waiting for event loop thread"
                )
                return
            self._thread = None
        self._close_control_push_sockets()
        self._destroy_context_if_needed()

    def _event_loop(self) -> None:
        frontend, _ = get_zmq_socket(
            self._context, zmq.ROUTER, self._frontend_endpoint, bind=True
        )

        encoder_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._encoder_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            encoder_pushes.append(sock)

        denoiser_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._denoiser_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            denoiser_pushes.append(sock)

        decoder_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._decoder_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            decoder_pushes.append(sock)

        encoder_result_pull, _ = get_zmq_socket(
            self._context, zmq.PULL, self._encoder_result_endpoint, bind=True
        )
        denoiser_result_pull, _ = get_zmq_socket(
            self._context, zmq.PULL, self._denoiser_result_endpoint, bind=True
        )
        decoder_result_pull, _ = get_zmq_socket(
            self._context, zmq.PULL, self._decoder_result_endpoint, bind=True
        )

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(encoder_result_pull, zmq.POLLIN)
        poller.register(denoiser_result_pull, zmq.POLLIN)
        poller.register(decoder_result_pull, zmq.POLLIN)

        self._encoder_pushes = encoder_pushes
        self._denoiser_pushes = denoiser_pushes
        self._decoder_pushes = decoder_pushes
        self._frontend = frontend

        all_sockets = (
            [frontend, encoder_result_pull, denoiser_result_pull, decoder_result_pull]
            + encoder_pushes
            + denoiser_pushes
            + decoder_pushes
        )

        try:
            while self._running:
                events = dict(poller.poll(timeout=10))

                self._handle_timeouts()

                if frontend in events:
                    self._handle_client_request(frontend)

                if encoder_result_pull in events:
                    self._handle_role_result(encoder_result_pull, RoleType.ENCODER)

                if denoiser_result_pull in events:
                    self._handle_role_result(denoiser_result_pull, RoleType.DENOISER)

                if decoder_result_pull in events:
                    self._handle_role_result(decoder_result_pull, RoleType.DECODER)

                self._drain_all_queues()

        except Exception:
            logger.exception("DiffusionServer event loop error")
        finally:
            for sock in all_sockets:
                sock.close(linger=0)
            self._close_control_push_sockets()
            self._destroy_context_if_needed()

    def _handle_role_result(self, result_pull: zmq.Socket, role: RoleType) -> None:
        try:
            frames = result_pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        if is_transfer_message(frames):
            self._handle_transfer_result(frames, role)
            return

        if role == RoleType.ENCODER:
            self._handle_encoder_result_frames(frames)
        elif role == RoleType.DECODER:
            self._handle_decoder_result_frames(frames)
        else:
            logger.warning(
                "DiffusionServer: unexpected non-transfer frames from %s", role.value
            )

    def _handle_client_request(self, frontend: zmq.Socket) -> None:
        try:
            parts = frontend.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return

        if len(parts) < 3:
            return

        client_identity = parts[0]
        payload = parts[-1]

        try:
            reqs = pickle.loads(payload)
        except (pickle.UnpicklingError, EOFError):
            logger.warning("DiffusionServer: failed to deserialize request")
            return

        if not isinstance(reqs, list):
            reqs = [reqs]

        req = reqs[0]

        if isinstance(req, dict) or not hasattr(req, "request_id"):
            # Send empty reply so REQ socket doesn't hang
            try:
                frontend.send_multipart(
                    [client_identity, b"", pickle.dumps({"status": "ignored"})],
                    zmq.NOBLOCK,
                )
            except zmq.Again:
                pass
            return

        request_id = getattr(req, "request_id", None)
        if request_id is None:
            request_id = f"ds-{time.monotonic()}"

        metrics = getattr(req, "metrics", None)
        request_arrival_time_s = (
            getattr(metrics, "arrival_time_s", None) if metrics is not None else None
        )
        try:
            self._tracker.submit(request_id, submit_time_s=request_arrival_time_s)
        except ValueError:
            logger.warning("DiffusionServer: duplicate request_id %s", request_id)
            return

        with self._lock:
            self._pending[request_id] = client_identity

        try:
            self._tracker.transition(request_id, RequestState.ENCODER_WAITING)
        except ValueError:
            pass
        self._encoder_tta.append(
            _EncoderTTAEntry(
                request_id=request_id,
                client_identity=client_identity,
                payload=payload,
            )
        )

    def _handle_encoder_result_frames(self, frames: list) -> None:
        request_id = self._extract_request_id(frames)
        if request_id is None:
            logger.warning("DiffusionServer: encoder result missing request_id")
            return

        _tensor_fields, scalar_fields = unpack_tensors(frames, device="cpu")
        error = scalar_fields.get("_disagg_error") or scalar_fields.get("error")
        if not error:
            logger.warning(
                "DiffusionServer: unexpected non-transfer encoder result for %s",
                request_id,
            )
            return

        record = self._tracker.get(request_id)
        if record is not None and record.encoder_instance is not None:
            self._encoder_free_slots[record.encoder_instance] += 1

        self._complete_terminal(request_id, RequestState.FAILED, str(error))

    def _handle_decoder_result_frames(self, frames: list) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        request_id = self._extract_request_id(frames)
        if request_id is None:
            logger.warning("DiffusionServer: decoder result missing request_id")
            return
        self._mark_profile_event(request_id, "completion_signal_time_s")

        record = self._tracker.get(request_id)
        p2p = self._transfer_state.get(request_id)
        if p2p is not None:
            self._recycle_prealloc_slot(p2p, RoleType.DECODER)
            self._release_receiver_slot_if_needed(p2p, record)
        elif record and record.decoder_instance is not None:
            self._decoder_free_slots[record.decoder_instance] += 1
            self._bump_capacity_epoch(RoleType.DECODER, record.decoder_instance)

        tensor_fields, scalar_fields = unpack_tensors(frames, device="cpu")

        output_batch = OutputBatch(
            output=tensor_fields.get("output"),
            audio=tensor_fields.get("audio"),
            audio_sample_rate=scalar_fields.get("audio_sample_rate"),
            error=scalar_fields.get("error"),
        )

        try:
            if output_batch.error:
                self._tracker.transition(
                    request_id, RequestState.FAILED, error=output_batch.error
                )
            else:
                self._tracker.transition(request_id, RequestState.DONE)
        except ValueError:
            pass

        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            logger.warning(
                "DiffusionServer: no pending client for decoder result %s",
                request_id,
            )
            self._mark_profile_event(request_id, "finish_time_s")
            self._finalize_server_profile(
                request_id,
                status="failed" if output_batch.error else "completed",
                error=output_batch.error,
            )
            self._tracker.remove(request_id)
            return

        self._mark_profile_event(request_id, "finish_time_s")
        try:
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(output_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer: failed to send result for %s: %s",
                request_id,
                e,
            )

        logger.debug("DiffusionServer: returned result for %s", request_id)
        self._transfer_state.pop(request_id, None)
        self._finalize_server_profile(
            request_id,
            status="failed" if output_batch.error else "completed",
            error=output_batch.error,
        )
        self._tracker.remove(request_id)

    def _dispatch_to_encoder(
        self, request_id: str, payload: bytes, encoder_idx: int
    ) -> None:
        self._encoder_free_slots[encoder_idx] -= 1

        try:
            self._tracker.transition(
                request_id,
                RequestState.ENCODER_RUNNING,
                encoder_instance=encoder_idx,
            )
        except ValueError:
            pass

        self._encoder_pushes[encoder_idx].send_multipart(
            [request_id.encode("utf-8"), payload]
        )
        logger.debug(
            "DiffusionServer: dispatched %s to encoder[%d] (free=%d)",
            request_id,
            encoder_idx,
            self._encoder_free_slots[encoder_idx],
        )

    def _drain_all_queues(self) -> None:
        self._drain_encoder_tta()
        self._drain_denoiser_tta()
        self._drain_decoder_tta()

    def _drain_encoder_tta(self) -> None:
        while self._encoder_tta:
            idx = self._dispatcher.select_encoder_with_capacity(
                self._encoder_free_slots
            )
            if idx is None:
                break
            entry = self._encoder_tta.popleft()
            self._dispatch_to_encoder(entry.request_id, entry.payload, idx)

    def _drain_denoiser_tta(self) -> None:
        while self._denoiser_tta:
            entry = self._denoiser_tta[0]
            p2p = entry.transfer_state
            idx = self._select_downstream_instance_with_capacity(
                RoleType.DENOISER,
                p2p,
            )
            if idx is None:
                break
            entry = self._denoiser_tta.popleft()
            if p2p is not None:
                p2p.downstream_tta_enqueued = False
            self._transfer_dispatch_to_denoiser(
                entry.request_id, entry.transfer_state, idx
            )

    def _drain_decoder_tta(self) -> None:
        while self._decoder_tta:
            entry = self._decoder_tta[0]
            p2p = entry.transfer_state
            idx = self._select_downstream_instance_with_capacity(
                RoleType.DECODER,
                p2p,
            )
            if idx is None:
                break
            entry = self._decoder_tta.popleft()
            if p2p is not None:
                p2p.downstream_tta_enqueued = False
            self._transfer_dispatch_to_decoder(
                entry.request_id, entry.transfer_state, idx
            )

    def _extract_request_id(self, frames: list) -> str | None:
        try:
            metadata = json.loads(frames[0])
            return metadata.get("scalar_fields", {}).get("request_id")
        except (json.JSONDecodeError, IndexError, TypeError):
            return None

    def _complete_terminal(
        self,
        request_id: str,
        terminal_state: RequestState,
        error_msg: str,
    ) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        logger.error("DiffusionServer: request %s failed: %s", request_id, error_msg)

        try:
            self._tracker.transition(request_id, terminal_state, error=error_msg)
        except ValueError:
            pass

        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            self._tracker.remove(request_id)
            return

        error_batch = OutputBatch(error=error_msg)
        try:
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(error_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer: failed to send error for %s: %s",
                request_id,
                e,
            )

        self._tracker.remove(request_id)

    def _handle_timeouts(self) -> None:
        now = time.monotonic()
        wait_timed_out = []
        alloc_phase_timed_out = []
        for request_id, p2p in list(self._transfer_state.items()):
            if (
                p2p.handoff_started_at is not None
                and (now - p2p.handoff_started_at) > self._downstream_wait_timeout_s
            ):
                wait_timed_out.append(request_id)
                continue
            if (
                p2p.transfer_phase == TransferPhase.WAITING_ALLOC_RESULT
                and p2p.phase_started_at is not None
                and (now - p2p.phase_started_at) > self._alloc_result_timeout_s
            ):
                alloc_phase_timed_out.append(request_id)

        for request_id in alloc_phase_timed_out:
            p2p = self._transfer_state.get(request_id)
            if p2p is None or p2p.transfer_phase != TransferPhase.WAITING_ALLOC_RESULT:
                continue
            if not p2p.receiver_role or p2p.receiver_instance < 0:
                continue
            record = self._tracker.get(request_id)
            role_enum = RoleType.from_string(p2p.receiver_role)
            rejected_instance = p2p.receiver_instance
            self._release_receiver_slot_if_needed(p2p, record, update_epoch=False)
            self._recycle_prealloc_slot(p2p, role_enum)
            self._clear_receiver_dispatch(p2p)
            self._requeue_downstream_transfer(
                request_id,
                p2p,
                role_enum=role_enum,
                rejected_instance=rejected_instance,
                now=now,
            )

        for request_id in wait_timed_out:
            record = self._tracker.get(request_id)
            p2p = self._transfer_state.pop(request_id, None)
            if p2p is None:
                continue
            self._set_transfer_phase(p2p, TransferPhase.ABORTING, now=now)
            if record is not None:
                self._release_sender_slot_if_needed(p2p, record, update_epoch=False)
                self._release_receiver_slot_if_needed(
                    p2p, record, update_epoch=False
                )
            receiver_role = (
                RoleType.from_string(p2p.receiver_role) if p2p.receiver_role else None
            )
            if receiver_role is not None:
                self._recycle_prealloc_slot(p2p, receiver_role)
            self._send_abort(
                request_id,
                p2p,
                to_sender=True,
                to_receiver=bool(p2p.receiver_control_endpoint),
                reason=(
                    f"Downstream wait timeout after {self._downstream_wait_timeout_s}s"
                ),
                source="timeout",
            )
            self._complete_terminal(
                request_id,
                RequestState.TIMED_OUT,
                f"DiffusionServer downstream wait timeout: request {request_id} "
                f"did not acquire a downstream slot within {self._downstream_wait_timeout_s}s",
            )

        timed_out = self._tracker.find_timed_out(self._timeout_s)
        for request_id in timed_out:
            if request_id in wait_timed_out:
                continue
            record = self._tracker.get(request_id)
            p2p = self._transfer_state.pop(request_id, None)
            if p2p is not None and record is not None:
                self._set_transfer_phase(p2p, TransferPhase.ABORTING, now=now)
                self._send_abort(
                    request_id,
                    p2p,
                    to_sender=True,
                    to_receiver=bool(p2p.receiver_control_endpoint),
                    reason=f"Global timeout after {self._timeout_s}s",
                    source="timeout",
                )
                self._release_sender_slot_if_needed(p2p, record)
                self._release_receiver_slot_if_needed(p2p, record)
                receiver_role = (
                    RoleType.from_string(p2p.receiver_role)
                    if p2p.receiver_role
                    else None
                )
                if receiver_role is not None:
                    self._recycle_prealloc_slot(p2p, receiver_role)
            elif record:
                self._free_slot_for_record(record)

            self._complete_terminal(
                request_id,
                RequestState.TIMED_OUT,
                f"DiffusionServer timeout: request {request_id} "
                f"not completed within {self._timeout_s}s",
            )

        all_timed_out = set(wait_timed_out) | set(timed_out)
        if all_timed_out:
            self._encoder_tta = deque(
                e for e in self._encoder_tta if e.request_id not in all_timed_out
            )
            self._denoiser_tta = deque(
                e for e in self._denoiser_tta if e.request_id not in all_timed_out
            )
            self._decoder_tta = deque(
                e for e in self._decoder_tta if e.request_id not in all_timed_out
            )

    def _free_slot_for_record(self, record) -> None:
        if (
            record.state in (RequestState.ENCODER_RUNNING, RequestState.ENCODER_DONE)
            and record.encoder_instance is not None
        ):
            self._encoder_free_slots[record.encoder_instance] += 1
        if (
            record.state
            in (
                RequestState.DENOISING_WAITING,
                RequestState.DENOISING_RUNNING,
                RequestState.DENOISING_DONE,
            )
            and record.denoiser_instance is not None
        ):
            self._denoiser_free_slots[record.denoiser_instance] += 1
            self._bump_capacity_epoch(RoleType.DENOISER, record.denoiser_instance)
        if (
            record.state in (RequestState.DECODER_WAITING, RequestState.DECODER_RUNNING)
            and record.decoder_instance is not None
        ):
            self._decoder_free_slots[record.decoder_instance] += 1
            self._bump_capacity_epoch(RoleType.DECODER, record.decoder_instance)

    def _peer_registry(self, role: RoleType) -> dict[int, dict]:
        if role == RoleType.ENCODER:
            return self._encoder_peers
        if role == RoleType.DENOISER:
            return self._denoiser_peers
        if role == RoleType.DECODER:
            return self._decoder_peers
        raise ValueError(f"Unsupported role for peer registry: {role}")

    def _role_pushes(self, role: RoleType) -> list[zmq.Socket]:
        if role == RoleType.ENCODER:
            return self._encoder_pushes
        if role == RoleType.DENOISER:
            return self._denoiser_pushes
        if role == RoleType.DECODER:
            return self._decoder_pushes
        raise ValueError(f"Unsupported role for push sockets: {role}")

    def _send_control_message(self, endpoint: str, msg) -> None:
        if not endpoint:
            return
        sock = self._control_push_sockets.get(endpoint)
        if sock is None:
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, endpoint, bind=False)
            self._control_push_sockets[endpoint] = sock
        sock.send_multipart(encode_transfer_msg(msg))

    def _bump_capacity_epoch(self, role: RoleType, instance_id: int | None) -> None:
        if instance_id is None or instance_id < 0:
            return
        if role == RoleType.DENOISER and instance_id < len(self._denoiser_capacity_epochs):
            self._denoiser_capacity_epochs[instance_id] += 1
        elif role == RoleType.DECODER and instance_id < len(self._decoder_capacity_epochs):
            self._decoder_capacity_epochs[instance_id] += 1

    def _current_capacity_epoch(self, role: RoleType, instance_id: int) -> int:
        if role == RoleType.DENOISER and 0 <= instance_id < len(self._denoiser_capacity_epochs):
            return self._denoiser_capacity_epochs[instance_id]
        if role == RoleType.DECODER and 0 <= instance_id < len(self._decoder_capacity_epochs):
            return self._decoder_capacity_epochs[instance_id]
        return 0

    def _excluded_instances_for_request(
        self, p2p: _TransferRequestState, role: RoleType
    ) -> set[int]:
        excluded = set()
        for instance_id, reject_epoch in p2p.rejected_instances.items():
            if self._current_capacity_epoch(role, instance_id) == reject_epoch:
                excluded.add(instance_id)
        return excluded

    def _select_downstream_instance_with_capacity(
        self,
        role: RoleType,
        p2p: _TransferRequestState | None = None,
        *,
        extra_excluded: set[int] | None = None,
    ) -> int | None:
        excluded = set()
        if p2p is not None:
            excluded |= self._excluded_instances_for_request(p2p, role)
        if extra_excluded:
            excluded |= set(extra_excluded)
        excluded_instances = excluded or None
        if role == RoleType.DENOISER:
            return self._dispatcher.select_denoiser_with_capacity(
                self._denoiser_free_slots, excluded_instances=excluded_instances
            )
        if role == RoleType.DECODER:
            return self._dispatcher.select_decoder_with_capacity(
                self._decoder_free_slots, excluded_instances=excluded_instances
            )
        return None

    def _enqueue_role_wait(
        self, queue_obj: deque[_RoleTTAEntry], request_id: str, p2p: _TransferRequestState
    ) -> None:
        if p2p.downstream_tta_enqueued:
            return
        self._set_transfer_phase(p2p, TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT)
        queue_obj.append(_RoleTTAEntry(request_id=request_id, transfer_state=p2p))
        p2p.downstream_tta_enqueued = True

    def _send_abort(
        self,
        request_id: str,
        p2p: _TransferRequestState | None,
        *,
        to_sender: bool,
        to_receiver: bool,
        reason: str,
        source: str,
    ) -> None:
        if p2p is None:
            return
        if to_sender and not p2p.sender_abort_sent and p2p.sender_control_endpoint:
            try:
                self._send_control_message(
                    p2p.sender_control_endpoint,
                    TransferAbortMsg(
                        request_id=request_id,
                        reason=reason,
                        source=source,
                    ),
                )
            except Exception:
                logger.exception("DiffusionServer: failed to send sender abort")
            else:
                p2p.sender_abort_sent = True
        if to_receiver and not p2p.receiver_abort_sent and p2p.receiver_control_endpoint:
            try:
                self._send_control_message(
                    p2p.receiver_control_endpoint,
                    TransferAbortMsg(
                        request_id=request_id,
                        reason=reason,
                        source=source,
                    ),
                )
            except Exception:
                logger.exception("DiffusionServer: failed to send receiver abort")
            else:
                p2p.receiver_abort_sent = True

    def _dispatch_transfer_alloc(
        self,
        request_id: str,
        p2p: _TransferRequestState,
        receiver_role: RoleType,
        receiver_idx: int,
    ) -> None:
        peer_info = self._peer_registry(receiver_role).get(receiver_idx, {})
        p2p.receiver_role = receiver_role.value
        p2p.receiver_instance = receiver_idx
        p2p.receiver_control_endpoint = peer_info.get("control_endpoint", "")
        p2p.receiver_session_id = peer_info.get("session_id", "")
        p2p.receiver_pool_ptr = peer_info.get("pool_ptr", 0)
        p2p.receiver_slot_offset = 0
        p2p.receiver_slot_size = 0
        p2p.receiver_meta_pool_ptr = peer_info.get("meta_pool_ptr", 0)
        p2p.receiver_meta_slot_offset = 0
        p2p.receiver_meta_slot_size = 0
        p2p.receiver_host_id = peer_info.get("host_id", "")
        p2p.receiver_supports_local_copy = bool(
            peer_info.get("supports_local_copy", False)
        )
        p2p.prealloc_slot_id = None
        p2p.receiver_slot_released = False
        p2p.receiver_prealloc_recycled = False
        p2p.transfer_completion_processed = False

        alloc_msg = TransferAllocMsg(
            request_id=request_id,
            data_size=p2p.data_size,
            meta_size=p2p.meta_size,
            source_role=p2p.sender_role,
            source_instance=p2p.sender_instance,
            source_control_endpoint=p2p.sender_control_endpoint,
            source_host_id=p2p.sender_host_id,
        )

        self._role_pushes(receiver_role)[receiver_idx].send_multipart(
            encode_transfer_msg(alloc_msg)
        )
        self._set_transfer_phase(p2p, TransferPhase.WAITING_ALLOC_RESULT)

    def _release_sender_slot_if_needed(
        self,
        p2p: _TransferRequestState | None,
        record,
        *,
        update_epoch: bool = True,
    ) -> None:
        if p2p is None or p2p.sender_slot_released or record is None:
            return

        if p2p.sender_role == RoleType.ENCODER.value and record.encoder_instance is not None:
            self._encoder_free_slots[record.encoder_instance] += 1
            p2p.sender_slot_released = True
        elif (
            p2p.sender_role == RoleType.DENOISER.value
            and record.denoiser_instance is not None
        ):
            self._denoiser_free_slots[record.denoiser_instance] += 1
            p2p.sender_slot_released = True
            if update_epoch:
                self._bump_capacity_epoch(RoleType.DENOISER, record.denoiser_instance)

    def _release_receiver_slot_if_needed(
        self,
        p2p: _TransferRequestState | None,
        record,
        *,
        update_epoch: bool = True,
    ) -> None:
        if p2p is None or p2p.receiver_slot_released or record is None:
            return

        if (
            p2p.receiver_role == RoleType.DENOISER.value
            and record.denoiser_instance is not None
        ):
            self._denoiser_free_slots[record.denoiser_instance] += 1
            p2p.receiver_slot_released = True
            if update_epoch:
                self._bump_capacity_epoch(RoleType.DENOISER, record.denoiser_instance)
        elif (
            p2p.receiver_role == RoleType.DECODER.value
            and record.decoder_instance is not None
        ):
            self._decoder_free_slots[record.decoder_instance] += 1
            p2p.receiver_slot_released = True
            if update_epoch:
                self._bump_capacity_epoch(RoleType.DECODER, record.decoder_instance)

    def _handle_transfer_result(self, frames: list, role: RoleType) -> None:
        try:
            msg = decode_transfer_msg(frames)
        except (ValueError, Exception) as e:
            logger.error("DiffusionServer: failed to decode transfer message: %s", e)
            return

        msg_type = msg.get("msg_type")

        if msg_type == TransferMsgType.REGISTER:
            self._handle_transfer_register(msg)
        elif msg_type == TransferMsgType.STAGED:
            self._handle_transfer_staged(msg)
        elif msg_type == TransferMsgType.ALLOC_ACCEPTED:
            self._handle_alloc_accepted(msg)
        elif msg_type == TransferMsgType.ALLOC_REJECT:
            self._handle_alloc_reject(msg)
        elif msg_type == TransferMsgType.PUSHED:
            self._handle_transfer_pushed(msg)
        elif msg_type == TransferMsgType.DONE:
            self._handle_transfer_done(msg, role)
        else:
            logger.warning("DiffusionServer: unknown transfer msg_type=%s", msg_type)

    def _handle_transfer_register(self, msg: dict) -> None:
        try:
            role = RoleType.from_string(msg.get("role", ""))
        except ValueError:
            logger.warning(
                "DiffusionServer transfer: unknown role in register: %s",
                msg.get("role"),
            )
            return
        info = {
            "instance_id": msg.get("instance_id", 0),
            "session_id": msg.get("session_id", ""),
            "pool_ptr": msg.get("pool_ptr", 0),
            "pool_size": msg.get("pool_size", 0),
            "meta_pool_ptr": msg.get("meta_pool_ptr", 0),
            "meta_pool_size": msg.get("meta_pool_size", 0),
            "control_endpoint": msg.get("control_endpoint", ""),
            "work_endpoint": msg.get("work_endpoint", ""),
            "rank0_only": bool(msg.get("rank0_only", True)),
            "role_device": msg.get("role_device", "auto"),
            "host_id": msg.get("host_id", ""),
            "supports_local_copy": bool(msg.get("supports_local_copy", False)),
            "data_shm_name": msg.get("data_shm_name"),
            "meta_shm_name": msg.get("meta_shm_name"),
        }
        info["free_preallocated_slots"] = []

        if role == RoleType.ENCODER:
            idx = info["instance_id"]
            self._encoder_peers[idx] = info
        elif role == RoleType.DENOISER:
            idx = info["instance_id"]
            self._denoiser_peers[idx] = info
        elif role == RoleType.DECODER:
            idx = info["instance_id"]
            self._decoder_peers[idx] = info
        else:
            idx = 0
        logger.info(
            "DiffusionServer transfer: registered %s[%d] session=%s control=%s prealloc=%d",
            role,
            idx,
            info["session_id"],
            info["control_endpoint"],
            len(info["free_preallocated_slots"]),
        )

    def _handle_alloc_reject(self, msg: dict) -> None:
        request_id = msg.get("request_id", "")
        p2p = self._transfer_state.get(request_id)
        if p2p is None:
            return

        receiver_role = msg.get("receiver_role", "")
        receiver_instance = msg.get("receiver_instance", -1)
        if (
            receiver_role != p2p.receiver_role
            or receiver_instance != p2p.receiver_instance
        ):
            return

        record = self._tracker.get(request_id)
        role_enum = RoleType.from_string(receiver_role)
        self._release_receiver_slot_if_needed(p2p, record, update_epoch=False)
        self._recycle_prealloc_slot(p2p, role_enum)
        self._clear_receiver_dispatch(p2p)

        if msg.get("retryable", True):
            retry_candidate = self._select_downstream_instance_with_capacity(
                role_enum,
                p2p,
                extra_excluded={receiver_instance},
            )
            if retry_candidate is not None:
                self._requeue_downstream_transfer(
                    request_id,
                    p2p,
                    role_enum=role_enum,
                    rejected_instance=receiver_instance,
                )
                return
            logger.error(
                "DiffusionServer: %s alloc reject from %s[%d] has no alternative "
                "instance; failing request immediately",
                request_id,
                role_enum.value,
                receiver_instance,
            )
            reason = (
                f"{msg.get('reason', 'fatal downstream allocation failure')}; "
                f"no alternative {role_enum.value} instance available"
            )
        else:
            reason = msg.get("reason", "fatal downstream allocation failure")

        self._set_transfer_phase(p2p, TransferPhase.ABORTING)
        self._send_abort(
            request_id,
            p2p,
            to_sender=True,
            to_receiver=False,
            reason=reason,
            source="alloc_failed",
        )
        self._release_sender_slot_if_needed(p2p, record, update_epoch=False)
        self._complete_terminal(
            request_id,
            RequestState.FAILED,
            f"Fatal downstream allocation failure: {reason}",
        )
        self._transfer_state.pop(request_id, None)

    def _transfer_dispatch_to_denoiser(
        self, request_id: str, p2p: _TransferRequestState, denoiser_idx: int
    ) -> None:
        self._denoiser_free_slots[denoiser_idx] -= 1

        try:
            self._tracker.update_instances(
                request_id,
                denoiser_instance=denoiser_idx,
            )
        except ValueError:
            pass

        self._dispatch_transfer_alloc(
            request_id=request_id,
            p2p=p2p,
            receiver_role=RoleType.DENOISER,
            receiver_idx=denoiser_idx,
        )

    def _recycle_prealloc_slot(
        self, p2p: _TransferRequestState, role: RoleType
    ) -> None:
        del role
        if p2p is None:
            return
        p2p.receiver_prealloc_recycled = True
        p2p.prealloc_slot_id = None

    def _transfer_dispatch_to_decoder(
        self, request_id: str, p2p: _TransferRequestState, decoder_idx: int
    ) -> None:
        self._decoder_free_slots[decoder_idx] -= 1

        try:
            self._tracker.update_instances(
                request_id,
                decoder_instance=decoder_idx,
            )
        except ValueError:
            pass

        self._dispatch_transfer_alloc(
            request_id=request_id,
            p2p=p2p,
            receiver_role=RoleType.DECODER,
            receiver_idx=decoder_idx,
        )

    def _transfer_return_to_client(self, request_id: str, result_frames: list) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        self._mark_profile_event(request_id, "completion_signal_time_s")
        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            self._mark_profile_event(request_id, "finish_time_s")
            self._finalize_server_profile(request_id, status="completed")
            self._tracker.remove(request_id)
            return

        try:
            raw_frames = []
            for f in result_frames:
                if isinstance(f, str):
                    raw_frames.append(bytes.fromhex(f))
                else:
                    raw_frames.append(f)

            tensor_fields, scalar_fields = unpack_tensors(raw_frames)
            output_batch = OutputBatch(
                output=tensor_fields.get("output"),
                audio=tensor_fields.get("audio"),
                audio_sample_rate=scalar_fields.get("audio_sample_rate"),
                error=scalar_fields.get("error"),
            )

            self._mark_profile_event(request_id, "finish_time_s")
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(output_batch)]
            )
        except Exception as e:
            logger.error(
                "DiffusionServer transfer: failed to send result for %s: %s",
                request_id,
                e,
            )
            self._mark_profile_event(request_id, "finish_time_s")

        self._finalize_server_profile(
            request_id,
            status="completed",
        )
        self._tracker.remove(request_id)

    def _transfer_return_to_client_from_msg(self, request_id: str, msg: dict) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        self._mark_profile_event(request_id, "completion_signal_time_s")
        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            self._mark_profile_event(request_id, "finish_time_s")
            self._finalize_server_profile(
                request_id,
                status="failed" if msg.get("error") else "completed",
                error=msg.get("error"),
            )
            self._tracker.remove(request_id)
            return

        output_batch = OutputBatch(error=msg.get("error"))

        try:
            self._mark_profile_event(request_id, "finish_time_s")
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(output_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer transfer: failed to send result for %s: %s",
                request_id,
                e,
            )
            self._mark_profile_event(request_id, "finish_time_s")
        self._finalize_server_profile(
            request_id,
            status="failed" if msg.get("error") else "completed",
            error=msg.get("error"),
        )
        self._tracker.remove(request_id)

    def _complete_terminal(
        self,
        request_id: str,
        terminal_state: RequestState,
        error_msg: str,
    ) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        logger.error("DiffusionServer: %s - %s", request_id, error_msg)

        try:
            self._tracker.transition(request_id, terminal_state, error=error_msg)
        except ValueError:
            pass

        self._encoder_tta = deque(
            entry for entry in self._encoder_tta if entry.request_id != request_id
        )
        self._denoiser_tta = deque(
            entry for entry in self._denoiser_tta if entry.request_id != request_id
        )
        self._decoder_tta = deque(
            entry for entry in self._decoder_tta if entry.request_id != request_id
        )

        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            self._mark_profile_event(request_id, "finish_time_s")
            self._finalize_server_profile(
                request_id,
                status=terminal_state.value,
                error=error_msg,
            )
            self._tracker.remove(request_id)
            return

        error_batch = OutputBatch(error=error_msg)
        try:
            self._mark_profile_event(request_id, "finish_time_s")
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(error_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer: failed to send error for %s: %s",
                request_id,
                e,
            )

        self._finalize_server_profile(
            request_id,
            status=terminal_state.value,
            error=error_msg,
        )
        self._tracker.remove(request_id)

    def _clear_receiver_dispatch(self, p2p: _TransferRequestState) -> None:
        p2p.receiver_role = ""
        p2p.receiver_session_id = ""
        p2p.receiver_pool_ptr = 0
        p2p.receiver_slot_offset = 0
        p2p.receiver_slot_size = 0
        p2p.receiver_meta_pool_ptr = 0
        p2p.receiver_meta_slot_offset = 0
        p2p.receiver_meta_slot_size = 0
        p2p.receiver_control_endpoint = ""
        p2p.receiver_host_id = ""
        p2p.receiver_supports_local_copy = False
        p2p.receiver_instance = -1
        p2p.prealloc_slot_id = None
        p2p.receiver_slot_released = False
        p2p.receiver_prealloc_recycled = False
        p2p.alloc_accepted = False
        p2p.receiver_abort_sent = False
        p2p.transfer_completion_processed = False

    def _requeue_downstream_transfer(
        self,
        request_id: str,
        p2p: _TransferRequestState,
        *,
        role_enum: RoleType,
        rejected_instance: int,
        now: float | None = None,
    ) -> None:
        p2p.rejected_instances[rejected_instance] = self._current_capacity_epoch(
            role_enum, rejected_instance
        )
        if p2p.handoff_started_at is not None:
            p2p.downstream_wait_since = p2p.handoff_started_at
        self._set_transfer_phase(
            p2p, TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT, now=now
        )
        if role_enum == RoleType.DENOISER:
            self._enqueue_role_wait(self._denoiser_tta, request_id, p2p)
        else:
            self._enqueue_role_wait(self._decoder_tta, request_id, p2p)

    def _handle_alloc_accepted(self, msg: dict) -> None:
        request_id = msg.get("request_id", "")
        p2p = self._transfer_state.get(request_id)
        if p2p is None:
            return

        receiver_role = msg.get("receiver_role", "")
        receiver_instance = msg.get("receiver_instance", -1)
        if (
            receiver_role != p2p.receiver_role
            or receiver_instance != p2p.receiver_instance
        ):
            return

        p2p.alloc_accepted = True
        p2p.downstream_wait_since = None
        self._set_transfer_phase(p2p, TransferPhase.SENDING)
        if p2p.prealloc_slot_id is not None and msg.get("prealloc_slot_id") is None:
            self._recycle_prealloc_slot(
                p2p, RoleType.from_string(p2p.receiver_role)
            )

    def _handle_transfer_staged(self, msg: dict) -> None:
        request_id = msg["request_id"]
        record = self._tracker.get(request_id)
        if record is None:
            logger.warning(
                "DiffusionServer transfer: staged request %s is not tracked", request_id
            )
            return
        encoder_idx = record.encoder_instance if record.encoder_instance is not None else 0
        encoder_peer = self._encoder_peers.get(encoder_idx, {})

        p2p = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            sender_session_id=msg.get("session_id", ""),
            sender_pool_ptr=msg.get("pool_ptr", 0),
            sender_slot_offset=msg.get("slot_offset", 0),
            sender_meta_pool_ptr=msg.get("meta_pool_ptr", 0),
            sender_meta_slot_offset=msg.get("meta_slot_offset", 0),
            sender_control_endpoint=encoder_peer.get("control_endpoint", ""),
            sender_host_id=encoder_peer.get("host_id", ""),
            data_size=msg.get("data_size", 0),
            meta_size=msg.get("meta_size", 0),
            sender_instance=encoder_idx,
            transfer_phase=TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT,
            handoff_started_at=time.monotonic(),
            phase_started_at=time.monotonic(),
            downstream_wait_since=time.monotonic(),
        )
        self._transfer_state[request_id] = p2p

        try:
            self._tracker.transition(request_id, RequestState.ENCODER_DONE)
        except ValueError:
            pass

        try:
            self._tracker.transition(request_id, RequestState.DENOISING_WAITING)
        except ValueError:
            pass
        self._enqueue_role_wait(self._denoiser_tta, request_id, p2p)

    def _handle_transfer_pushed(self, msg: dict) -> None:
        request_id = msg["request_id"]
        p2p = self._transfer_state.get(request_id)
        if p2p is None:
            logger.warning(
                "DiffusionServer transfer: no state for pushed %s", request_id
            )
            return

        if p2p.transfer_completion_processed:
            return

        record = self._tracker.get(request_id)
        p2p.transfer_completion_processed = True
        if not msg.get("success", True):
            p2p.last_send_error = msg.get("error") or "transfer push failed"
            self._set_transfer_phase(p2p, TransferPhase.ABORTING)
            self._send_abort(
                request_id,
                p2p,
                to_sender=True,
                to_receiver=bool(p2p.receiver_control_endpoint),
                reason=p2p.last_send_error,
                source="transfer_failed",
            )
            if record is not None:
                self._release_sender_slot_if_needed(p2p, record)
                self._release_receiver_slot_if_needed(p2p, record)
            if p2p.receiver_role:
                self._recycle_prealloc_slot(
                    p2p, RoleType.from_string(p2p.receiver_role)
                )
            self._complete_terminal(
                request_id,
                RequestState.FAILED,
                f"Transfer push failed: {msg.get('error') or 'unknown error'}",
            )
            self._transfer_state.pop(request_id, None)
            return

        self._release_sender_slot_if_needed(p2p, record)
        self._set_transfer_phase(p2p, TransferPhase.RUNNING_DOWNSTREAM)
        if record is None:
            return

        try:
            if p2p.receiver_role == RoleType.DENOISER.value:
                self._tracker.transition(request_id, RequestState.DENOISING_RUNNING)
            elif p2p.receiver_role == RoleType.DECODER.value:
                self._tracker.transition(request_id, RequestState.DECODER_RUNNING)
        except ValueError:
            pass

    def _handle_transfer_done(self, msg: dict, role: RoleType) -> None:
        request_id = msg.get("request_id", "")
        error = msg.get("error")
        p2p = self._transfer_state.get(request_id)

        if role == RoleType.DENOISER:
            record = self._tracker.get(request_id)

            if p2p is not None:
                self._recycle_prealloc_slot(p2p, RoleType.DENOISER)
                if error or not msg.get("staged_for_decoder"):
                    self._release_receiver_slot_if_needed(p2p, record)

            if error:
                self._complete_terminal(
                    request_id,
                    RequestState.FAILED,
                    f"Denoiser error: {error}",
                )
                self._transfer_state.pop(request_id, None)
                return

            try:
                self._tracker.transition(request_id, RequestState.DENOISING_DONE)
            except ValueError:
                pass

            if p2p is not None and msg.get("staged_for_decoder"):
                if p2p.decoder_dispatch_enqueued:
                    return
                denoiser_idx = record.denoiser_instance if record else 0
                denoiser_peer = self._denoiser_peers.get(denoiser_idx, {})
                self._clear_receiver_dispatch(p2p)
                p2p.sender_role = RoleType.DENOISER.value
                p2p.sender_session_id = msg.get("session_id", "")
                p2p.sender_pool_ptr = msg.get("pool_ptr", 0)
                p2p.sender_slot_offset = msg.get("slot_offset", 0)
                p2p.sender_meta_pool_ptr = msg.get("meta_pool_ptr", 0)
                p2p.sender_meta_slot_offset = msg.get("meta_slot_offset", 0)
                p2p.sender_control_endpoint = denoiser_peer.get("control_endpoint", "")
                p2p.sender_host_id = denoiser_peer.get("host_id", "")
                p2p.data_size = msg.get("data_size", 0)
                p2p.meta_size = msg.get("meta_size", 0)
                p2p.sender_instance = denoiser_idx
                p2p.sender_slot_released = False
                p2p.transfer_completion_processed = False
                p2p.alloc_accepted = False
                p2p.handoff_started_at = time.monotonic()
                p2p.phase_started_at = p2p.handoff_started_at
                p2p.transfer_phase = TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT
                p2p.downstream_wait_since = time.monotonic()
                p2p.downstream_tta_enqueued = False
                p2p.rejected_instances.clear()
                p2p.send_attempts = 0
                p2p.last_send_error = None
                p2p.sender_abort_sent = False
                p2p.receiver_abort_sent = False
                p2p.decoder_dispatch_enqueued = True

                try:
                    self._tracker.transition(request_id, RequestState.DECODER_WAITING)
                except ValueError:
                    pass
                self._enqueue_role_wait(self._decoder_tta, request_id, p2p)
            else:
                self._transfer_state.pop(request_id, None)

        elif role == RoleType.DECODER:
            record = self._tracker.get(request_id)
            if p2p is not None:
                self._recycle_prealloc_slot(p2p, RoleType.DECODER)
                self._release_receiver_slot_if_needed(p2p, record)

            if error:
                self._complete_terminal(
                    request_id,
                    RequestState.FAILED,
                    f"Decoder error: {error}",
                )
            else:
                try:
                    self._tracker.transition(request_id, RequestState.DONE)
                except ValueError:
                    pass

                result_frames = msg.get("result_frames")
                if result_frames:
                    self._transfer_return_to_client(request_id, result_frames)
                else:
                    self._transfer_return_to_client_from_msg(request_id, msg)

            self._transfer_state.pop(request_id, None)

    def get_stats(self) -> dict:
        with self._lock:
            pending_count = len(self._pending)
        return {
            "role": "diffusion_server",
            "transfer_mode": self._transfer_mode,
            "num_encoders": self._num_encoders,
            "num_denoisers": self._num_denoisers,
            "num_decoders": self._num_decoders,
            "pending_requests": pending_count,
            "dispatch_policy": type(self._dispatcher.encoder_policy).__name__,
            "encoder_free_slots": list(self._encoder_free_slots),
            "denoiser_free_slots": list(self._denoiser_free_slots),
            "decoder_free_slots": list(self._decoder_free_slots),
            "encoder_tta_depth": len(self._encoder_tta),
            "denoiser_tta_depth": len(self._denoiser_tta),
            "decoder_tta_depth": len(self._decoder_tta),
            "transfer_active_transfers": len(self._transfer_state),
            "encoder_peers": len(self._encoder_peers),
            "denoiser_peers": len(self._denoiser_peers),
            "decoder_peers": len(self._decoder_peers),
            "tracker": self._tracker.snapshot(),
        }
