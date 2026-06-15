# SPDX-License-Identifier: Apache-2.0
"""Per-instance transfer manager for disaggregated diffusion roles."""

from __future__ import annotations

import logging
import threading

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    SlotHandle,
    TransferMetaBuffer,
    TransferTensorBuffer,
    estimate_transfer_meta_bytes,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    BaseTransferEngine,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.state import (
    PendingReceive,
    StagedTransfer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.utils import (
    record_device_event_if_needed,
    record_load_event,
    tensor_fields_need_device_event,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

logger = logging.getLogger(__name__)


class DiffusionTransferManager:
    """Manages host-side tensor/meta transfers for one diffusion role instance."""

    def __init__(
        self,
        engine: BaseTransferEngine,
        buffer: TransferTensorBuffer,
        meta_buffer: TransferMetaBuffer | None = None,
        *,
        host_id: str = "",
        send_retry_limit: int = 2,
    ):
        self._engine = engine
        self._buffer = buffer
        if meta_buffer is None:
            # Temporary v1 compatibility: the old scheduler constructs only a
            # tensor buffer and carries metadata in TransferReadyMsg frames.
            meta_buffer = TransferMetaBuffer(
                slot_count=16,
                slot_size=64 * 1024,
                role_name=getattr(buffer, "_role_name", "compat"),
            )
        self._meta_buffer = meta_buffer
        self._host_id = host_id
        self._send_retry_limit = max(0, int(send_retry_limit))
        self._lock = threading.Lock()

        self._engine.register_buffer(self._buffer.pool_data_ptr, self._buffer.pool_size)
        self._engine.register_buffer(
            self._meta_buffer.pool_data_ptr, self._meta_buffer.pool_size
        )

        self._staged: dict[str, StagedTransfer] = {}
        self._pending_receives: dict[str, PendingReceive] = {}

        logger.info(
            "DiffusionTransferManager initialized: session=%s, data_pool=%d bytes, meta_pool=%d bytes",
            self._engine.session_id,
            self._buffer.pool_size,
            self._meta_buffer.pool_size,
        )

    @property
    def session_id(self) -> str:
        return self._engine.session_id

    @property
    def pool_data_ptr(self) -> int:
        return self._buffer.pool_data_ptr

    @property
    def pool_size(self) -> int:
        return self._buffer.pool_size

    @property
    def meta_pool_ptr(self) -> int:
        return self._meta_buffer.pool_data_ptr

    @property
    def meta_pool_size(self) -> int:
        return self._meta_buffer.pool_size

    @property
    def data_shm_name(self) -> str | None:
        return self._buffer.shared_memory_name

    @property
    def meta_shm_name(self) -> str | None:
        return self._meta_buffer.shared_memory_name

    @property
    def host_id(self) -> str:
        return self._host_id

    @staticmethod
    def _tensor_fields_use_cuda(
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
    ) -> bool:
        return tensor_fields_need_device_event(tensor_fields)

    @staticmethod
    def _estimate_transfer_size(
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
    ) -> int:
        total_size = 0
        for value in tensor_fields.values():
            if value is None:
                continue
            tensors = value if isinstance(value, list) else [value]
            for tensor in tensors:
                if tensor is None:
                    continue
                total_size += tensor.nelement() * tensor.element_size()
                total_size = (total_size + 511) & ~511
        return total_size

    def stage_tensors(
        self,
        request_id: str,
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
        scalar_fields: dict | None = None,
        stream: torch.Stream | None = None,
    ) -> StagedTransfer | None:
        """Compatibility wrapper for tests/debugging."""
        staged, ready_event = self.stage_tensors_async(
            request_id, tensor_fields, scalar_fields, stream
        )
        if staged is None:
            return None
        if ready_event is not None and hasattr(ready_event, "synchronize"):
            ready_event.synchronize()
        self.mark_staged_ready(request_id)
        return staged

    def stage_tensors_async(
        self,
        request_id: str,
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
        scalar_fields: dict | None = None,
        stream: torch.Stream | None = None,
    ) -> tuple[StagedTransfer | None, object | None]:
        """Stage tensors and binary metadata without blocking the caller."""
        scalar_fields = scalar_fields or {}
        total_size = self._estimate_transfer_size(tensor_fields)
        data_slot = None
        manifest = {}

        if total_size > 0:
            data_slot = self._buffer.allocate(total_size, request_id)
            if data_slot is None:
                logger.warning(
                    "TransferManager: failed to allocate %d bytes for %s",
                    total_size,
                    request_id,
                )
                return None, None
            manifest = self._buffer.write_tensors_from_gpu(
                data_slot, tensor_fields, stream
            )

        meta_size = estimate_transfer_meta_bytes(manifest, scalar_fields)
        meta_slot = self._meta_buffer.allocate(request_id)
        if meta_slot is None:
            if data_slot is not None:
                self._buffer.free(data_slot)
            logger.warning(
                "TransferManager: failed to allocate meta slot for %s", request_id
            )
            return None, None
        if meta_size > meta_slot.size:
            if data_slot is not None:
                self._buffer.free(data_slot)
            self._meta_buffer.free(meta_slot)
            logger.warning(
                "TransferManager: metadata exceeds meta slot for %s (%d > %d)",
                request_id,
                meta_size,
                meta_slot.size,
            )
            return None, None

        try:
            meta_size = self._meta_buffer.write_metadata(
                meta_slot, manifest, scalar_fields
            )
        except Exception:
            if data_slot is not None:
                self._buffer.free(data_slot)
            self._meta_buffer.free(meta_slot)
            raise

        ready_event = record_device_event_if_needed(
            enabled=self._tensor_fields_use_cuda(tensor_fields),
            stream=stream,
        )

        staged = StagedTransfer(
            request_id=request_id,
            slot=data_slot,
            meta_slot=meta_slot,
            manifest=manifest,
            transfer_size=total_size,
            meta_size=meta_size,
            scalar_fields=scalar_fields,
            ready=ready_event is None,
        )
        with self._lock:
            self._staged[request_id] = staged

        if ready_event is None:
            self.mark_staged_ready(request_id)

        logger.debug(
            "TransferManager: staged_async %s (data=%d bytes, meta=%d bytes)",
            request_id,
            total_size,
            meta_size,
        )
        return staged, ready_event

    def mark_staged_ready(self, request_id: str) -> None:
        with self._lock:
            staged = self._staged.get(request_id)
            if staged is None:
                return
            staged.ready = True

    def _load_received_transfer(
        self,
        request_id: str,
        device: torch.device | str = current_platform.device_type,
        stream: torch.Stream | None = None,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], dict, object | None]:
        """Load data and v2 metadata from a pending receive slot."""
        with self._lock:
            pending = self._pending_receives.get(request_id)

        if pending is None:
            raise ValueError(
                f"TransferManager: no pending receive slot for {request_id}"
            )
        if pending.meta_slot is None:
            raise ValueError(
                f"TransferManager: no pending receive meta slot for {request_id}"
            )

        manifest, scalar_fields = self._meta_buffer.read_metadata(pending.meta_slot)
        tensors = self._buffer.read_tensors_from_manifest(
            pending.slot,
            manifest,
            device=device,
            stream=stream,
        )

        load_event = record_load_event(device, stream)
        return tensors, scalar_fields, load_event

    def load_tensors_async(
        self,
        request_id: str,
        manifest: dict | None = None,
        device: torch.device | str = current_platform.device_type,
        stream: torch.Stream | None = None,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], object | None]:
        if manifest is not None:
            # Temporary v1 compatibility: old TransferReadyMsg carries the
            # manifest/scalars in the control message and only RDMA-copies
            # tensor bytes. Load directly from that manifest instead of
            # reading the v2 metadata buffer.
            with self._lock:
                pending = self._pending_receives.get(request_id)
            if pending is None:
                raise ValueError(
                    f"TransferManager: no pending receive slot for {request_id}"
                )
            tensors = self._buffer.read_tensors_from_manifest(
                pending.slot,
                manifest,
                device=device,
                stream=stream,
            )
            load_event = record_load_event(device, stream)
            return tensors, load_event

        tensors, _scalar_fields, load_event = self._load_received_transfer(
            request_id, device=device, stream=stream
        )
        return tensors, load_event

    def load_transfer_async(
        self,
        request_id: str,
        device: torch.device | str = current_platform.device_type,
        stream: torch.Stream | None = None,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], dict, object | None]:
        return self._load_received_transfer(request_id, device=device, stream=stream)

    def push_to_peer(
        self,
        request_id: str,
        dest_session_id: str,
        dest_addr: int,
        transfer_size: int,
    ) -> bool:
        """Compatibility wrapper for tests/debugging."""
        with self._lock:
            staged = self._staged.get(request_id)
        if staged is None:
            logger.error("TransferManager: no staged transfer for %s", request_id)
            return False
        if staged.slot is None or transfer_size <= 0:
            return True
        src_addr = self._buffer.pool_data_ptr + staged.slot.offset
        ret = self._engine.transfer_sync(
            dest_session_id, src_addr, dest_addr, transfer_size
        )
        return ret == 0

    def free_staged(self, request_id: str) -> None:
        with self._lock:
            staged = self._staged.pop(request_id, None)
        if staged is None:
            return
        if staged.slot is not None:
            self._buffer.free(staged.slot)
        if staged.meta_slot is not None:
            self._meta_buffer.free(staged.meta_slot)

    def allocate_receive_slot(
        self, request_id: str, size: int, meta_size: int = 0
    ) -> PendingReceive | None:
        slot = self._buffer.allocate(size, request_id) if size > 0 else None
        meta_slot = self._meta_buffer.allocate(request_id)
        if meta_slot is None:
            if slot is not None:
                self._buffer.free(slot)
            logger.warning(
                "TransferManager: failed to allocate receive meta slot for %s",
                request_id,
            )
            return None
        if meta_size > meta_slot.size:
            if slot is not None:
                self._buffer.free(slot)
            self._meta_buffer.free(meta_slot)
            logger.warning(
                "TransferManager: receive meta slot too small for %s (%d > %d)",
                request_id,
                meta_size,
                meta_slot.size,
            )
            return None
        pending = PendingReceive(request_id=request_id, slot=slot, meta_slot=meta_slot)
        with self._lock:
            self._pending_receives[request_id] = pending
        return pending

    def load_tensors(
        self,
        request_id: str,
        manifest: dict | None = None,
        device: torch.device | str = current_platform.device_type,
        stream: torch.Stream | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Compatibility wrapper for tests/debugging."""
        if manifest is not None:
            tensors, load_event = self.load_tensors_async(
                request_id,
                manifest=manifest,
                device=device,
                stream=stream,
            )
            if load_event is not None and hasattr(load_event, "synchronize"):
                load_event.synchronize()
            return tensors

        tensors, _scalar_fields, load_event = self._load_received_transfer(
            request_id, device=device, stream=stream
        )
        if load_event is not None and hasattr(load_event, "synchronize"):
            load_event.synchronize()
        return tensors

    def register_prealloc_as_receive(
        self,
        request_id: str,
        slot: SlotHandle | None,
        meta_slot: SlotHandle | None = None,
        slot_id: int | None = None,
    ) -> PendingReceive:
        if meta_slot is None:
            meta_slot = self._meta_buffer.allocate(request_id)
            if meta_slot is None:
                raise RuntimeError(
                    f"TransferManager: failed to allocate receive meta slot for {request_id}"
                )
        pending = PendingReceive(
            request_id=request_id,
            slot=slot,
            meta_slot=meta_slot,
            slot_id=slot_id,
        )
        with self._lock:
            self._pending_receives[request_id] = pending
        return pending

    def get_pending_receive(self, request_id: str) -> PendingReceive | None:
        with self._lock:
            return self._pending_receives.get(request_id)

    def free_receive_slot(self, request_id: str) -> None:
        with self._lock:
            pending = self._pending_receives.pop(request_id, None)
        if pending is None:
            return
        if pending.slot is not None:
            self._buffer.free(pending.slot)
        if pending.meta_slot is not None:
            self._meta_buffer.free(pending.meta_slot)

    def get_receive_slot_addr(self, request_id: str) -> int | None:
        """Test/debug accessor; production transfer uses peer info messages."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None or pending.slot is None:
            return None
        return self._buffer.pool_data_ptr + pending.slot.offset

    def get_receive_slot_offset(self, request_id: str) -> int | None:
        """Test/debug accessor; production transfer uses peer info messages."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None or pending.slot is None:
            return None
        return pending.slot.offset

    def get_receive_meta_addr(self, request_id: str) -> int | None:
        """Test/debug accessor; production transfer uses peer info messages."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None or pending.meta_slot is None:
            return None
        return self._meta_buffer.pool_data_ptr + pending.meta_slot.offset

    def get_receive_meta_offset(self, request_id: str) -> int | None:
        """Test/debug accessor; production transfer uses peer info messages."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None or pending.meta_slot is None:
            return None
        return pending.meta_slot.offset

    def validate_receive_ready(
        self,
        request_id: str,
        *,
        dest_session_id: str | None = None,
        dest_slot_offset: int | None = None,
        dest_meta_slot_offset: int | None = None,
        data_size: int | None = None,
        meta_size: int | None = None,
    ) -> str | None:
        """Validate that a READY message targets the current pending receive slot."""
        if dest_session_id and dest_session_id != self.session_id:
            return (
                f"receiver session mismatch: ready={dest_session_id}, "
                f"current={self.session_id}"
            )

        with self._lock:
            pending = self._pending_receives.get(request_id)

        if pending is None:
            return f"no pending receive slot for {request_id}"

        if pending.slot is None:
            if data_size is not None and int(data_size) > 0:
                return f"data slot missing for non-empty transfer: ready={data_size}"
        else:
            if dest_slot_offset is not None and int(dest_slot_offset) != int(
                pending.slot.offset
            ):
                return (
                    f"data slot offset mismatch: ready={dest_slot_offset}, "
                    f"pending={pending.slot.offset}"
                )
            if data_size is not None and int(data_size) > int(pending.slot.size):
                return (
                    f"data size exceeds receive slot: ready={data_size}, "
                    f"slot={pending.slot.size}"
                )

        if dest_meta_slot_offset is not None and pending.meta_slot is not None:
            if int(dest_meta_slot_offset) != int(pending.meta_slot.offset):
                return (
                    f"metadata slot offset mismatch: ready={dest_meta_slot_offset}, "
                    f"pending={pending.meta_slot.offset}"
                )
        if meta_size is not None and pending.meta_slot is not None:
            if int(meta_size) > int(pending.meta_slot.size):
                return (
                    f"metadata size exceeds receive slot: ready={meta_size}, "
                    f"slot={pending.meta_slot.size}"
                )

        return None

    def get_staged_info(self, request_id: str) -> StagedTransfer | None:
        with self._lock:
            return self._staged.get(request_id)

    def has_active_transfers(self) -> bool:
        with self._lock:
            return bool(self._staged or self._pending_receives)

    def free_slots_count(self, typical_size: int = 64 * 1024 * 1024) -> int:
        return self._buffer.free_slots_count(typical_size)

    def cleanup(self) -> None:
        self._engine.deregister_buffer(self._buffer.pool_data_ptr)
        self._engine.deregister_buffer(self._meta_buffer.pool_data_ptr)
        self._buffer.cleanup()
        self._meta_buffer.cleanup()
        logger.info("DiffusionTransferManager cleaned up")
