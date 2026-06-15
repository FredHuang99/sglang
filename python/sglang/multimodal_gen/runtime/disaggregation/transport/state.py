# SPDX-License-Identifier: Apache-2.0
"""Transfer manager state records for diffusion disaggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
        SlotHandle,
    )


@dataclass
class StagedTransfer:
    request_id: str
    slot: SlotHandle | None
    meta_slot: SlotHandle | None
    manifest: dict
    transfer_size: int = 0
    meta_size: int = 0
    scalar_fields: dict = field(default_factory=dict)
    ready: bool = False


@dataclass
class PendingReceive:
    request_id: str
    slot: SlotHandle | None
    meta_slot: SlotHandle | None
    slot_id: int | None = None


@dataclass
class PendingPeerSend:
    request_id: str
    dest_session_id: str = ""
    dest_addr: int = 0
    transfer_size: int = 0
    meta_dest_addr: int = 0
    meta_transfer_size: int = 0
    receiver_role: str = ""
    receiver_instance: int = -1
    receiver_control_endpoint: str = ""
    receiver_host_id: str = ""
    receiver_supports_local_copy: bool = False
    dest_shm_name: str | None = None
    dest_shm_offset: int = 0
    meta_dest_shm_name: str | None = None
    meta_dest_shm_offset: int = 0
    prealloc_slot_id: int | None = None
    send_attempts: int = 0
    max_send_retries: int = 2
    last_error: str | None = None
    state: str = "waiting_stage"


@dataclass
class SendCompletion:
    request_id: str
    peer_info: PendingPeerSend | None
    staged: StagedTransfer | None
    success: bool
    error_msg: str | None = None
    retryable: bool = True
