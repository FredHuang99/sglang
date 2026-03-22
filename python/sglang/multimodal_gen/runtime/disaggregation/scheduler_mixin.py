# SPDX-License-Identifier: Apache-2.0
"""Mixin that adds disaggregated diffusion scheduling to the Scheduler.

Extracted from scheduler.py to keep the core scheduler lean.
All transfer, compute, and event-loop logic for disaggregated roles
(encoder / denoiser / decoder) lives here.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import queue
import socket
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import torch
import zmq

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferMetaBuffer,
    TransferTensorBuffer,
    estimate_transfer_meta_bytes,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
    send_tensors,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    create_transfer_engine,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
    DiffusionTransferManager,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TRANSFER_MAGIC,
    TransferAbortMsg,
    TransferAllocAcceptedMsg,
    TransferAllocRejectMsg,
    TransferFailedMsg,
    TransferDoneMsg,
    TransferMsgType,
    TransferPeerInfoMsg,
    TransferPushedMsg,
    TransferReadyMsg,
    TransferRegisterMsg,
    TransferStagedMsg,
    decode_transfer_msg,
    encode_transfer_msg,
    is_transfer_message,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler

logger = init_logger(__name__)


@dataclasses.dataclass
class _PendingOutboundTransfer:
    request_id: str
    staged: object
    stage_event: object | None
    msg_type: str
    staged_for_decoder: bool = False


@dataclasses.dataclass
class _PendingInboundTransfer:
    request_id: str
    role_name: str
    scalar_fields: dict[str, Any]
    tensors: dict[str, torch.Tensor | list[torch.Tensor]]
    load_event: object | None
    prealloc_slot_id: int | None = None

# ---------------------------------------------------------------------------
# Field extraction: split Req into tensors (transfer buffer) and scalars (JSON)
# ---------------------------------------------------------------------------

# Fields that should never be transferred (non-serializable, internal, or receiver rebuilds)
_EXCLUDE_FIELDS = frozenset(
    {
        "sampling_params",
        "generator",
        "modules",
        "metrics",
        "extra_step_kwargs",
        "extra",
        "condition_image",
        "vae_image",
        "pixel_values",
        "preprocessed_image",
        "image_embeds",
        "original_condition_image_size",
        "vae_image_sizes",
        "output",
        "audio",
        "audio_sample_rate",
        "trajectory_timesteps",
        "trajectory_latents",
        "trajectory_audio_latents",
        "timestep",
        "step_index",
        "prompt_template",
        "max_sequence_length",
    }
)

_SAMPLING_PARAMS_FIELDS = [
    "request_id",
    "guidance_scale",
    "guidance_scale_2",
    "height",
    "width",
    "num_frames",
    "fps",
    "num_inference_steps",
    "seed",
    "enable_sequence_shard",
]

_DIST_REPLICATE = "replicate_to_all_ranks"
_DIST_STAGE_SHARD = "scatter_stage_managed_sp"
_DIST_MODEL_MANAGED = "model_managed_full_input"

_COLLECTIVE_LIST_MARKER = "__sgl_collective_list__"
_COLLECTIVE_LIST_LEN = "__sgl_collective_list_len__"


def _is_tensor_like(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return True
    return False


def _to_json_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        converted = []
        for item in value:
            if isinstance(item, torch.Tensor):
                converted.append(item.tolist())
            else:
                converted.append(item)
        return converted
    return value


def _is_default(value, field_info) -> bool:
    if field_info.default is not dataclasses.MISSING:
        return value == field_info.default
    if field_info.default_factory is not dataclasses.MISSING:
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
    return False


def _extract_extra_fields(extra: dict, scalar_fields: dict) -> None:
    """Extract JSON-serializable entries from Req.extra into scalar_fields."""
    for key, value in extra.items():
        if key.startswith("_"):
            continue
        try:
            json.dumps(value)
            scalar_fields[f"_extra_{key}"] = value
        except (TypeError, ValueError, OverflowError):
            pass


def extract_transfer_fields(req) -> tuple[dict, dict]:
    """Extract all transferable fields from a Req, split into tensors and scalars."""
    tensor_fields = {}
    scalar_fields = {}

    for f in dataclasses.fields(req):
        if f.name in _EXCLUDE_FIELDS:
            continue

        value = getattr(req, f.name, None)
        if value is None:
            continue
        if _is_default(value, f):
            continue

        if _is_tensor_like(value):
            tensor_fields[f.name] = value
        else:
            try:
                scalar_fields[f.name] = _to_json_serializable(value)
            except (TypeError, ValueError):
                pass

    extra = getattr(req, "extra", None)
    if extra:
        _extract_extra_fields(extra, scalar_fields)

    sp = getattr(req, "sampling_params", None)
    if sp is not None:
        for name in _SAMPLING_PARAMS_FIELDS:
            if name in scalar_fields:
                continue
            value = getattr(sp, name, None)
            if value is not None:
                scalar_fields[name] = _to_json_serializable(value)

    return tensor_fields, scalar_fields


def estimate_transfer_bytes(
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


def estimate_transfer_manifest(
    tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
) -> dict[str, list[dict]]:
    manifest: dict[str, list[dict]] = {}
    byte_offset = 0
    for name, value in tensor_fields.items():
        if value is None:
            continue
        tensors = value if isinstance(value, list) else [value]
        entries = []
        for tensor in tensors:
            if tensor is None:
                continue
            nbytes = tensor.nelement() * tensor.element_size()
            entries.append(
                {
                    "offset": byte_offset,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).split(".")[-1],
                    "nbytes": nbytes,
                }
            )
            byte_offset += nbytes
            byte_offset = (byte_offset + 511) & ~511
        if entries:
            manifest[name] = entries
    return manifest


def _encode_collective_tensor_value(value):
    if isinstance(value, dict):
        return {key: _encode_collective_tensor_value(item) for key, item in value.items()}
    if isinstance(value, list):
        encoded = {
            _COLLECTIVE_LIST_MARKER: True,
            _COLLECTIVE_LIST_LEN: len(value),
        }
        for idx, item in enumerate(value):
            encoded[str(idx)] = _encode_collective_tensor_value(item)
        return encoded
    return value


def _decode_collective_tensor_value(value):
    if isinstance(value, dict) and value.get(_COLLECTIVE_LIST_MARKER):
        length = int(value.get(_COLLECTIVE_LIST_LEN, 0))
        decoded = [None] * length
        for idx in range(length):
            key = str(idx)
            if key in value:
                decoded[idx] = _decode_collective_tensor_value(value[key])
        return decoded
    if isinstance(value, dict):
        return {key: _decode_collective_tensor_value(item) for key, item in value.items()}
    return value


class SchedulerDisaggMixin:
    """Disaggregated diffusion scheduling: transfer, compute, event loops."""

    def _disagg_uses_cuda(self: Scheduler) -> bool:
        return self.server_args.resolved_role_device() == "cuda"

    def _disagg_local_device(self: Scheduler) -> str:
        if self._disagg_uses_cuda():
            return f"cuda:{self.worker.local_rank}"
        return "cpu"

    def _wait_transfer_event(self: Scheduler, event) -> None:
        if event is None:
            return
        if self._disagg_uses_cuda():
            torch.cuda.current_stream().wait_event(event)
            return
        if hasattr(event, "synchronize"):
            event.synchronize()

    def _transfer_event_ready(self: Scheduler, event) -> bool:
        if event is None:
            return True
        if hasattr(event, "query"):
            try:
                return bool(event.query())
            except Exception:
                return False
        if hasattr(event, "is_set"):
            return bool(event.is_set())
        return False

    def _broadcast_tensor_payload_to_all_ranks(
        self: Scheduler,
        tensor_payload: dict[str, torch.Tensor | list[torch.Tensor]] | None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if tensor_payload is None:
            encoded_payload = None
        else:
            encoded_payload = _encode_collective_tensor_value(tensor_payload)

        if torch.distributed.is_initialized():
            if self.server_args.sp_degree != 1:
                encoded_payload = self.worker.sp_group.broadcast_tensor_dict(
                    encoded_payload,
                    src=0,
                )
            if self.server_args.enable_cfg_parallel:
                encoded_payload = self.worker.cfg_group.broadcast_tensor_dict(
                    encoded_payload,
                    src=0,
                )
            if self.server_args.tp_size > 1:
                encoded_payload = self.worker.tp_group.broadcast_tensor_dict(
                    encoded_payload,
                    src=0,
                )

        if encoded_payload is None:
            return {}
        return _decode_collective_tensor_value(encoded_payload)

    def _decoder_uses_model_managed_sp(self: Scheduler) -> bool:
        if self._disagg_role != RoleType.DECODER or self.server_args.sp_degree <= 1:
            return False
        if self.worker is None or self.worker.pipeline is None:
            return False
        vae = self.worker.pipeline.get_module("vae")
        return bool(getattr(vae, "use_parallel_decode", False))

    def _classify_disagg_tensor_distribution(
        self: Scheduler,
        scalar_fields: dict[str, Any],
        tensors: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> dict[str, str]:
        req = self._build_disagg_req(scalar_fields, {})
        categories: dict[str, str] = {}

        for name in tensors:
            category = _DIST_REPLICATE
            if self.server_args.sp_degree > 1:
                if self._disagg_role == RoleType.DENOISER and name in {
                    "latents",
                    "image_latent",
                }:
                    if getattr(req, "enable_sequence_shard", False):
                        category = _DIST_MODEL_MANAGED
                    else:
                        category = _DIST_STAGE_SHARD
                elif (
                    self._disagg_role == RoleType.DECODER
                    and name == "latents"
                    and self._decoder_uses_model_managed_sp()
                ):
                    category = _DIST_MODEL_MANAGED
            categories[name] = category

        return categories

    def _apply_disagg_stage_managed_sp(
        self: Scheduler,
        req: Req,
        categories: dict[str, str],
    ) -> Req:
        if self._disagg_role != RoleType.DENOISER or self.server_args.sp_degree <= 1:
            return req

        pre_sharded_fields: set[str] = set()
        for field_name, category in categories.items():
            if category != _DIST_STAGE_SHARD:
                continue
            value = getattr(req, field_name, None)
            if not isinstance(value, torch.Tensor):
                continue
            sharded_value, did_shard = self.server_args.pipeline_config.shard_latents_for_sp(
                req,
                value,
            )
            setattr(req, field_name, sharded_value)
            if did_shard:
                pre_sharded_fields.add(field_name)

        if pre_sharded_fields:
            req._disagg_pre_sharded_fields = tuple(sorted(pre_sharded_fields))

        return req

    def _prepare_disagg_req_for_compute(self: Scheduler, req: Req) -> Req:
        if self._disagg_role == RoleType.DENOISER:
            scheduler_mod = self.worker.pipeline.get_module("scheduler")
            num_steps = getattr(req, "num_inference_steps", None)
            if scheduler_mod is not None and num_steps is not None:
                device = torch.device(self._disagg_local_device())
                extra_kwargs = {}
                mu = req.extra.get("mu") if hasattr(req, "extra") else None
                if mu is not None:
                    extra_kwargs["mu"] = mu
                scheduler_mod.set_timesteps(num_steps, device=device, **extra_kwargs)
        return req

    def _build_disagg_compute_req(
        self: Scheduler,
        scalar_fields: dict[str, Any],
        rank0_tensors: dict[str, torch.Tensor | list[torch.Tensor]] | None,
    ) -> Req:
        tensors = rank0_tensors or {}
        categories = self._classify_disagg_tensor_distribution(scalar_fields, tensors)
        if (
            self.server_args.sp_degree != 1
            or self.server_args.tp_size > 1
            or self.server_args.enable_cfg_parallel
        ):
            tensors = self._broadcast_tensor_payload_to_all_ranks(rank0_tensors)
            if tensors and not categories:
                categories = self._classify_disagg_tensor_distribution(
                    scalar_fields, tensors
                )

        req = self._build_disagg_req(scalar_fields, tensors)
        req = self._apply_disagg_stage_managed_sp(req, categories)
        return self._prepare_disagg_req_for_compute(req)

    def _role_has_inbound_transfer(self: Scheduler) -> bool:
        return self._disagg_role in (RoleType.DENOISER, RoleType.DECODER)

    def _role_has_outbound_transfer(self: Scheduler) -> bool:
        return self._disagg_role in (RoleType.ENCODER, RoleType.DENOISER)

    def _role_accepts_peer_info(self: Scheduler) -> bool:
        return self._disagg_role in (RoleType.ENCODER, RoleType.DENOISER)

    def _role_accepts_ready_failed(self: Scheduler) -> bool:
        return self._disagg_role in (RoleType.DENOISER, RoleType.DECODER)

    def _prune_aborted_requests(self: Scheduler) -> None:
        aborted = getattr(self, "_aborted_request_ids", None)
        if not aborted:
            return
        cutoff = time.monotonic() - getattr(self, "_aborted_request_ttl_s", 300.0)
        stale = [rid for rid, ts in aborted.items() if ts < cutoff]
        for rid in stale:
            aborted.pop(rid, None)

    def _remember_aborted_request(self: Scheduler, request_id: str) -> None:
        if not request_id:
            return
        self._prune_aborted_requests()
        self._aborted_request_ids[request_id] = time.monotonic()

    def _is_request_aborted(self: Scheduler, request_id: str) -> bool:
        self._prune_aborted_requests()
        return request_id in getattr(self, "_aborted_request_ids", {})

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_disagg_state(self: Scheduler, server_args, local_rank: int) -> None:
        """Initialize all disaggregation state, sockets, and transfer infrastructure."""
        from sglang.multimodal_gen.runtime.disaggregation.metrics import DisaggMetrics

        self._disagg_role = server_args.disagg_role
        self._disagg_timeout_s = float(getattr(server_args, "disagg_timeout", 600))
        self._disagg_metrics = None
        self._disagg_mode = getattr(server_args, "disagg_mode", False)
        self._pool_work_pull = None
        self._pool_result_push = None
        self._transfer_manager = None
        self._transfer_stream = None
        self._control_queue = None
        self._transferring_queue = None
        self._prefetch_queue = None
        self._swapping_queue = None
        self._compute_ready_queue = None
        self._swap_out_queue = None
        self._send_ready_queue = None
        self._preallocated_slots = {}
        self._aborted_request_ids = {}
        self._aborted_request_ttl_s = max(60.0, self._disagg_timeout_s * 2.0)

        if self._disagg_role != RoleType.MONOLITHIC:
            self._disagg_metrics = DisaggMetrics(role=self._disagg_role.value)
            if self._disagg_uses_cuda():
                device = torch.device(f"cuda:{local_rank}")
                self._transfer_stream = torch.cuda.Stream(device=device)
            self._init_disagg_sockets()
            if not (
                self._disagg_role == RoleType.ENCODER
                and getattr(server_args, "warmup", False)
            ):
                self._init_disagg_transfer_manager()

    def _init_disagg_sockets(self: Scheduler):
        """Initialize ZMQ sockets for disaggregated mode (DiffusionServer-mediated).

        Only rank 0 creates ZMQ sockets. Non-rank-0 processes participate
        via NCCL broadcast from rank 0 (see _disagg_recv_work).
        """
        if self.gpu_id != 0:
            logger.info(
                "Pool mode %s rank %d: no ZMQ sockets (non-rank-0)",
                self._disagg_role.value.upper(),
                self.gpu_id,
            )
            return

        sa = self.server_args

        # PULL: receive work from DiffusionServer
        self._pool_work_pull, _ = get_zmq_socket(
            self.context,
            zmq.PULL,
            sa.pool_work_endpoint,
            bind=True,
            max_bind_retries=5,
            same_port=True,
        )
        # PUSH: send results to DiffusionServer
        self._pool_result_push, _ = get_zmq_socket(
            self.context, zmq.PUSH, sa.pool_result_endpoint, bind=False
        )
        logger.info(
            "Disagg %s rank 0: work_pull=%s, result_push=%s",
            self._disagg_role.value.upper(),
            sa.pool_work_endpoint,
            sa.pool_result_endpoint,
        )

    def _init_disagg_transfer_manager(
        self: Scheduler,
        measured_transfer_bytes: int | None = None,
        measured_meta_bytes: int | None = None,
    ):
        """Initialize TransferManager for transfer mode (rank 0 only).

        Creates host-side data/meta buffers and a BaseTransferEngine, then wraps
        them in a DiffusionTransferManager.
        Also sends a transfer_register message to DiffusionServer.
        """
        if self.gpu_id != 0:
            return

        sa = self.server_args

        # Pool size: configurable, default 256 MiB
        pool_size = getattr(sa, "disagg_transfer_pool_size", 256 * 1024 * 1024)
        if measured_transfer_bytes is not None:
            redundancy = float(getattr(sa, "disagg_transfer_redundancy", 1.0))
            max_slots = max(1, int(getattr(sa, "disagg_max_slots_per_instance", 1)))
            computed_pool_size = max(
                measured_transfer_bytes,
                int(math.ceil(measured_transfer_bytes * max_slots * redundancy)),
            )
            if computed_pool_size > 0:
                pool_size = computed_pool_size

        # Create transfer engine
        hostname = getattr(sa, "disagg_p2p_hostname", "127.0.0.1")
        ib_device = getattr(sa, "disagg_ib_device", None)
        engine = create_transfer_engine(
            hostname=hostname,
            gpu_id=self.gpu_id,
            ib_device=ib_device,
        )

        buffer = TransferTensorBuffer(
            pool_size=pool_size,
            device="cpu",
            role_name=self._disagg_role.value,
        )
        max_slots = max(1, int(getattr(sa, "disagg_max_slots_per_instance", 1)))
        meta_slot_size = measured_meta_bytes or (64 * 1024)
        meta_buffer = TransferMetaBuffer(
            slot_count=max_slots,
            slot_size=meta_slot_size,
            role_name=self._disagg_role.value,
        )

        # Create transfer manager
        self._transfer_manager = DiffusionTransferManager(
            engine=engine,
            buffer=buffer,
            meta_buffer=meta_buffer,
            host_id=socket.gethostname(),
        )

        # Pre-allocate receive slots for receivers (denoiser/decoder)
        preallocated_slot_info = []
        if self._role_has_inbound_transfer():
            capacity = max_slots
            slot_size = max(1, int(math.ceil(pool_size / capacity)))
            for i in range(capacity):
                slot = buffer.allocate(slot_size, f"prealloc_data_{i}")
                meta_slot = meta_buffer.allocate(f"prealloc_meta_{i}")
                if slot is not None and meta_slot is not None:
                    self._preallocated_slots[i] = {
                        "data": slot,
                        "meta": meta_slot,
                    }
                    preallocated_slot_info.append(
                        {
                            "offset": slot.offset,
                            "size": slot.size,
                            "slot_id": i,
                            "addr": self._transfer_manager.pool_data_ptr + slot.offset,
                            "meta_offset": meta_slot.offset,
                            "meta_size": meta_slot.size,
                            "meta_addr": self._transfer_manager.meta_pool_ptr
                            + meta_slot.offset,
                        }
                    )
                else:
                    if slot is not None:
                        buffer.free(slot)
                    if meta_slot is not None:
                        meta_buffer.free(meta_slot)
            if preallocated_slot_info:
                logger.info(
                    "Transfer %s: pre-allocated %d receive slots",
                    self._disagg_role.value.upper(),
                    len(preallocated_slot_info),
                )

        # Register with DiffusionServer
        register_msg = TransferRegisterMsg(
            role=self._disagg_role.value,
            instance_id=getattr(sa, "disagg_instance_id", 0),
            session_id=self._transfer_manager.session_id,
            pool_ptr=self._transfer_manager.pool_data_ptr,
            pool_size=self._transfer_manager.pool_size,
            control_endpoint=(
                sa.pool_control_advertised_endpoint or sa.pool_control_endpoint or ""
            ),
            work_endpoint=sa.pool_work_endpoint or "",
            rank0_only=True,
            role_device=sa.resolved_role_device(),
            host_id=self._transfer_manager.host_id,
            supports_local_copy=bool(
                self._transfer_manager.data_shm_name
                and self._transfer_manager.meta_shm_name
            ),
            data_shm_name=self._transfer_manager.data_shm_name,
            meta_pool_ptr=self._transfer_manager.meta_pool_ptr,
            meta_pool_size=self._transfer_manager.meta_pool_size,
            meta_shm_name=self._transfer_manager.meta_shm_name,
            preallocated_slots=preallocated_slot_info,
        )
        self._pool_result_push.send_multipart(encode_transfer_msg(register_msg))
        logger.info(
            "Transfer %s: registered with DS (session=%s, pool=%d bytes, prealloc=%d)",
            self._disagg_role.value.upper(),
            self._transfer_manager.session_id,
            pool_size,
            len(preallocated_slot_info),
        )

        if self._role_has_inbound_transfer():
            self._control_queue = queue.Queue()
            self._transferring_queue = queue.Queue()
            self._prefetch_queue = self._transferring_queue
            self._swapping_queue = queue.Queue()
            self._compute_ready_queue = queue.Queue()

        if self._role_has_outbound_transfer():
            self._swap_out_queue = deque()
            self._send_ready_queue = deque()

        if sa.pool_control_endpoint:
            self._transfer_manager.start_background_loops(
                self.context,
                sa.pool_control_endpoint,
                on_ready=self._on_direct_transfer_ready,
                on_failed=self._on_direct_transfer_failed,
                on_abort=self._on_direct_transfer_abort,
                on_send_completion=self._on_direct_send_completion,
                send_concurrency=max(
                    1, int(getattr(sa, "disagg_max_slots_per_instance", 1))
                ),
                start_send_loop=self._role_has_outbound_transfer(),
            )

    def _run_disagg_startup_warmup(
        self: Scheduler, warmup_reqs: list[Req]
    ) -> None:
        """Run startup warmup before creating the encoder transfer manager."""
        if self._transfer_manager is not None:
            return
        measured_transfer_bytes = None
        measured_meta_bytes = None
        if self._disagg_role == RoleType.ENCODER and warmup_reqs:
            for req in warmup_reqs:
                req_result = self.worker.execute_forward([req], return_req=True)
                if isinstance(req_result, Req):
                    tensor_fields, scalar_fields = extract_transfer_fields(req_result)
                    measured_transfer_bytes = estimate_transfer_bytes(tensor_fields)
                    measured_meta_bytes = estimate_transfer_meta_bytes(
                        estimate_transfer_manifest(tensor_fields), scalar_fields
                    )
        self._init_disagg_transfer_manager(
            measured_transfer_bytes,
            measured_meta_bytes,
        )

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _on_direct_transfer_ready(self: Scheduler, msg: dict) -> None:
        request_id = msg.get("request_id", "")
        if request_id and self._is_request_aborted(request_id):
            return
        if not self._role_accepts_ready_failed():
            logger.debug(
                "Transfer %s: ignoring direct READY for role without inbound path",
                self._disagg_role.value.upper(),
            )
            return
        if self._transferring_queue is not None:
            self._transferring_queue.put(msg)
            return
        self._handle_transfer_ready(msg)

    def _on_direct_transfer_failed(self: Scheduler, msg: dict) -> None:
        request_id = msg.get("request_id", "")
        if request_id and self._is_request_aborted(request_id):
            return
        if not self._role_accepts_ready_failed():
            logger.debug(
                "Transfer %s: ignoring direct FAILED for role without inbound path",
                self._disagg_role.value.upper(),
            )
            return
        if self._control_queue is not None:
            self._control_queue.put(msg)
            return
        self._handle_transfer_failed(msg)

    def _on_direct_transfer_abort(self: Scheduler, msg: dict) -> None:
        request_id = msg.get("request_id", "")
        if not request_id:
            return
        if self._control_queue is not None:
            self._control_queue.put(msg)
            return
        self._handle_transfer_abort(msg)

    def _on_direct_send_completion(
        self: Scheduler,
        request_id: str,
        peer_info,
        staged,
        success: bool,
        error_msg: str | None,
    ) -> None:
        if not self._role_has_outbound_transfer():
            return
        if not success or staged is None:
            failed_msg = TransferFailedMsg(
                request_id=request_id,
                error=error_msg or "transfer_sync failed",
                receiver_role=getattr(peer_info, "receiver_role", ""),
                receiver_instance=getattr(peer_info, "receiver_instance", -1),
                prealloc_slot_id=getattr(peer_info, "prealloc_slot_id", None),
            )
            try:
                self._transfer_manager.send_direct_message(
                    getattr(peer_info, "receiver_control_endpoint"),
                    failed_msg,
                )
            except Exception:
                logger.exception(
                    "Transfer %s: failed to notify downstream failure for %s",
                    self._disagg_role.value.upper(),
                    request_id,
                )
            self._pool_result_push.send_multipart(
                encode_transfer_msg(
                    TransferPushedMsg(
                        request_id=request_id,
                        success=False,
                        error=error_msg or "transfer_sync failed",
                    )
                )
            )
            return

        ready_msg = TransferReadyMsg(
            request_id=request_id,
            prealloc_slot_id=getattr(peer_info, "prealloc_slot_id", None),
        )
        try:
            self._transfer_manager.send_direct_message(
                getattr(peer_info, "receiver_control_endpoint"),
                ready_msg,
            )
        except Exception:
            logger.exception(
                "Transfer %s: failed to notify downstream ready for %s",
                self._disagg_role.value.upper(),
                request_id,
            )
            self._pool_result_push.send_multipart(
                encode_transfer_msg(
                    TransferPushedMsg(
                        request_id=request_id,
                        success=False,
                        error="failed to notify downstream ready",
                    )
                )
            )
            return

        self._pool_result_push.send_multipart(
            encode_transfer_msg(
                TransferPushedMsg(request_id=request_id, success=True, error=None)
            )
        )

    def _prefetch_transfer_ready(self: Scheduler, msg: dict) -> _PendingInboundTransfer:
        """Start receiver-side H2D/load and stash loaded tensors for later distribution."""
        request_id = msg["request_id"]
        role_name = self._disagg_role.value.upper()

        if self._disagg_metrics:
            self._disagg_metrics.record_request_start(request_id)

        prealloc_slot_id = msg.get("prealloc_slot_id")
        local_device = self._disagg_local_device()
        tensors, scalar_fields, load_event = self._transfer_manager.load_transfer_async(
            request_id,
            device=local_device,
            stream=self._transfer_stream,
        )

        return _PendingInboundTransfer(
            request_id=request_id,
            role_name=role_name,
            scalar_fields=scalar_fields,
            tensors=tensors,
            load_event=load_event,
            prealloc_slot_id=prealloc_slot_id,
        )

    def _release_pending_receive(
        self: Scheduler, request_id: str, prealloc_slot_id: int | None
    ) -> None:
        if self._transfer_manager is None:
            return
        if prealloc_slot_id is not None:
            with self._transfer_manager._lock:
                self._transfer_manager._pending_receives.pop(request_id, None)
            return
        self._transfer_manager.free_receive_slot(request_id)

    def _drain_transfer_control_socket(self: Scheduler) -> int:
        if self.gpu_id != 0 or self._pool_work_pull is None or self._control_queue is None:
            return 0

        drained = 0
        while True:
            try:
                frames = [bytes(f) for f in self._pool_work_pull.recv_multipart(zmq.NOBLOCK)]
            except zmq.Again:
                break

            if not self._is_transfer_frames(frames):
                logger.warning(
                    "Transfer %s: ignoring non-transfer frames on receiver work socket",
                    self._disagg_role.value.upper(),
                )
                continue

            try:
                msg = decode_transfer_msg(frames)
            except Exception:
                logger.exception(
                    "Transfer %s: failed to decode receiver control message",
                    self._disagg_role.value.upper(),
                )
                continue
            self._control_queue.put(msg)
            drained += 1

        return drained

    def _process_transfer_control_queue(self: Scheduler) -> bool:
        if self._control_queue is None:
            return False

        handled = False
        while True:
            try:
                msg = self._control_queue.get_nowait()
            except queue.Empty:
                break

            handled = True
            msg_type = msg.get("msg_type", "")
            if msg_type == TransferMsgType.ALLOC:
                self._handle_transfer_alloc(msg)
            elif msg_type == TransferMsgType.ABORT:
                self._handle_transfer_abort(msg)
            elif msg_type == TransferMsgType.FAILED:
                self._handle_transfer_failed(msg)
            elif (
                msg_type == TransferMsgType.READY
                and self._transferring_queue is not None
            ):
                self._transferring_queue.put(msg)
            else:
                logger.warning(
                    "Transfer %s: unexpected control msg_type=%s",
                    self._disagg_role.value.upper(),
                    msg_type,
                )

        return handled

    def _process_prefetch_queue_once(self: Scheduler) -> bool:
        if self._transferring_queue is None or self._swapping_queue is None:
            return False

        try:
            msg = self._transferring_queue.get_nowait()
        except queue.Empty:
            return False

        if self._is_request_aborted(msg.get("request_id", "")):
            return True

        item = self._prefetch_transfer_ready(msg)
        self._swapping_queue.put(item)
        return True

    def _process_swapping_queue_once(self: Scheduler) -> bool:
        if self._swapping_queue is None or self._compute_ready_queue is None:
            return False

        try:
            item = self._swapping_queue.get_nowait()
        except queue.Empty:
            return False

        if self._is_request_aborted(item.request_id):
            return True

        load_event = item.load_event
        if not self._transfer_event_ready(load_event):
            self._swapping_queue.put(item)
            return False

        self._compute_ready_queue.put(item)
        return True

    def _run_prefetched_compute_item(
        self: Scheduler,
        item: _PendingInboundTransfer,
        *,
        is_multi_rank: bool,
    ) -> None:
        if self._is_request_aborted(item.request_id):
            return
        self._release_pending_receive(item.request_id, item.prealloc_slot_id)

        if is_multi_rank:
            self._broadcast_to_all_ranks(("compute", item.scalar_fields))

        req = self._build_disagg_compute_req(item.scalar_fields, item.tensors)
        if self._disagg_role == RoleType.DENOISER:
            self._disagg_denoiser_compute(req, item.request_id, item.role_name)
        elif self._disagg_role == RoleType.DECODER:
            self._disagg_decoder_compute(req, item.request_id, item.role_name)

    def _process_compute_ready_queue_once(self: Scheduler, is_multi_rank: bool) -> bool:
        if self._compute_ready_queue is None:
            return False

        try:
            item = self._compute_ready_queue.get_nowait()
        except queue.Empty:
            return False

        self._run_prefetched_compute_item(item, is_multi_rank=is_multi_rank)
        return True

    def _enqueue_outbound_transfer(
        self: Scheduler,
        request_id: str,
        staged,
        stage_event,
        *,
        msg_type: str,
        staged_for_decoder: bool = False,
    ) -> None:
        if self._swap_out_queue is None:
            return
        self._swap_out_queue.append(
            _PendingOutboundTransfer(
                request_id=request_id,
                staged=staged,
                stage_event=stage_event,
                msg_type=msg_type,
                staged_for_decoder=staged_for_decoder,
            )
        )

    def _process_swap_out_queue_once(self: Scheduler) -> bool:
        if not self._swap_out_queue or self._send_ready_queue is None:
            return False

        item = self._swap_out_queue.popleft()
        if self._is_request_aborted(item.request_id):
            return True
        if not self._transfer_event_ready(item.stage_event):
            self._swap_out_queue.append(item)
            return False

        self._send_ready_queue.append(item)
        return True

    def _process_send_ready_queue_once(self: Scheduler) -> bool:
        if not self._send_ready_queue:
            return False

        item = self._send_ready_queue.popleft()
        if self._is_request_aborted(item.request_id):
            return True
        staged = item.staged or self._transfer_manager.get_staged_info(item.request_id)
        if staged is None:
            logger.error(
                "Transfer %s: missing staged payload for %s before notifying server",
                self._disagg_role.value.upper(),
                item.request_id,
            )
            return False

        self._transfer_manager.mark_staged_ready(item.request_id)

        if item.msg_type == TransferMsgType.STAGED:
            msg = TransferStagedMsg(
                request_id=item.request_id,
                data_size=staged.transfer_size,
                meta_size=staged.meta_size,
                session_id=self._transfer_manager.session_id,
                pool_ptr=self._transfer_manager.pool_data_ptr,
                slot_offset=staged.slot.offset if staged.slot else 0,
                meta_pool_ptr=self._transfer_manager.meta_pool_ptr,
                meta_slot_offset=staged.meta_slot.offset if staged.meta_slot else 0,
            )
        else:
            msg = TransferDoneMsg(
                request_id=item.request_id,
                staged_for_decoder=item.staged_for_decoder,
                session_id=self._transfer_manager.session_id,
                pool_ptr=self._transfer_manager.pool_data_ptr,
                slot_offset=staged.slot.offset if staged.slot else 0,
                meta_pool_ptr=self._transfer_manager.meta_pool_ptr,
                meta_slot_offset=staged.meta_slot.offset if staged.meta_slot else 0,
                data_size=staged.transfer_size,
                meta_size=staged.meta_size,
            )

        self._pool_result_push.send_multipart(encode_transfer_msg(msg))
        return True

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def _broadcast_to_all_ranks(self: Scheduler, data):
        """Broadcast *data* from rank 0 to all other ranks.

        Rank 0 passes the real payload; non-rank-0 passes ``None``.
        Broadcasts through all applicable groups (SP, CFG, TP).
        """
        sa = self.server_args

        if sa.sp_degree != 1:
            data = broadcast_pyobj(
                data,
                self.worker.sp_group.rank,
                self.worker.sp_cpu_group,
                src=self.worker.sp_group.ranks[0],
            )

        if sa.enable_cfg_parallel:
            data = broadcast_pyobj(
                data,
                self.worker.cfg_group.rank,
                self.worker.cfg_cpu_group,
                src=self.worker.cfg_group.ranks[0],
            )

        if sa.tp_size > 1:
            data = broadcast_pyobj(
                data,
                self.worker.tp_group.rank,
                self.worker.tp_cpu_group,
                src=self.worker.tp_group.ranks[0],
            )

        return data

    # ------------------------------------------------------------------
    # Event loops
    # ------------------------------------------------------------------

    def _disagg_recv_work(self: Scheduler) -> list[bytes] | None:
        """Receive work frames in pool mode, with multi-rank broadcast.

        Rank 0: recv from ZMQ PULL socket, broadcast to other ranks.
        Non-rank-0: receive via NCCL broadcast from rank 0.

        Returns list of bytes frames, or None on shutdown.
        """
        if self.gpu_id == 0:
            raw_frames = self._pool_work_pull.recv_multipart()
            frames = [bytes(f) for f in raw_frames]
        else:
            frames = None

        return self._broadcast_to_all_ranks(frames)

    def _try_recv_work_noblock(self: Scheduler) -> list[bytes] | None:
        if self.gpu_id != 0 or self._pool_work_pull is None:
            return None
        try:
            raw_frames = self._pool_work_pull.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return None
        return [bytes(f) for f in raw_frames]

    def _disagg_prefetch_event_loop(self: Scheduler, role_name: str) -> None:
        """Event loop for receiver roles that need H2D prefetch + compute overlap."""
        is_multi_rank = (
            self.server_args.sp_degree != 1
            or self.server_args.tp_size > 1
            or self.server_args.enable_cfg_parallel
        )

        while self._running:
            try:
                handled_work = False
                handled_work |= self._drain_transfer_control_socket() > 0
                handled_work |= self._process_transfer_control_queue()
                handled_work |= self._process_prefetch_queue_once()
                handled_work |= self._process_swapping_queue_once()
                computed = self._process_compute_ready_queue_once(is_multi_rank)
                handled_work |= computed
                handled_work |= self._process_swap_out_queue_once()
                handled_work |= self._process_send_ready_queue_once()

                if is_multi_rank and not computed:
                    self._broadcast_to_all_ranks(("skip",))

                if not handled_work:
                    time.sleep(0.001)

                self._consecutive_error_count = 0

            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    "Pool %s rank %d prefetch loop: error (attempt %d/%d): %s",
                    role_name,
                    self.gpu_id,
                    self._consecutive_error_count,
                    self._max_consecutive_errors,
                    e,
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    raise RuntimeError(
                        f"Pool {role_name} rank {self.gpu_id} terminated after "
                        f"{self._max_consecutive_errors} consecutive errors: {e}"
                    ) from e

        # Shutdown: notify non-rank-0 to exit
        if is_multi_rank:
            self._broadcast_to_all_ranks(None)
        self._cleanup_disagg()

    def _disagg_encoder_non_rank0_event_loop(self: Scheduler) -> None:
        role_name = self._disagg_role.value.upper()
        logger.info(
            "Pool %s rank %d: entering non-rank-0 encoder loop",
            role_name,
            self.gpu_id,
        )

        while True:
            try:
                msg = self._broadcast_to_all_ranks(None)

                if msg is None:
                    break

                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "encoder_work":
                    self._disagg_encoder_step(send_tensors, frames=msg[1])

            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    "Pool %s rank %d encoder follower loop: error (attempt %d/%d): %s",
                    role_name,
                    self.gpu_id,
                    self._consecutive_error_count,
                    self._max_consecutive_errors,
                    e,
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    raise RuntimeError(
                        f"Pool {role_name} rank {self.gpu_id} terminated after "
                        f"{self._max_consecutive_errors} consecutive errors: {e}"
                    ) from e

        self._cleanup_disagg()

    def _disagg_encoder_rank0_event_loop(self: Scheduler) -> None:
        role_name = self._disagg_role.value.upper()
        is_multi_rank = (
            self.server_args.sp_degree != 1
            or self.server_args.tp_size > 1
            or self.server_args.enable_cfg_parallel
        )

        while self._running:
            try:
                handled_work = False
                frames = self._try_recv_work_noblock()
                if frames is not None:
                    handled_work = True
                    if is_multi_rank:
                        self._broadcast_to_all_ranks(("encoder_work", frames))
                    self._disagg_encoder_step(send_tensors, frames=frames)

                handled_work |= self._process_swap_out_queue_once()
                handled_work |= self._process_send_ready_queue_once()

                if not handled_work:
                    time.sleep(0.001)

                self._consecutive_error_count = 0

            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    "Pool %s rank %d encoder loop: error (attempt %d/%d): %s",
                    role_name,
                    self.gpu_id,
                    self._consecutive_error_count,
                    self._max_consecutive_errors,
                    e,
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    raise RuntimeError(
                        f"Pool {role_name} rank {self.gpu_id} terminated after "
                        f"{self._max_consecutive_errors} consecutive errors: {e}"
                    ) from e

        if is_multi_rank:
            self._broadcast_to_all_ranks(None)
        self._cleanup_disagg()

    def _disagg_non_rank0_event_loop(self: Scheduler) -> None:
        """Event loop for non-rank-0 receivers in multi-rank prefetch mode.

        Blocks on broadcast from rank 0:
          - ("compute", scalar_fields): build minimal Req 閳?execute_forward
          - ("skip",): continue (rank 0 handled a control msg or timed out)
          - None: shutdown, exit loop
        """
        role_name = self._disagg_role.value.upper()
        logger.info(
            "Pool %s rank %d: entering non-rank-0 prefetch loop",
            role_name,
            self.gpu_id,
        )

        while True:
            try:
                msg = self._broadcast_to_all_ranks(None)

                if msg is None:
                    # Shutdown signal
                    break

                if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] == "compute":
                    scalar_fields = msg[1]
                    self._disagg_compute_non_rank0(scalar_fields)
                # else: ("skip",) 閳?continue

            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    "Pool %s rank %d non-rank-0 loop: error (attempt %d/%d): %s",
                    role_name,
                    self.gpu_id,
                    self._consecutive_error_count,
                    self._max_consecutive_errors,
                    e,
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    raise RuntimeError(
                        f"Pool {role_name} rank {self.gpu_id} terminated after "
                        f"{self._max_consecutive_errors} consecutive errors: {e}"
                    ) from e

        self._cleanup_disagg()

    def _disagg_event_loop(self: Scheduler) -> None:
        """Event loop for all roles in pool mode (DiffusionServer-mediated)."""
        role_name = self._disagg_role.value.upper()
        is_rank0 = self.gpu_id == 0
        is_multi_rank = (
            self.server_args.sp_degree != 1
            or self.server_args.tp_size > 1
            or self.server_args.enable_cfg_parallel
        )
        use_prefetch = self._compute_ready_queue is not None
        logger.info(
            "Pool mode %s rank %d event loop started " "(multi_rank=%s, prefetch=%s)",
            role_name,
            self.gpu_id,
            is_multi_rank,
            use_prefetch,
        )

        if self._disagg_role == RoleType.ENCODER:
            if not is_rank0 and is_multi_rank:
                self._disagg_encoder_non_rank0_event_loop()
            else:
                self._disagg_encoder_rank0_event_loop()
            return

        if use_prefetch:
            self._disagg_prefetch_event_loop(role_name)
            return

        if (
            not is_rank0
            and is_multi_rank
            and self._disagg_role in (RoleType.DENOISER, RoleType.DECODER)
        ):
            self._disagg_non_rank0_event_loop()
            return

        while self._running:
            time.sleep(0.01)

        self._cleanup_disagg()

    def _cleanup_disagg(self: Scheduler):
        """Clean up all pool mode resources (sockets, threads, transfer manager)."""
        if self._transfer_manager is not None:
            self._transfer_manager.cleanup()
        if self._pool_work_pull is not None:
            self._pool_work_pull.close()
        if self._pool_result_push is not None:
            self._pool_result_push.close()

    # ------------------------------------------------------------------
    # Transfer message handling
    # ------------------------------------------------------------------

    @staticmethod
    def _is_transfer_frames(frames: list) -> bool:
        """Check if ZMQ multipart frames carry a transfer control message."""
        return is_transfer_message(frames)

    def _handle_transfer_msg(self: Scheduler, frames: list) -> None:
        """Dispatch a transfer control message to the appropriate handler (rank 0)."""
        msg = decode_transfer_msg(frames)
        msg_type = msg.get("msg_type", "")
        request_id = msg.get("request_id", "")

        logger.debug(
            "Transfer %s: received %s for %s",
            self._disagg_role.value.upper(),
            msg_type,
            request_id,
        )

        if msg_type == TransferMsgType.ALLOC:
            self._handle_transfer_alloc(msg)
        elif msg_type == TransferMsgType.ABORT:
            self._handle_transfer_abort(msg)
        elif msg_type == TransferMsgType.READY:
            if self._role_accepts_ready_failed():
                self._handle_transfer_ready(msg)
        elif msg_type == TransferMsgType.FAILED:
            if self._role_accepts_ready_failed():
                self._handle_transfer_failed(msg)
        else:
            logger.warning(
                "Transfer %s: unknown message type %s",
                self._disagg_role.value.upper(),
                msg_type,
            )

    def _handle_transfer_non_rank0(self: Scheduler, frames: list) -> None:
        """Transfer control is rank0-only; follower ranks only react to broadcasts."""
        return

    def _handle_transfer_alloc(self: Scheduler, msg: dict) -> None:
        """Handle transfer_alloc by reserving a paired data/meta receive slot."""
        request_id = msg["request_id"]
        if self._is_request_aborted(request_id):
            return
        data_size = msg.get("data_size", 0)
        meta_size = msg.get("meta_size", 0)
        source_host_id = msg.get("source_host_id", "")
        preallocated_slot = msg.get("preallocated_slot")

        def release_pending() -> None:
            if self._transfer_manager is not None:
                if preallocated_slot is not None:
                    with self._transfer_manager._lock:
                        self._transfer_manager._pending_receives.pop(request_id, None)
                else:
                    self._transfer_manager.free_receive_slot(request_id)

        def reject_alloc(reason: str, *, retryable: bool) -> None:
            release_pending()
            self._pool_result_push.send_multipart(
                encode_transfer_msg(
                    TransferAllocRejectMsg(
                        request_id=request_id,
                        receiver_role=self._disagg_role.value,
                        receiver_instance=getattr(self.server_args, "disagg_instance_id", 0),
                        retryable=retryable,
                        reason=reason,
                        prealloc_slot_id=(
                            preallocated_slot.get("slot_id")
                            if preallocated_slot is not None
                            else None
                        ),
                    )
                )
            )

        pending = None
        if preallocated_slot is not None:
            slot_id = preallocated_slot.get("slot_id")
            prealloc_info = self._preallocated_slots.get(slot_id)
            if prealloc_info is None:
                logger.error(
                    "Transfer %s: preallocated slot %s not found for %s",
                    self._disagg_role.value.upper(),
                    slot_id,
                    request_id,
                )
                reject_alloc("receiver preallocated slot not found", retryable=True)
                return
            slot = prealloc_info["data"]
            meta_slot = prealloc_info["meta"]
            if slot.size < data_size:
                logger.error(
                    "Transfer %s: preallocated slot %s too small for %s (%d < %d)",
                    self._disagg_role.value.upper(),
                    slot_id,
                    request_id,
                    slot.size,
                    data_size,
                )
                reject_alloc("receiver preallocated slot too small", retryable=True)
                return
            if meta_slot.size < meta_size:
                logger.error(
                    "Transfer %s: preallocated meta slot %s too small for %s (%d < %d)",
                    self._disagg_role.value.upper(),
                    slot_id,
                    request_id,
                    meta_slot.size,
                    meta_size,
                )
                reject_alloc("receiver preallocated meta slot too small", retryable=True)
                return
            pending = self._transfer_manager.register_prealloc_as_receive(
                request_id,
                slot,
                meta_slot,
                slot_id=slot_id,
            )
        else:
            pending = self._transfer_manager.allocate_receive_slot(
                request_id,
                data_size,
                meta_size,
            )

        if pending is None:
            logger.error(
                "Transfer %s: failed to allocate receive slot for %s (data=%d bytes, meta=%d bytes)",
                self._disagg_role.value.upper(),
                request_id,
                data_size,
                meta_size,
            )
            reject_alloc("receiver failed to allocate slot", retryable=True)
            return

        source_control_endpoint = msg.get("source_control_endpoint", "")
        if not source_control_endpoint:
            logger.error(
                "Transfer %s: missing source control endpoint for %s",
                self._disagg_role.value.upper(),
                request_id,
            )
            reject_alloc("missing source control endpoint", retryable=False)
            return

        same_host = source_host_id == self._transfer_manager.host_id
        peer_msg = TransferPeerInfoMsg(
            request_id=request_id,
            dest_session_id=self._transfer_manager.session_id,
            dest_addr=(
                self._transfer_manager.pool_data_ptr + pending.slot.offset
                if pending.slot is not None
                else 0
            ),
            transfer_size=data_size,
            meta_dest_addr=self._transfer_manager.meta_pool_ptr + pending.meta_slot.offset,
            meta_transfer_size=meta_size,
            receiver_role=self._disagg_role.value,
            receiver_instance=getattr(self.server_args, "disagg_instance_id", 0),
            receiver_control_endpoint=(
                self.server_args.pool_control_advertised_endpoint
                or self.server_args.pool_control_endpoint
                or ""
            ),
            receiver_host_id=self._transfer_manager.host_id,
            receiver_supports_local_copy=bool(
                self._transfer_manager.data_shm_name
                and self._transfer_manager.meta_shm_name
            ),
            dest_shm_name=self._transfer_manager.data_shm_name if same_host else None,
            dest_shm_offset=pending.slot.offset if pending.slot is not None else 0,
            meta_dest_shm_name=self._transfer_manager.meta_shm_name if same_host else None,
            meta_dest_shm_offset=pending.meta_slot.offset,
            prealloc_slot_id=preallocated_slot.get("slot_id") if preallocated_slot is not None else None,
        )
        try:
            self._transfer_manager.send_direct_message(
                source_control_endpoint,
                peer_msg,
            )
        except Exception:
            logger.exception(
                "Transfer %s: failed to send peer info to upstream for %s",
                self._disagg_role.value.upper(),
                request_id,
            )
            reject_alloc("failed to send peer info to upstream", retryable=False)
            return

        self._pool_result_push.send_multipart(
            encode_transfer_msg(
                TransferAllocAcceptedMsg(
                    request_id=request_id,
                    receiver_role=self._disagg_role.value,
                    receiver_instance=getattr(self.server_args, "disagg_instance_id", 0),
                    prealloc_slot_id=(
                        preallocated_slot.get("slot_id")
                        if preallocated_slot is not None
                        else None
                    ),
                )
            )
        )

        logger.debug(
            "Transfer %s: allocated receive slot for %s (data_offset=%d, data_size=%d, meta_offset=%d, meta_size=%d)",
            self._disagg_role.value.upper(),
            request_id,
            pending.slot.offset if pending.slot is not None else 0,
            pending.slot.size if pending.slot is not None else 0,
            pending.meta_slot.offset,
            pending.meta_slot.size,
        )

    def _handle_transfer_failed(self: Scheduler, msg: dict) -> None:
        request_id = msg["request_id"]
        error = msg.get("error", "transfer failed")
        prealloc_slot_id = msg.get("prealloc_slot_id")
        self._remember_aborted_request(request_id)
        self._release_pending_receive(request_id, prealloc_slot_id)

        logger.error(
            "Transfer %s: upstream transfer failed for %s: %s",
            self._disagg_role.value.upper(),
            request_id,
            error,
        )

    def _handle_transfer_abort(self: Scheduler, msg: dict) -> None:
        request_id = msg.get("request_id", "")
        if not request_id:
            return
        self._remember_aborted_request(request_id)
        if self._transfer_manager is not None:
            self._transfer_manager.abort_request(request_id)
        logger.warning(
            "Transfer %s: aborted %s (%s)",
            self._disagg_role.value.upper(),
            request_id,
            msg.get("reason", "server abort"),
        )

    def _handle_transfer_ready(self: Scheduler, msg: dict) -> None:
        if self._is_request_aborted(msg.get("request_id", "")):
            return
        if self._transferring_queue is not None:
            self._transferring_queue.put(msg)
            return

        item = self._prefetch_transfer_ready(msg)
        self._wait_transfer_event(item.load_event)
        self._run_prefetched_compute_item(item, is_multi_rank=False)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _disagg_compute_non_rank0(self: Scheduler, scalar_fields: dict) -> None:
        """Follower-rank compute: receive distributed tensors, rebuild Req, then compute."""
        req = self._build_disagg_compute_req(scalar_fields, None)

        if self._disagg_role == RoleType.DENOISER:
            self.worker.execute_forward([req], return_req=True)

        elif self._disagg_role == RoleType.DECODER:
            req.save_output = False
            req.return_file_paths_only = False
            self.worker.execute_forward([req])

    def _build_disagg_req(self: Scheduler, scalar_fields: dict, tensors: dict) -> Req:
        """Reconstruct a Req from transfer scalar fields and loaded GPU tensors.

        Initializes all dataclass field defaults first, then overlays
        scalar and tensor fields from the transfer message.
        """
        scalar_fields = dict(scalar_fields)
        req = object.__new__(Req)
        # Initialize all dataclass fields with their defaults
        for f in dataclasses.fields(Req):
            if f.default is not dataclasses.MISSING:
                object.__setattr__(req, f.name, f.default)
            elif f.default_factory is not dataclasses.MISSING:
                object.__setattr__(req, f.name, f.default_factory())
        # Ensure sampling_params is not None so __getattr__ delegation works
        object.__setattr__(req, "sampling_params", SamplingParams())
        # Restore _extra_* prefixed fields into req.extra dict
        extra_keys = [k for k in scalar_fields if k.startswith("_extra_")]
        for key in extra_keys:
            req.extra[key[len("_extra_") :]] = scalar_fields.pop(key)
        # Overlay scalar fields from the transfer message
        req.__dict__.update(scalar_fields)
        # Set tensor fields
        for key, value in tensors.items():
            setattr(req, key, value)
        # Recreate torch.Generator from seed (not serializable over transfer)
        seed = scalar_fields.get("seed")
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            req.generator = gen
        return req

    def _disagg_denoiser_compute(
        self: Scheduler, req: Req, request_id: str, role_name: str
    ) -> None:
        """Run denoiser compute, then enqueue staged output for decoder transfer."""
        if self._is_request_aborted(request_id):
            return
        # Run denoising
        start_time = time.monotonic()
        result = self.worker.execute_forward([req], return_req=True)
        duration_s = time.monotonic() - start_time

        if self._is_request_aborted(request_id):
            return

        if not isinstance(result, Req):
            error_msg = getattr(result, "error", "denoiser error")
            done_msg = TransferDoneMsg(request_id=request_id, error=str(error_msg))
            self._pool_result_push.send_multipart(encode_transfer_msg(done_msg))
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return

        # Stage denoiser output for decoder transfer (async staging)
        tensor_fields, scalar_fields = extract_transfer_fields(result)
        staged, stage_event = self._transfer_manager.stage_tensors_async(
            request_id=request_id,
            tensor_fields=tensor_fields,
            scalar_fields=scalar_fields,
            stream=self._transfer_stream,
        )

        if staged is None:
            done_msg = TransferDoneMsg(
                request_id=request_id,
                error="Failed to stage denoiser output for decoder",
            )
            self._pool_result_push.send_multipart(encode_transfer_msg(done_msg))
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return

        self._enqueue_outbound_transfer(
            request_id,
            staged,
            stage_event,
            msg_type=TransferMsgType.DONE,
            staged_for_decoder=True,
        )

        if self._disagg_metrics:
            self._disagg_metrics.record_request_complete(request_id)

        logger.debug(
            "Transfer DENOISER: processed %s in %.2f s, staged for decoder",
            request_id,
            duration_s,
        )

    def _disagg_decoder_compute(
        self: Scheduler, req: Req, request_id: str, role_name: str
    ) -> None:
        """Run decoder compute in transfer mode, send result to DS.

        Decoder result is sent as raw ZMQ multipart frames (same format as
        relay mode) so DiffusionServer handles it via _handle_decoder_result_frames
        without hex/JSON overhead.
        """

        # Check for upstream error
        disagg_error = getattr(req, "_disagg_error", None)
        if disagg_error:
            if self._pool_result_push is not None:
                send_tensors(
                    self._pool_result_push,
                    {},
                    {
                        "request_id": request_id,
                        "error": f"Upstream error: {disagg_error}",
                    },
                )
            return

        req.save_output = False
        req.return_file_paths_only = False
        if self._is_request_aborted(request_id):
            return

        start_time = time.monotonic()
        output_batch = self.worker.execute_forward([req])
        duration_s = time.monotonic() - start_time

        if self._is_request_aborted(request_id):
            return

        # Send result as raw ZMQ frames (no TRANSFER_MAGIC prefix).
        # DiffusionServer will route it through _handle_decoder_result_frames,
        # the same path as relay mode.
        tensor_fields = {}
        scalar_fields = {"request_id": request_id}
        if output_batch.output is not None:
            tensor_fields["output"] = output_batch.output
        if output_batch.audio is not None:
            tensor_fields["audio"] = output_batch.audio
        if output_batch.audio_sample_rate is not None:
            scalar_fields["audio_sample_rate"] = output_batch.audio_sample_rate
        if output_batch.error is not None:
            scalar_fields["error"] = output_batch.error

        if self._pool_result_push is not None:
            send_tensors(self._pool_result_push, tensor_fields, scalar_fields)

        if self._disagg_metrics:
            if output_batch.error:
                self._disagg_metrics.record_request_failed(request_id)
            else:
                self._disagg_metrics.record_request_complete(request_id)

        logger.debug("Transfer DECODER: processed %s in %.2f s", request_id, duration_s)

    def _disagg_encoder_step(
        self: Scheduler,
        send_tensors_fn,
        frames=None,
    ):
        """Single encoder step in pool mode."""
        # Receive: [request_id_bytes, pickled_req_bytes]
        if frames is None:
            frames = self._pool_work_pull.recv_multipart()
        pickled_req = frames[-1]
        reqs = pickle.loads(pickled_req)
        if not isinstance(reqs, list):
            reqs = [reqs]

        req = reqs[0]
        request_id = getattr(req, "request_id", "unknown")

        if self._disagg_metrics:
            self._disagg_metrics.record_request_start(request_id)

        # Run encoder stages
        req_result = self.worker.execute_forward(reqs, return_req=True)

        if self._is_request_aborted(request_id):
            return

        if not isinstance(req_result, Req):
            # Error 閳?send error via scalar fields (rank 0 only)
            if self._pool_result_push is not None:
                error_msg = getattr(req_result, "error", "encoder error")
                send_tensors_fn(
                    self._pool_result_push,
                    {},
                    {"request_id": request_id, "_disagg_error": str(error_msg)},
                )
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return

        # Pack and send encoder output (rank 0 only sends)
        tensor_fields, scalar_fields = extract_transfer_fields(req_result)

        if self._pool_result_push is not None:
            if self._transfer_manager is not None:
                # Transfer mode: stage tensors to TransferBuffer, send transfer_staged
                self._disagg_encoder_transfer_stage(
                    request_id, tensor_fields, scalar_fields
                )
            else:
                # Fallback: send error (transfer manager not initialized)
                send_tensors_fn(
                    self._pool_result_push,
                    {},
                    {"request_id": request_id, "_disagg_error": "No transfer manager"},
                )

        if self._disagg_metrics:
            self._disagg_metrics.record_request_complete(request_id)

        logger.debug("Pool ENCODER: processed %s", request_id)

    def _disagg_encoder_transfer_stage(
        self: Scheduler, request_id: str, tensor_fields: dict, scalar_fields: dict
    ) -> None:
        """Stage encoder output and send transfer_staged to DS.

        The actual server notification is delayed until the per-request D2H event
        becomes ready, so the sender can trigger dispatch at request granularity.
        """
        staged, stage_event = self._transfer_manager.stage_tensors_async(
            request_id=request_id,
            tensor_fields=tensor_fields,
            scalar_fields=scalar_fields,
            stream=self._transfer_stream,
        )

        if self._is_request_aborted(request_id):
            if self._transfer_manager is not None:
                self._transfer_manager.abort_request(request_id)
            return

        if staged is None:
            # Staging failed 閳?send error via relay as fallback
            send_tensors(
                self._pool_result_push,
                {},
                {"request_id": request_id, "_disagg_error": "Transfer staging failed"},
            )
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return

        self._enqueue_outbound_transfer(
            request_id,
            staged,
            stage_event,
            msg_type=TransferMsgType.STAGED,
        )
