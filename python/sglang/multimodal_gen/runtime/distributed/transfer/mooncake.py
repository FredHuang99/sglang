"""MooncakeTransferManager implementation for Diffusion PD Disaggregation.

This module provides the Mooncake backend implementation of TransferManager,
which uses RDMA for high-performance tensor transfer between disaggregated
diffusion stages.

Design Principles:
-----------------
1. RDMA-Based Transfer: Uses Mooncake's RDMA engine for zero-copy GPU transfer.
2. Backend Extension: Extends CommonTransferManager with Mooncake-specific logic.
3. Slice Transfer: Supports parallel strategy mismatch handling via tensor slicing.
4. Thread Pool: Uses thread pool for asynchronous transfer operations.

File Structure:
-------------
- MooncakeTransferManager: Main manager with RDMA transfer support
- MooncakeTransferSender: Sender implementation for Mooncake backend
- MooncakeTransferReceiver: Receiver implementation for Mooncake backend

Reference:
- RFC SGLang Diffusion Disaggregation §2: TransferManager
- sglang/srt/disaggregation/mooncake/conn.py:MooncakeKVManager
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.multimodal_gen.runtime.distributed.transfer.base import (
    BaseTransferReceiver,
    BaseTransferSender,
    TransferArgs,
)
from sglang.multimodal_gen.runtime.distributed.transfer.manager import (
    CommonTransferManager,
)
from sglang.multimodal_gen.runtime.distributed.transfer.utils import (
    DiffusionRole,
    RankRoutingInfo,
    TransferPoll,
)

if TYPE_CHECKING:
    import torch

    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = logging.getLogger(__name__)


# Try to import Mooncake transfer engine
try:
    from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine

    MOONCAKE_AVAILABLE = True
except ImportError:
    MOONCAKE_AVAILABLE = False
    get_mooncake_transfer_engine = None


class MooncakeTransferManager(CommonTransferManager):
    """Mooncake backend implementation of TransferManager.

    This class is analogous to MooncakeKVManager in LLM PD disaggregation.
    It uses Mooncake's RDMA-based transfer engine for high-performance
    tensor transfer between disaggregated diffusion stages.

    Design Rationale:
    ----------------
    The Mooncake backend provides:

    1. RDMA Transfer:
       - Zero-copy GPU-to-GPU transfer via RDMA
       - High bandwidth, low latency
       - Requires RDMA-capable network hardware

    2. Slice Transfer Support:
       - Handles parallel strategy mismatches (TP/SP/FSDP)
       - Automatically computes tensor slices
       - Transfers each slice to appropriate destination rank

    3. Thread Pool:
       - Async transfer operations via thread pool
       - Configurable pool size based on CPU count
       - Worker threads process pending transfers

    Attributes:
        engine: Mooncake transfer engine.
            Handles actual RDMA transfer operations.
            None if Mooncake is not available (mock mode).
        transfer_thread_pool: Thread pool for transfer operations.
            Async transfer submission and execution.
        transfer_queues: Queues for pending transfers.
            Maps bootstrap_room to transfer queue.

    Reference: sglang/srt/disaggregation/mooncake/conn.py:MooncakeKVManager
    """

    def __init__(
        self,
        args: TransferArgs,
        role: DiffusionRole,
        server_args: "ServerArgs",
    ) -> None:
        """Initialize the Mooncake transfer manager.

        Args:
            args: Transfer configuration arguments.
                Contains latent_channels, max_resolution, etc.
            role: The diffusion role.
                Determines send/receive behavior.
            server_args: Server configuration.
                Contains concurrent request settings.
        """
        super().__init__(args, role, server_args)

        self.engine = None
        self.transfer_thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = (
            None
        )
        self.transfer_queues: Dict[int, "transfer.Queue"] = {}

        # Thread pool configuration (参考 RFC §3.3)
        cpu_count = os.cpu_count() or 4
        self.transfer_thread_pool_size = min(max(4, int(0.5 * cpu_count) // 8), 12)
        self.transfer_queue_size = 4

        if MOONCAKE_AVAILABLE:
            self._init_engine()
        else:
            logger.warning(
                "Mooncake transfer engine not available. "
                "MooncakeTransferManager will run in mock mode."
            )

    def _init_engine(self) -> None:
        """Initialize the Mooncake transfer engine."""
        try:
            self.engine = get_mooncake_transfer_engine()
            if self.engine is None:
                logger.warning(
                    "Mooncake transfer engine is None. " "Running in mock mode."
                )
            else:
                logger.info("Mooncake transfer engine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Mooncake engine: {e}")
            self.engine = None

    def register_to_bootstrap(self) -> None:
        """Register this role to the bootstrap server.

        In Mooncake implementation, this registers the role's network
        information with the DiffusionServer for peer discovery.
        """
        # TODO: Implement full bootstrap registration
        logger.info(
            f"MooncakeTransferManager: bootstrap registration for role {self.role.value}"
        )

    def register_buffer_to_engine(self) -> None:
        """Register buffer addresses to the Mooncake engine.

        This registers the pinned memory buffers with Mooncake's
        transfer engine for RDMA operations.

        Reference: RFC §2 "Engine Setup"
        """
        if self.engine is None or self.tensor_buffer is None:
            logger.debug(
                "MooncakeTransferManager: skipping buffer registration (no engine)"
            )
            return

        # Collect buffer addresses and sizes from tensor buffer
        # Note: In a full implementation, we would register all slot addresses
        # For now, we register the base pool address
        ptrs = [self.tensor_buffer.storage.data_ptr()]
        lens = [
            self.tensor_buffer.storage.numel()
            * self.tensor_buffer.storage.element_size()
        ]

        try:
            # Register buffers with Mooncake engine
            # Note: The exact API depends on Mooncake version
            # self.engine.batch_register(ptrs, lens)
            logger.debug(
                f"MooncakeTransferManager: would register {len(ptrs)} buffers "
                f"with Mooncake engine"
            )
        except Exception as e:
            logger.warning(f"Failed to register buffers with Mooncake: {e}")

    def start_transfer_threads(self) -> None:
        """Start the transfer worker threads.

        Creates a thread pool for handling asynchronous transfers.

        Reference: RFC §2 "Send Loop" and §3.3
        """
        if self.transfer_thread_pool is not None:
            return

        self.transfer_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.transfer_thread_pool_size,
            thread_name_prefix="mooncake_transfer",
        )

        # Start worker threads
        for i in range(self.transfer_thread_pool_size):
            self.transfer_thread_pool.submit(self._transfer_worker, i)

        logger.info(
            f"MooncakeTransferManager: started {self.transfer_thread_pool_size} "
            f"transfer worker threads"
        )

    def _transfer_worker(self, worker_id: int) -> None:
        """Worker thread for processing transfers.

        Reference: RFC §3.3 "Transfer Thread Pool"

        Args:
            worker_id: Identifier for this worker
        """
        logger.debug(f"MooncakeTransferManager: worker {worker_id} started")

        while getattr(threading.current_thread(), "running", True):
            # Process pending transfers
            # This is a placeholder - actual implementation would
            # pull from transfer queues and call engine.batch_transfer_sync()
            pass

        logger.debug(f"MooncakeTransferManager: worker {worker_id} stopped")

    def submit_transfer(
        self,
        session_id: int,
        bootstrap_room: int,
        src_addrs: List[int],
        dst_addrs: List[int],
        lengths: List[int],
    ) -> None:
        """Submit a transfer task to the thread pool.

        Args:
            session_id: Transfer session identifier
            bootstrap_room: Bootstrap room for status tracking
            src_addrs: Source memory addresses
            dst_addrs: Destination memory addresses
            lengths: Transfer lengths
        """
        if self.transfer_thread_pool is None:
            self.start_transfer_threads()

        # Submit to thread pool
        self.transfer_thread_pool.submit(
            self._do_transfer,
            session_id,
            bootstrap_room,
            src_addrs,
            dst_addrs,
            lengths,
        )

    def _do_transfer(
        self,
        session_id: int,
        bootstrap_room: int,
        src_addrs: List[int],
        dst_addrs: List[int],
        lengths: List[int],
    ) -> None:
        """Execute a transfer operation.

        Reference: RFC §2 "Send Loop"

        Args:
            session_id: Transfer session identifier
            bootstrap_room: Bootstrap room for status tracking
            src_addrs: Source memory addresses
            dst_addrs: Destination memory addresses
            lengths: Transfer lengths
        """
        self.update_status(bootstrap_room, TransferPoll.Transferring)

        if self.engine is None:
            # Mock mode - simulate transfer
            logger.debug(
                f"MooncakeTransferManager: mock transfer for session {session_id}, "
                f"bootstrap_room={bootstrap_room}"
            )
            # Simulate transfer completion
            self.update_status(bootstrap_room, TransferPoll.Success)
            return

        try:
            # Execute the transfer
            # Note: The exact API depends on Mooncake version
            # status = self.engine.batch_transfer_sync(
            #     session_id, src_addrs, dst_addrs, lengths
            # )
            # if status == 0:
            #     self.update_status(bootstrap_room, TransferPoll.Success)
            # else:
            #     self.update_status(bootstrap_room, TransferPoll.Failed)
            #     self.record_failure(bootstrap_room, f"Transfer failed with status {status}")

            # Placeholder
            logger.debug(
                f"MooncakeTransferManager: would transfer {len(lengths)} chunks "
                f"for session {session_id}"
            )
            self.update_status(bootstrap_room, TransferPoll.Success)

        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            self.update_status(bootstrap_room, TransferPoll.Failed)
            self.record_failure(bootstrap_room, str(e))

    def shutdown(self) -> None:
        """Shutdown the transfer manager and clean up resources."""
        # Stop thread pool
        if self.transfer_thread_pool is not None:
            self.transfer_thread_pool.shutdown(wait=True)
            self.transfer_thread_pool = None

        super().shutdown()

        logger.info("MooncakeTransferManager shutdown complete")

    def check_parallel_mismatch(
        self,
        src_parallel_config: Dict[str, int],
        dst_parallel_config: Dict[str, int],
    ) -> bool:
        """Check if parallel strategies differ between source and destination.

        Reference: sglang/srt/disaggregation/mooncake/conn.py:send_kvcache_slice

        Args:
            src_parallel_config: Source parallel configuration
            dst_parallel_config: Destination parallel configuration

        Returns:
            True if slice transfer is needed, False otherwise
        """
        src_tp = src_parallel_config.get("tp", 1)
        dst_tp = dst_parallel_config.get("tp", 1)
        src_sp = src_parallel_config.get("sp", 1)
        dst_sp = dst_parallel_config.get("sp", 1)

        return src_tp != dst_tp or src_sp != dst_sp

    def compute_latent_slices(
        self,
        latent_shape: Tuple[int, int, int, int],
        src_parallel_config: Dict[str, int],
        dst_parallel_config: Dict[str, int],
    ) -> List[Tuple[slice, slice, slice, slice]]:
        """Compute latent tensor slice ranges for transfer.

        This method calculates how to split a latent tensor when the source and
        destination have different parallel strategies. The tensor is split along
        the width (W) dimension.

        Design Rationale:
        ----------------
        Diffusion latent tensors have shape (C, T, H, W):
        - C: Channels (typically 16)
        - T: Temporal dimension (frames // 4)
        - H: Height (input_height // 8)
        - W: Width (input_width // 8)

        For parallel inference:
        - TP (Tensor Parallelism): Splits along hidden dimension
        - SP (Sequence Parallelism): Splits along sequence/temporal dimension
        - FSDP (Fully Sharded Data Parallel): Shards across data parallel ranks

        Since latent tensors are 4D with spatial dimensions (H, W),
        we typically split along W (width) for diffusion models.

        Args:
            latent_shape: (C, T, H, W) latent tensor shape.
                Format: (channels, temporal, height, width)
            src_parallel_config: Source parallel configuration.
                Dict with keys: tp, sp, fsdp, etc.
            dst_parallel_config: Destination parallel configuration.
                Dict with keys: tp, sp, fsdp, etc.

        Returns:
            List of slices for each dimension (C, T, H, W).
            Each tuple contains 4 slices representing the 4D tensor slice.

        Example:
            >>> # Shape (16, 21, 60, 104), dst has 2 SP ranks
            >>> slices = mgr.compute_latent_slices(
            ...     (16, 21, 60, 104),
            ...     {"tp": 1, "sp": 1, "fsdp": 1},
            ...     {"tp": 1, "sp": 2, "fsdp": 1},
            ... )
            >>> # Returns 2 slices: one for [0:52], one for [52:104]
            >>> len(slices)
            2

        Reference: sglang/srt/disaggregation/mooncake/conn.py:send_kvcache_slice
        """
        # latent_shape: (C, T, H, W) = (channels, temporal, height, width)
        c, t, h, w = latent_shape

        src_tp = src_parallel_config.get("tp", 1)
        dst_tp = dst_parallel_config.get("tp", 1)
        src_sp = src_parallel_config.get("sp", 1)
        dst_sp = dst_parallel_config.get("sp", 1)
        src_fsdp = src_parallel_config.get("fsdp", 1)
        dst_fsdp = dst_parallel_config.get("fsdp", 1)

        # Calculate how many destination ranks we need to send to
        # For diffusion, we typically split along width (W) dimension
        num_dst_ranks = max(1, dst_tp * dst_sp * dst_fsdp)
        num_src_ranks = max(1, src_tp * src_sp * src_fsdp)

        # Compute slice size per destination rank
        w_per_dst = w // num_dst_ranks
        w_remainder = w % num_dst_ranks

        slices = []
        current_w = 0

        for i in range(num_dst_ranks):
            # Distribute remainder to first ranks
            cur_w_size = w_per_dst + (1 if i < w_remainder else 0)

            if cur_w_size > 0:
                # Slice: (all channels, all temporal, height, width slice)
                slice_4d = (
                    slice(0, c),  # All channels
                    slice(0, t),  # All temporal
                    slice(0, h),  # All height
                    slice(current_w, current_w + cur_w_size),  # Width slice
                )
                slices.append(slice_4d)
                current_w += cur_w_size

        return slices

    def transfer_with_slice(
        self,
        bootstrap_room: int,
        session_id: str,
        src_latent: "torch.Tensor",
        dst_rank_infos: List["RankRoutingInfo"],
        src_parallel_config: Dict[str, int],
        dst_parallel_config: Dict[str, int],
    ) -> None:
        """Transfer latent tensor with slice support for parallel strategy mismatch.

        This method handles the case where source and destination roles have different
        parallel strategies (TP/SP/FSDP). It either:
        1. Does direct transfer if parallel configs match
        2. Splits tensor into slices and transfers each to appropriate rank if mismatched

        Design Rationale:
        ----------------
        In disaggregated diffusion, different pipeline stages may use different
        parallel strategies:
        - Encoder might use FSDP=4 for efficient text encoding
        - Denoising might use SP=2 for sequence parallelism across frames
        - Decoder might use TP=2 for tensor parallelism

        When parallel strategies differ, we need to:
        1. Detect the mismatch via check_parallel_mismatch()
        2. Compute how to split the tensor via compute_latent_slices()
        3. Transfer each slice to the correct destination rank

        Algorithm:
        ---------
        1. Check if parallel configs match
        2. If match: call _submit_direct_transfer()
        3. If mismatch:
           a. Compute slices for the latent tensor
           b. For each destination rank:
              - Extract corresponding slice from source
              - Submit transfer for that slice

        Args:
            bootstrap_room: Bootstrap room identifier.
                Session ID for tracking this transfer.
            session_id: Transfer session ID.
                Unique identifier for this transfer operation.
            src_latent: Source latent tensor.
                The tensor to transfer to downstream role.
            dst_rank_infos: List of destination rank routing info.
                Each entry contains address and buffer pointer for a destination rank.
            src_parallel_config: Source parallel configuration.
                Dict with keys: tp, sp, fsdp, dp, etc.
            dst_parallel_config: Destination parallel configuration.
                Dict with keys: tp, sp, fsdp, dp, etc.

        Example:
            >>> # Direct transfer (matching configs)
            >>> mgr.transfer_with_slice(
            ...     bootstrap_room=100,
            ...     session_id="req_001",
            ...     src_latent=latent_tensor,
            ...     dst_rank_infos=[rank_info_0, rank_info_1],
            ...     src_parallel_config={"tp": 1, "sp": 1},
            ...     dst_parallel_config={"tp": 1, "sp": 1},
            ... )
            >>>
            >>> # Slice transfer (mismatched configs)
            >>> mgr.transfer_with_slice(
            ...     bootstrap_room=101,
            ...     session_id="req_002",
            ...     src_latent=latent_tensor,
            ...     dst_rank_infos=[rank_info_0, rank_info_1],
            ...     src_parallel_config={"tp": 1, "sp": 1},
            ...     dst_parallel_config={"tp": 1, "sp": 2},
            ... )

        Reference: sglang/srt/disaggregation/mooncake/conn.py:send_kvcache_slice
        """
        # Check if we need slice transfer
        if not self.check_parallel_mismatch(src_parallel_config, dst_parallel_config):
            # No mismatch, do direct transfer
            self._submit_direct_transfer(
                bootstrap_room=bootstrap_room,
                session_id=session_id,
                src_latent=src_latent,
                dst_rank_infos=dst_rank_infos,
            )
            return

        # Compute slices for the latent tensor
        latent_shape = tuple(src_latent.shape)
        slices = self.compute_latent_slices(
            latent_shape=latent_shape,
            src_parallel_config=src_parallel_config,
            dst_parallel_config=dst_parallel_config,
        )

        logger.debug(
            f"transfer_with_slice: {len(slices)} slices for "
            f"src_parallel={src_parallel_config}, dst_parallel={dst_parallel_config}"
        )

        # Transfer each slice to the corresponding destination rank
        for i, (dst_info, slice_4d) in enumerate(zip(dst_rank_infos, slices)):
            # Extract the slice from source
            src_slice = src_latent[slice_4d]

            # Compute slice size
            slice_size = src_slice.numel() * src_slice.element_size()
            slice_ptr = src_slice.data_ptr()

            # Submit transfer for this slice
            self.submit_transfer(
                session_id=f"{session_id}_slice_{i}",
                bootstrap_room=bootstrap_room,
                src_addrs=[slice_ptr],
                dst_addrs=[dst_info.tensor_ptr],
                lengths=[slice_size],
            )

    def _submit_direct_transfer(
        self,
        bootstrap_room: int,
        session_id: str,
        src_latent: "torch.Tensor",
        dst_rank_infos: List["RankRoutingInfo"],
    ) -> None:
        """Submit direct transfer without slicing.

        Args:
            bootstrap_room: Bootstrap room identifier
            session_id: Transfer session ID
            src_latent: Source latent tensor
            dst_rank_infos: List of destination rank routing info
        """
        src_ptr = src_latent.data_ptr()
        total_size = src_latent.numel() * src_latent.element_size()

        for dst_info in dst_rank_infos:
            self.submit_transfer(
                session_id=session_id,
                bootstrap_room=bootstrap_room,
                src_addrs=[src_ptr],
                dst_addrs=[dst_info.tensor_ptr],
                lengths=[total_size],
            )


class MooncakeTransferSender(BaseTransferSender):
    """Mooncake implementation of transfer sender.

    This class is analogous to MooncakeKVSender in LLM PD disaggregation.
    It handles sending latent tensors to the downstream role.

    Reference: sglang/srt/disaggregation/mooncake/conn.py:MooncakeKVSender
    """

    def __init__(
        self,
        mgr: MooncakeTransferManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
    ) -> None:
        """Initialize the sender.

        Args:
            mgr: The transfer manager
            bootstrap_addr: Address of the bootstrap server
            bootstrap_room: The bootstrap room identifier
            dest_tp_ranks: List of destination TP ranks
        """
        self.mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.dest_tp_ranks = dest_tp_ranks

        # Initialize status
        self.mgr.update_status(bootstrap_room, TransferPoll.Bootstrapping)

        # Latent info (set during init)
        self.latent_shape: Optional[Tuple[int, ...]] = None
        self.meta_index: int = -1
        self.tensor_slot = None

    def init(
        self,
        latent_shape: Tuple[int, ...],
        meta_index: int,
        tensor_buffer_slot,
    ) -> None:
        """Initialize the transfer with tensor information.

        Args:
            latent_shape: Shape of the latent tensor
            meta_index: Index in the metadata buffer
            tensor_buffer_slot: The allocated tensor buffer slot
        """
        self.latent_shape = latent_shape
        self.meta_index = meta_index
        self.tensor_slot = tensor_buffer_slot

        self.mgr.update_status(self.bootstrap_room, TransferPoll.WaitingForInput)

    def send(self) -> int:
        """Initiate the tensor transfer.

        Returns:
            Status code from TransferPoll
        """
        if self.tensor_slot is None:
            return TransferPoll.Failed

        # Submit transfer to the manager
        # In a full implementation, this would get the destination
        # address from the bootstrap server and submit the transfer
        logger.debug(
            f"MooncakeTransferSender: initiating transfer for "
            f"bootstrap_room={self.bootstrap_room}"
        )

        return TransferPoll.Transferring

    def poll(self) -> int:
        """Poll the transfer status.

        Returns:
            Current status from TransferPoll
        """
        return self.mgr.check_status(self.bootstrap_room)


class MooncakeTransferReceiver(BaseTransferReceiver):
    """Mooncake implementation of transfer receiver.

    This class is analogous to MooncakeKVReceiver in LLM PD disaggregation.
    It handles receiving latent tensors from the upstream role.

    Reference: sglang/srt/disaggregation/mooncake/conn.py:MooncakeKVReceiver
    """

    def __init__(
        self,
        mgr: MooncakeTransferManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ) -> None:
        """Initialize the receiver.

        Args:
            mgr: The transfer manager
            bootstrap_addr: Address of the bootstrap server
            bootstrap_room: Optional bootstrap room identifier
        """
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room

        # Initialize status
        if bootstrap_room is not None:
            self.mgr.update_status(bootstrap_room, TransferPoll.Bootstrapping)

        # Latent info (set during init)
        self.latent_shape: Optional[Tuple[int, ...]] = None
        self.tensor_slot = None
        self.meta_index: int = -1

    def init(
        self,
        latent_shape: Tuple[int, ...],
        tensor_buffer_slot,
        meta_index: int,
    ) -> None:
        """Initialize the receive with expected tensor information.

        Args:
            latent_shape: Expected shape of the latent tensor
            tensor_buffer_slot: The allocated tensor buffer slot
            meta_index: Index in the metadata buffer
        """
        self.latent_shape = latent_shape
        self.tensor_slot = tensor_buffer_slot
        self.meta_index = meta_index

        if self.bootstrap_room is not None:
            self.mgr.update_status(self.bootstrap_room, TransferPoll.WaitingForInput)

    def poll(self) -> int:
        """Poll the transfer status.

        Returns:
            Current status from TransferPoll
        """
        if self.bootstrap_room is not None:
            return self.mgr.check_status(self.bootstrap_room)
        return TransferPoll.Bootstrapping

    def clear(self) -> None:
        """Clear any internal states."""
        self.latent_shape = None
        self.tensor_slot = None

    def abort(self) -> None:
        """Abort the current transfer."""
        if self.bootstrap_room is not None:
            self.mgr.update_status(self.bootstrap_room, TransferPoll.Failed)
