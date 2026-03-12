"""Utilities for Diffusion PD Disaggregation Transfer Layer.

This module defines the core enumerations and utilities for the transfer layer,
including role types, transfer backends, and transfer status polling.

Design Principles:
-----------------
1. Role-based Design: Different pipeline stages (Encoder/Denoising/Decoder)
   have different resource requirements and must be configured accordingly.
2. Stateless Utilities: Functions in this module are stateless to ensure
   thread-safety and easy testing.
3. RFC Compliance: All designs follow RFC SGLang Diffusion Disaggregation.

Reference:
- RFC SGLang Diffusion Disaggregation §2: Core Components per Role
- sglang/srt/disaggregation/utils.py for LLM PD patterns

File Structure:
--------------
- RankRoutingInfo: Routing cache data structure
- DiffusionRole: Pipeline stage enumeration
- TransferBackend: Transfer backend enumeration
- TransferPoll: Transfer state machine
- calculate_latent_size: Latent tensor size calculation
- align_to_power_of_two: Memory alignment utility
- get_slot_config: Role-specific buffer configuration
- group_concurrent_contiguous: Transfer optimization
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


@dataclass
class RankRoutingInfo:
    """Routing information for a downstream rank.

    This class stores the network address and buffer pointers for a downstream
    role instance, used for direct tensor transfer without a central coordinator.

    Design Rationale:
    ----------------
    In disaggregated diffusion, each role instance needs to know the network
    address and buffer location of its peers to enable direct RDMA transfer.
    The routing table caches this information to avoid repeated address lookups.

    Data Flow:
    ----------
    1. When a role starts, it registers its address with the Bootstrap Server
    2. When another role needs to send data, it queries the routing table
    3. The routing table maps instance_id -> RankRoutingInfo

    Attributes:
        instance_id: Unique identifier for the role instance (e.g., "encoder_001")
            Used as the key in routing table lookups.
        rank_id: Rank ID within the role
            Identifies which rank within a distributed group.
        ip: IP address for RDMA transfer
            The network address used for direct memory access.
        port: Port number for communication
            The port for ZMQ/HTTP control plane communication.
        tensor_ptr: Memory pointer for tensor data
            GPU memory address for latent tensor transfer.
        meta_ptr: Memory pointer for metadata
            CPU pinned memory address for metadata transfer.
        parallel_config: Parallel strategy configuration
            Dict containing dp_rank, tp_rank, pp_rank, sp_size, fsdp_size.
            Used to detect parallel strategy mismatches for slice transfer.
        timestamp: Time when this info was last updated
            Used for expiration to avoid stale routing information.

    Example:
        >>> info = RankRoutingInfo(
        ...     instance_id="encoder_001",
        ...     rank_id=0,
        ...     ip="10.0.1.10",
        ...     port=50001,
        ...     tensor_ptr=0x7f8a1000,
        ...     meta_ptr=0x7f8a2000,
        ...     parallel_config={"dp": 1, "tp": 1, "sp": 1, "fsdp": 1}
        ... )
    """

    instance_id: str
    rank_id: int
    ip: str
    port: int
    tensor_ptr: int
    meta_ptr: int
    parallel_config: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def is_expired(self, max_age_seconds: float = 300.0) -> bool:
        """Check if this routing info has expired.

        Routing information can become stale when:
        - A role instance restarts
        - Network configuration changes
        - Long-running requests complete

        Args:
            max_age_seconds: Maximum age in seconds before considering expired.
                Default 300 seconds (5 minutes) to balance freshness vs overhead.

        Returns:
            True if expired (older than max_age_seconds), False otherwise.

        Example:
            >>> info = RankRoutingInfo(...)
            >>> info.is_expired(max_age_seconds=60)
            False  # if recently updated
            >>> info.is_expired(max_age_seconds=60)
            True  # if older than 60 seconds
        """
        return (time.time() - self.timestamp) > max_age_seconds


class DiffusionRole(Enum):
    """Role types in Diffusion PD disaggregation.

    Three stages in the diffusion pipeline that can be disaggregated:

    1. ENCODER (value="encoder"):
        - Text encoding: Converts prompt to embeddings
        - Conditioning: Prepares conditioning signals (CLIP, T5, etc.)
        - Latent preparation: Generates initial noise latent

        Resource profile:
        - Short execution time (fast)
        - Lower memory footprint
        - Can use aggressive parallelization

    2. DENOISING (value="denoising"):
        - DiT denoising: Multi-step iterative denoising
        - Largest compute and memory requirements
        - Longest execution time

        Resource profile:
        - Longest execution time (bottleneck)
        - Highest memory usage
        - Benefits from more buffer slots (backpressure absorption)

    3. DECODER (value="decoder"):
        - VAE decoding: Converts latent to final image/video
        - Shorter execution time than Denoising

        Resource profile:
        - Medium execution time
        - Moderate memory footprint
        - Similar buffer needs to Encoder

    4. NULL (value="null"):
        - Non-disaggregated mode
        - All stages run on same node
        - No network transfer needed

    Design Rationale:
    ----------------
    Each role has distinct characteristics requiring different buffer configs.
    The is_sender() and is_receiver() methods help determine data flow direction.

    Example:
        >>> role = DiffusionRole.ENCODER
        >>> role.is_sender()
        True
        >>> role.is_receiver()
        False
        >>> role = DiffusionRole.DENOISING
        >>> role.is_sender()
        True
        >>> role.is_receiver()
        True
    """

    ENCODER = "encoder"
    DENOISING = "denoising"
    DECODER = "decoder"
    NULL = "null"

    def is_sender(self) -> bool:
        """Check if this role sends data to next stage.

        Returns:
            True if role sends to downstream, False otherwise.

        Roles that send:
        - ENCODER: Sends latent to DENOISING
        - DENOISING: Sends denoised latent to DECODER
        """
        return self in (DiffusionRole.ENCODER, DiffusionRole.DENOISING)

    def is_receiver(self) -> bool:
        """Check if this role receives data from previous stage.

        Returns:
            True if role receives from upstream, False otherwise.

        Roles that receive:
        - DENOISING: Receives from ENCODER
        - DECODER: Receives from DENOISING
        """
        return self in (DiffusionRole.DENOISING, DiffusionRole.DECODER)


class TransferBackend(Enum):
    """Backend for tensor transfer in disaggregated diffusion.

    Supported backends:

    1. MOONCAKE (value="mooncake"):
        - RDMA-based transfer using Mooncake library
        - Highest performance for GPU-to-GPU transfer
        - Requires RDMA-capable network hardware

    Future backends (reserved):
    - NIXL: NVIDIA transport library
    - MORI: Alternative transport

    Design Rationale:
    ----------------
    The backend abstraction allows plugging different transport mechanisms
    while maintaining a consistent API for the transfer layer.
    """

    MOONCAKE = "mooncake"
    # NIXL = "nixl"  # Reserved for future
    # MORI = "mori"  # Reserved for future


class TransferPoll:
    """Transfer status polling states.

    State Machine:
    --------------
    Bootstrapping -> WaitingForInput -> Transferring -> Success
                            |                    |
                            |                    +---> Failed
                            +-------------------> Failed

    State Descriptions:
    -------------------
    - Failed (value=0): Transfer failed (terminal state)
    - Bootstrapping (value=1): Initializing connection
    - WaitingForInput (value=2): Waiting for upstream to prepare data
    - Transferring (value=3): Actively transferring over network
    - Success (value=4): Transfer completed successfully (terminal state)

    Design Rationale:
    ----------------
    Using IntEnum values allows efficient status comparison and storage.
    The order matters: higher values represent later stages (for max() operations).
    """

    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


def calculate_latent_size(
    height: int,
    width: int,
    num_frames: int,
    latent_channels: int = 16,
    dtype_size: int = 2,
) -> int:
    """Calculate latent tensor size in bytes.

    VAE Architecture:
    -----------------
    Diffusion models use Variational Autoencoders (VAE) for:
    1. Encoding: Image/Video -> Latent (encoder)
    2. Decoding: Latent -> Image/Video (decoder)

    Downsampling Ratios:
    -------------------
    - Spatial: 8x reduction (height//8, width//8)
    - Temporal: 4x reduction ((num_frames-1)//4 + 1)

    Calculation:
    ------------
    latent_size = channels * latent_t * latent_h * latent_w * dtype_size
                 = channels * ((num_frames-1)//4 + 1) * (height//8) * (width//8) * dtype_size

    Args:
        height: Input image/video height in pixels.
            Example: 480, 720, 1080
        width: Input image/video width in pixels.
            Example: 832, 1280, 1920
        num_frames: Number of frames in the input video.
            Example: 81 frames for 5-second video at 16fps
        latent_channels: Number of latent channels.
            Default 16 for most diffusion models (Wan, FLUX, etc.)
        dtype_size: Size of data type in bytes.
            Default 2 for fp16, 4 for bf16/fp32

    Returns:
        Size in bytes required for the latent tensor.

    Example:
        >>> # 480p video, 5 seconds, fp16
        >>> calculate_latent_size(480, 832, 81)
        4194304  # ~4MB
        >>> # 1080p video, 15 seconds, fp16
        >>> calculate_latent_size(1080, 1920, 241)
        63158272  # ~60MB

    Reference: RFC §3.2 "Wan2.2 T2V Latent Size Calculation"
    """
    latent_h = height // 8
    latent_w = width // 8
    latent_t = (num_frames - 1) // 4 + 1
    return latent_channels * latent_t * latent_h * latent_w * dtype_size


def align_to_power_of_two(size: int) -> int:
    """Align size to the next power of two.

    RDMA Requirements:
    -----------------
    RDMA transfers require memory to be:
    1. Physically contiguous
    2. Aligned to specific boundaries (typically power of 2)

    Algorithm:
    ----------
    Start with 1 and double until >= size:
    - size=5  -> 1,2,4,8 -> 8
    - size=8  -> 8
    - size=9  -> 1,2,4,8,16 -> 16

    Args:
        size: Original size in bytes that needs alignment.

    Returns:
        Next power of two >= size.

    Example:
        >>> align_to_power_of_two(1000)
        1024
        >>> align_to_power_of_two(1024)
        1024
        >>> align_to_power_of_two(1025)
        2048
    """
    if size <= 0:
        return 1

    # Find the next power of two
    power = 1
    while power < size:
        power <<= 1
    return power


def get_slot_config(
    role: DiffusionRole,
    concurrent_requests: int,
) -> Tuple[int, int]:
    """Get slot configuration for a specific role.

    Role-Specific Sizing Rationale:
    --------------------------------
    Different pipeline stages have different resource characteristics:

    ENCODER:
    - Fast execution, lower memory needs
    - Can tolerate more backpressure
    - Config: num_slots = concurrent * 1.5, slot_size = 16MB

    DENOISING:
    - Slowest stage (bottleneck)
    - Needs more buffering to absorb backpressure
    - Long execution time per request
    - Config: num_slots = concurrent * 3, slot_size = 32MB

    DECODER:
    - Similar to Encoder in resource needs
    - Config: num_slots = concurrent, slot_size = 16MB

    Args:
        role: The diffusion role (ENCODER, DENOISING, DECODER, or NULL)
        concurrent_requests: Expected number of concurrent requests.
            This determines the number of buffer slots to pre-allocate.

    Returns:
        Tuple of (num_slots, slot_size_bytes):
        - num_slots: Number of buffer slots to allocate
        - slot_size_bytes: Size of each slot in bytes

    Example:
        >>> get_slot_config(DiffusionRole.ENCODER, 4)
        (6, 16777216)  # 6 slots, 16MB each
        >>> get_slot_config(DiffusionRole.DENOISING, 4)
        (12, 33554432)  # 12 slots, 32MB each
        >>> get_slot_config(DiffusionRole.DECODER, 4)
        (4, 16777216)  # 4 slots, 16MB each

    Reference: RFC §3.2 "Role-Specific Sizing"
    """
    # Slot sizes in bytes (16MB, 32MB, 64MB for alignment)
    ENCODER_SLOT_SIZE = 16 * 1024 * 1024  # 16MB
    DENOISING_SLOT_SIZE = 32 * 1024 * 1024  # 32MB
    DECODER_SLOT_SIZE = 16 * 1024 * 1024  # 16MB

    if role == DiffusionRole.ENCODER:
        # Encoder: output is smaller, lower backpressure
        # Add 50% extra slots for flexibility
        num_slots = int(concurrent_requests * 1.5)
        slot_size = ENCODER_SLOT_SIZE
    elif role == DiffusionRole.DENOISING:
        # Denoising: needs more buffering, longer execution time
        # Triple slots to handle backpressure from slow processing
        num_slots = concurrent_requests * 3
        slot_size = DENOISING_SLOT_SIZE
    elif role == DiffusionRole.DECODER:
        # Decoder: input similar to encoder output
        num_slots = concurrent_requests
        slot_size = DECODER_SLOT_SIZE
    else:
        # Default fallback for NULL role
        num_slots = concurrent_requests
        slot_size = ENCODER_SLOT_SIZE

    return num_slots, slot_size


def group_concurrent_contiguous(
    src_indices: "List[int]", dst_indices: "List[int]"
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Group contiguous indices to reduce transfer overhead.

    Problem:
    --------
    When transferring tensor slices, we may have many small, non-contiguous
    indices. Each RDMA operation has overhead, so combining contiguous
    indices into larger transfers improves efficiency.

    Algorithm:
    -----------
    1. Calculate differences between consecutive indices
    2. Find breakpoints where diff != 1 (not consecutive)
    3. Split at breakpoints into groups
    4. Convert each group to (start, length) format

    Example:
    --------
    Input:
        src_indices = [0, 1, 2, 5, 6, 7, 10]
        dst_indices = [0, 1, 2, 5, 6, 7, 10]

    Process:
        diff = [1, 1, 3, 1, 1, 3]
        breakpoints = [3, 6]  # positions 3 and 6
        groups = [[0,1,2], [5,6,7], [10]]

    Output:
        src_result = [(0, 3), (5, 3), (10, 1)]
        dst_result = [(0, 3), (5, 3), (10, 1)]

    Benefit:
    --------
    7 separate transfers -> 3 batched transfers

    Args:
        src_indices: List of source indices to group.
            Must be same length as dst_indices.
        dst_indices: List of destination indices to group.
            Must be same length as src_indices.

    Returns:
        Tuple of (src_groups, dst_groups):
        - src_groups: List of (start, length) tuples for source
        - dst_groups: List of (start, length) tuples for destination

    Example:
        >>> src, dst = group_concurrent_contiguous([0,1,2,5,6], [0,1,2,5,6])
        >>> src
        [(0, 3), (5, 2)]
        >>> dst
        [(0, 3), (5, 2)]

    Reference: sglang/srt/disaggregation/common/utils.py:group_concurrent_contiguous
    """
    if not src_indices or not dst_indices:
        return [], []

    import numpy as np

    src_arr = np.array(src_indices, dtype=np.int32)
    dst_arr = np.array(dst_indices, dtype=np.int32)

    # Find breakpoints where indices are not consecutive
    src_diff = np.diff(src_arr)
    dst_diff = np.diff(dst_arr)

    # Break where either src or dst is not consecutive
    brk = np.where((src_diff != 1) | (dst_diff != 1))[0] + 1

    src_groups = np.split(src_arr, brk)
    dst_groups = np.split(dst_arr, brk)

    # Convert to (start, length) format
    src_result = [(int(g[0]), len(g)) for g in src_groups if len(g) > 0]
    dst_result = [(int(g[0]), len(g)) for g in dst_groups if len(g) > 0]

    return src_result, dst_result
