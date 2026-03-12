"""TransferBuffer implementations for Diffusion PD Disaggregation.

This module provides TransferTensorBuffer and TransferMetaBuffer for managing
latent tensor and metadata transfer between disaggregated diffusion stages.

Design Principles:
-----------------
1. Buddy Allocation: Uses BuddyAllocator for dynamic memory allocation with
   support for varying latent sizes based on resolution and frame count.
2. Role-Based Sizing: Different pipeline stages have different buffer requirements.
3. Thread Safety: All buffer operations are protected by locks for concurrent access.

File Structure:
-------------
- TransferPoll: Status enumeration for transfer lifecycle
- DiffusionRole: Role enumeration for pipeline stages
- calculate_latent_size: Compute latent tensor memory requirements
- get_slot_config: Get role-specific buffer configuration
- TransferTensorBufferSlot: Dataclass for allocated tensor slot info
- TransferMetaBufferSlot: Dataclass for allocated metadata slot info
- TransferTensorBuffer: Buffer for latent tensor transfer
- TransferMetaBuffer: Buffer for metadata transfer

Reference:
- RFC SGLang Diffusion Disaggregation §3: TransferBuffer
- RFC §3.2: Role-Specific Sizing
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.multimodal_gen.runtime.cache.allocator import (
    BuddyAllocator,
    create_buddy_allocator,
)

logger = logging.getLogger(__name__)


# Transfer status polling states (same as TransferPoll in utils)
class TransferPoll(IntEnum):
    """Transfer status polling states.

    Represents the lifecycle of a tensor transfer:

    State Transition Diagram:
    ------------------------
    Bootstrapping -> WaitingForInput -> Transferring -> Success
                            |                    |
                            |                    +---> Failed
                            +-------------------> Failed

    State Descriptions:
    -------------------
    - Bootstrapping (value=1): Initializing connection and buffers
    - WaitingForInput (value=2): Waiting for data to be ready at sender
    - Transferring (value=3): Actively transferring data over network
    - Success (value=4): Transfer completed successfully (terminal)
    - Failed (value=0): Transfer failed (terminal)
    """

    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


# Role types in Diffusion PD disaggregation
class DiffusionRole(IntEnum):
    """Role types in Diffusion PD disaggregation.

    Three stages in the diffusion pipeline that can be disaggregated:

    1. ENCODER (value=0):
       - Text encoding: Converts prompt to embeddings
       - Conditioning: Prepares conditioning signals (CLIP, T2I adapter, etc.)
       - Latent preparation: Generates initial noise latent

       Resource profile:
       - Fast execution (shortest time)
       - Lowest memory footprint
       - Can use aggressive parallelization

    2. DENOISING (value=1):
       - DiT denoising: Multi-step iterative denoising
       - Largest compute and memory requirements
       - Longest execution time (bottleneck)

       Resource profile:
       - Slowest execution (bottleneck stage)
       - Highest memory usage
       - Benefits from more buffer slots for backpressure

    3. DECODER (value=2):
       - VAE decoding: Converts latent to final image/video
       - Medium execution time

       Resource profile:
       - Medium execution time
       - Moderate memory footprint
       - Similar buffer needs to Encoder

    4. NULL (value=3):
       - Non-disaggregated mode
       - All stages run on same node
       - No network transfer needed
    """

    ENCODER = 0
    DENOISING = 1
    DECODER = 2
    NULL = 3


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
    """
    latent_h = height // 8
    latent_w = width // 8
    latent_t = (num_frames - 1) // 4 + 1
    return latent_channels * latent_t * latent_h * latent_w * dtype_size


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


# Metadata slot size (fixed, 256 bytes, 64-byte aligned)
META_SLOT_SIZE = 256


@dataclass
class TransferTensorBufferSlot:
    """Represents an allocated tensor buffer slot.

    Contains information about an allocated memory region for latent tensor transfer.
    This dataclass is returned by TransferTensorBuffer.allocate_for_request() and
    contains all the information needed for RDMA transfer.

    Attributes:
        slot_index: Index into the buddy allocator's slot table.
            Used to identify and free this allocation.
        num_slots: Number of buddy slots used.
            Since buddy allocation rounds up to power-of-2, a single request
            may use multiple slots.
        ptr: Memory pointer (GPU address for RDMA).
            The base address where the latent tensor data should be written/read.
        size_bytes: Total size of the allocated region in bytes.
            Calculated from resolution, frame count, and latent channels.
        shape: Shape of the latent tensor.
            Format: (batch, channels, temporal, height, width)
        resolution: Original (width, height) of the input.
            Used for debugging and metadata tracking.
        num_frames: Number of frames in the input video.
            Used for temporal dimension calculation.
        dtype: Data type of the tensor.
            Typically torch.float16 or torch.bfloat16.

    Example:
        >>> slot = TransferTensorBufferSlot(
        ...     slot_index=0,
        ...     num_slots=1,
        ...     ptr=0x7f8a1000000,
        ...     size_bytes=4194304,
        ...     shape=(1, 16, 21, 60, 104),
        ...     resolution=(832, 480),
        ...     num_frames=81,
        ...     dtype=torch.float16,
        ... )
    """

    slot_index: int
    num_slots: int  # Number of buddy slots used
    ptr: int  # Memory pointer (GPU address for RDMA)
    size_bytes: int
    shape: Tuple[int, ...]
    resolution: Tuple[int, int]  # (width, height)
    num_frames: int
    dtype: torch.dtype


@dataclass
class TransferMetaBufferSlot:
    """Represents an allocated metadata buffer slot.

    Contains transfer metadata for a request. Metadata includes resolution,
    frame count, transfer status, and bootstrap room identifier.

    Attributes:
        slot_index: Index into the metadata buffer.
            Used to identify and free this slot.
        ptr: Memory pointer to the metadata region.
            CPU pinned memory address for metadata transfer.
        request_id: Request identifier.
            Original request ID from the scheduler.
        resolution: (width, height) of the input.
            Original input resolution for this transfer.
        num_frames: Number of frames.
            Original frame count for this transfer.
        status: Current transfer status from TransferPoll.
            Updated throughout the transfer lifecycle.
        bootstrap_room: Bootstrap room identifier.
            Session ID for this transfer operation.

    Memory Layout:
        ------------
        The metadata is packed in a fixed-size slot (256 bytes):
        - request_id: 8 bytes (int64)
        - width: 4 bytes (int32)
        - height: 4 bytes (int32)
        - num_frames: 4 bytes (int32)
        - status: 4 bytes (int32)
        - bootstrap_room: 4 bytes (int32)
        - padding: 228 bytes (reserved)

    Example:
        >>> slot = TransferMetaBufferSlot(
        ...     slot_index=0,
        ...     ptr=0x7f8a2000000,
        ...     request_id=12345,
        ...     resolution=(1920, 1080),
        ...     num_frames=81,
        ...     status=TransferPoll.Success,
        ...     bootstrap_room=100,
        ... )
    """

    slot_index: int
    ptr: int  # Memory pointer
    request_id: int
    resolution: Tuple[int, int]
    num_frames: int
    status: int = TransferPoll.Bootstrapping
    bootstrap_room: int = 0


class TransferTensorBuffer:
    """Buffer for transferring latent tensors.

    Uses BuddyAllocator for dynamic memory allocation with support for
    varying latent sizes based on resolution and frame count.

    Design Rationale:
    ----------------
    The TransferTensorBuffer provides:

    1. Dynamic Allocation:
       - Uses buddy allocator to handle variable-sized latent tensors
       - Supports requests of different resolutions and frame counts
       - Automatic splitting and coalescing of memory blocks

    2. Role-Based Sizing:
       - Different roles have different buffer requirements
       - Denoising needs more slots (bottleneck stage)
       - Encoder/Decoder need fewer slots

    3. RDMA-Ready:
       - Allocates pinned memory for efficient RDMA transfers
       - Provides memory pointers for direct hardware access
       - Thread-safe allocation for concurrent requests

    Attributes:
        role: The diffusion role (ENCODER, DENOISING, or DECODER).
            Determines slot configuration via get_slot_config().
        allocator: The buddy allocator managing memory.
            Handles splitting, allocation, and coalescing.
        storage: The underlying memory buffer.
            Pinned CPU memory for RDMA transfer.
        _allocated_slots: Dict mapping slot_index -> TransferTensorBufferSlot.
            Tracks currently allocated slots.

    Reference: RFC §3.3 "TensorBuffer"
    """

    def __init__(
        self,
        role: DiffusionRole,
        concurrent_requests: int = 4,
        latent_channels: int = 16,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize the tensor buffer.

        Args:
            role: The diffusion role (ENCODER, DENOISING, or DECODER).
                Used to determine slot configuration.
            concurrent_requests: Expected concurrent requests.
                Determines number of buffer slots to pre-allocate.
            latent_channels: Number of latent channels.
                Default 16 for most diffusion models.
            dtype: Data type for tensors.
                torch.float16 or torch.bfloat16 typically.

        Example:
            >>> buffer = TransferTensorBuffer(
            ...     role=DiffusionRole.DENOISING,
            ...     concurrent_requests=4,
            ...     latent_channels=16,
            ... )
            >>> print(f"Free slots: {buffer.get_free_slots_count()}")
            Free slots: 12
        """
        self.role = role
        self.concurrent_requests = concurrent_requests
        self.latent_channels = latent_channels
        self.dtype = dtype
        self.dtype_size = 2 if dtype == torch.float16 else 4  # fp16=2, bf16=4

        # Get role-specific slot configuration
        num_slots, slot_size = get_slot_config(role, concurrent_requests)

        # Create buddy allocator (it creates the pinned memory pool internally)
        self.allocator = create_buddy_allocator(
            num_slots=num_slots,
            slot_size=slot_size,
        )

        # Use the allocator's pool directly for tensor views
        self.storage = self.allocator.pool

        # Track allocated slots
        self._allocated_slots: Dict[int, TransferTensorBufferSlot] = {}
        self._lock = threading.Lock()

        logger.info(
            f"TransferTensorBuffer initialized for role {role.value}: "
            f"num_slots={num_slots}, slot_size={slot_size} bytes"
        )

    def allocate_for_request(
        self,
        resolution: Tuple[int, int],
        num_frames: int,
        request_id: Optional[int] = None,
    ) -> Optional[TransferTensorBufferSlot]:
        """Allocate buffer for a request.

        Args:
            resolution: (width, height) of the input
            num_frames: Number of frames
            request_id: Optional request ID for tracking

        Returns:
            TransferTensorBufferSlot if successful, None if insufficient memory

        Reference: RFC §3.2 "Role-Specific Sizing"
        """
        width, height = resolution

        # Calculate required size
        size_bytes = calculate_latent_size(
            height=height,
            width=width,
            num_frames=num_frames,
            latent_channels=self.latent_channels,
            dtype_size=self.dtype_size,
        )

        with self._lock:
            # Allocate from buddy allocator
            alloc_slot = self.allocator.allocate(size_bytes, request_id)

            if alloc_slot is None:
                logger.warning(
                    f"TransferTensorBuffer: failed to allocate {size_bytes} bytes "
                    f"for request {request_id}, resolution={resolution}, frames={num_frames}"
                )
                return None

            # Create the tensor buffer slot
            latent_shape = (
                1,  # batch
                self.latent_channels,
                (num_frames - 1) // 4 + 1,  # t
                height // 8,  # h
                width // 8,  # w
            )

            # Get the memory pointer
            ptr = self.allocator.get_ptr(alloc_slot.slot_index)

            tensor_slot = TransferTensorBufferSlot(
                slot_index=alloc_slot.slot_index,
                num_slots=alloc_slot.size // (self.allocator.slot_size),
                ptr=ptr,
                size_bytes=size_bytes,
                shape=latent_shape,
                resolution=resolution,
                num_frames=num_frames,
                dtype=self.dtype,
            )

            self._allocated_slots[alloc_slot.slot_index] = tensor_slot

            logger.debug(
                f"TransferTensorBuffer: allocated slot {alloc_slot.slot_index} "
                f"for request {request_id}, size={size_bytes} bytes"
            )

            return tensor_slot

    def free(self, slot_index: int) -> bool:
        """Free an allocated slot.

        Args:
            slot_index: Index of the slot to free

        Returns:
            True if successful
        """
        with self._lock:
            if slot_index in self._allocated_slots:
                del self._allocated_slots[slot_index]

            return self.allocator.free(slot_index)

    def get_ptr(self, slot_index: int) -> Optional[int]:
        """Get the memory pointer for a slot.

        Args:
            slot_index: Index of the slot

        Returns:
            Memory pointer, or None if not found
        """
        return self.allocator.get_ptr(slot_index)

    def get_tensor_view(self, slot_index: int) -> Optional[torch.Tensor]:
        """Get a tensor view of the allocated slot.

        Note: This returns a view of the raw bytes. For actual use with different
        dtypes, the caller should reinterpret the bytes appropriately.

        Args:
            slot_index: Index of the slot

        Returns:
            Tensor view, or None if not found
        """
        with self._lock:
            if slot_index not in self._allocated_slots:
                return None

            slot = self._allocated_slots[slot_index]
            # The pool is uint8, calculate offset directly
            offset = slot.ptr - self.allocator.pool_ptr
            # Return view of the bytes (caller interprets based on dtype)
            return self.storage[offset : offset + slot.size_bytes]

    def get_free_slots_count(self) -> int:
        """Get the number of free slots.

        Returns:
            Number of available slots
        """
        return self.allocator.get_free_slots_count()

    def get_stats(self) -> Dict:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        stats = self.allocator.get_stats()
        stats["role"] = self.role.value
        return stats


class TransferMetaBuffer:
    """Buffer for transferring metadata.

    Stores lightweight, non-tensor metadata for transfer requests,
    such as resolution, frame count, and transfer status.

    Reference: RFC §3.1 "TransferMetaBuffer"
    """

    def __init__(
        self,
        num_slots: int = 256,
    ) -> None:
        """Initialize the metadata buffer.

        Args:
            num_slots: Maximum number of concurrent metadata slots
        """
        self.num_slots = num_slots
        self.slot_size = META_SLOT_SIZE

        # Allocate pinned memory for metadata
        self.storage = torch.empty(
            num_slots * self.slot_size,
            dtype=torch.uint8,
            pin_memory=True,
        )
        self.storage_ptr = self.storage.data_ptr()

        # Track allocated slots
        self._allocated_slots: Dict[int, TransferMetaBufferSlot] = {}
        self._next_slot_index = 0
        self._lock = threading.Lock()

        logger.info(
            f"TransferMetaBuffer initialized: num_slots={num_slots}, "
            f"slot_size={self.slot_size} bytes"
        )

    def allocate(
        self,
        request_id: int,
        resolution: Tuple[int, int],
        num_frames: int,
    ) -> TransferMetaBufferSlot:
        """Allocate a metadata slot for a request.

        Args:
            request_id: Request identifier
            resolution: (width, height) of the input
            num_frames: Number of frames

        Returns:
            TransferMetaBufferSlot
        """
        with self._lock:
            if self._next_slot_index >= self.num_slots:
                raise RuntimeError(
                    f"TransferMetaBuffer: no more slots available "
                    f"(max={self.num_slots})"
                )

            slot_index = self._next_slot_index
            self._next_slot_index += 1

            ptr = self.storage_ptr + slot_index * self.slot_size

            slot = TransferMetaBufferSlot(
                slot_index=slot_index,
                ptr=ptr,
                request_id=request_id,
                resolution=resolution,
                num_frames=num_frames,
                status=TransferPoll.Bootstrapping,
            )

            self._allocated_slots[slot_index] = slot

            # Initialize metadata in storage
            self._write_metadata(slot)

            logger.debug(
                f"TransferMetaBuffer: allocated slot {slot_index} "
                f"for request {request_id}"
            )

            return slot

    def free(self, slot_index: int) -> bool:
        """Free a metadata slot.

        Args:
            slot_index: Index of the slot to free

        Returns:
            True if successful
        """
        with self._lock:
            if slot_index in self._allocated_slots:
                del self._allocated_slots[slot_index]
                logger.debug(f"TransferMetaBuffer: freed slot {slot_index}")
                return True
            return False

    def get_slot(self, slot_index: int) -> Optional[TransferMetaBufferSlot]:
        """Get a metadata slot.

        Args:
            slot_index: Index of the slot

        Returns:
            TransferMetaBufferSlot, or None if not found
        """
        with self._lock:
            return self._allocated_slots.get(slot_index)

    def update_status(self, slot_index: int, status: int) -> bool:
        """Update transfer status for a slot.

        Args:
            slot_index: Index of the slot
            status: New status from TransferPoll

        Returns:
            True if successful
        """
        with self._lock:
            if slot_index not in self._allocated_slots:
                return False
            slot = self._allocated_slots[slot_index]
            slot.status = status
            self._write_metadata(slot)
            return True

    def _write_metadata(self, slot: TransferMetaBufferSlot) -> None:
        """Write metadata to storage.

        Layout: request_id(8) | width(4) | height(4) | num_frames(4) |
                status(4) | bootstrap_room(4) | padding(228)
        """
        import struct

        offset = slot.slot_index * self.slot_size

        # Pack metadata
        data = struct.pack(
            "qiiiii",
            slot.request_id,
            slot.resolution[0],
            slot.resolution[1],
            slot.num_frames,
            int(slot.status),
            slot.bootstrap_room,
        )

        # Write to storage
        data_array = torch.from_numpy(np.frombuffer(data, dtype=np.uint8))
        self.storage[offset : offset + len(data)].copy_(data_array)

    def get_free_slots_count(self) -> int:
        """Get the number of free slots.

        Returns:
            Number of available slots
        """
        with self._lock:
            return self.num_slots - self._next_slot_index
