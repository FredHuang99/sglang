"""BuddyAllocator for dynamic memory allocation in Diffusion PD Disaggregation.

This module implements a buddy memory allocator for managing pinned memory
buffers used in latent tensor transfer between disaggregated diffusion stages.

Design Principles:
-----------------
1. Buddy Algorithm: Uses binary splitting for efficient memory allocation.
2. Power-of-Two Alignment: All allocations aligned to 2^N boundaries for RDMA.
3. Coalescing: Merges adjacent free blocks to reduce fragmentation.
4. Thread Safety: All operations protected by locks for concurrent access.

File Structure:
-------------
- AllocSlot: Dataclass representing an allocated memory slot
- BuddyAllocator: Main allocator class with split/aggregate/coalesce
- create_buddy_allocator: Factory function for allocator creation

Reference:
- RFC SGLang Diffusion Disaggregation §3.2: Dynamic Allocation (Split/Aggregate)
- RFC §3.2: Defragmentation (Coalesce)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class AllocSlot:
    """Represents an allocated memory slot.

    Attributes:
        slot_index: Unique identifier for this allocation.
            Used to reference the slot in free() and get_ptr().
        start: Start offset in bytes from the pool base.
            Used to compute actual memory pointer.
        size: Size of the allocation in bytes.
            May be larger than requested due to power-of-2 alignment.
        allocated: Whether this slot is currently allocated.
            False indicates the slot is free.
        request_id: Optional request identifier.
            Used for tracking which request owns this slot.

    Example:
        >>> slot = AllocSlot(
        ...     slot_index=0,
        ...     start=0,
        ...     size=16777216,  # 16MB
        ...     allocated=True,
        ...     request_id=12345,
        ... )
    """

    slot_index: int
    start: int  # Start offset in bytes
    size: int  # Size in bytes
    allocated: bool = False
    request_id: Optional[int] = None


class BuddyAllocator:
    """Buddy allocator for pinned memory management.

    Implements a buddy memory allocation algorithm with support for:
    - Dynamic allocation with 2^N alignment
    - Split: Dividing larger blocks for smaller allocations
    - Aggregate (Coalesce): Merging adjacent free blocks

    This allocator is designed for RDMA transfers which require
    physically contiguous memory.

    Design Rationale:
    ----------------
    Why Buddy Algorithm for RDMA?
    1. Power-of-Two Alignment:
       - RDMA hardware requires memory to be aligned to specific boundaries
       - Buddy algorithm naturally provides this alignment

    2. Fast Allocation/Deallocation:
       - O(log N) allocation by finding best-fit block
       - O(1) deallocation with immediate coalescing

    3. Reduced Fragmentation:
       - Coalescing adjacent free blocks prevents fragmentation
       - Splitting only when necessary optimizes memory usage

    Algorithm Details:
    -----------------
    Split (Allocation):
    1. Find smallest free block that fits the request
    2. If block is larger than needed, split it in half
    3. Repeat until block size matches (or barely exceeds) request
    4. Mark as allocated, return slot

    Coalesce (Deallocation):
    1. Add freed block to free list
    2. Check if adjacent blocks are also free
    3. If yes, merge them into a larger block
    4. Repeat until no more merging possible

    Attributes:
        pool: The pinned memory buffer.
            Pre-allocated CPU memory for RDMA transfer.
        pool_ptr: Base memory pointer.
            Used to compute absolute addresses.
        _free_blocks: List of (start, size) tuples.
            Tracks all currently free memory regions.
        _allocated_slots: Dict mapping slot_index -> AllocSlot.
            Tracks all allocated slots.

    Reference: RFC §3.2 "Dynamic Allocation (Split/Aggregate)"
    """

    def __init__(
        self,
        total_size: int,
        slot_size: int,
        num_slots: int,
    ) -> None:
        """Initialize the buddy allocator.

        Args:
            total_size: Total pool size in bytes.
                Computed as num_slots * slot_size.
            slot_size: Size of each slot in bytes.
                Should be power of 2 for efficient buddy allocation.
            num_slots: Number of slots in the pool.
                Determines total memory: num_slots * slot_size.

        Example:
            >>> # 4 slots of 16MB each = 64MB total
            >>> allocator = BuddyAllocator(
            ...     total_size=67108864,
            ...     slot_size=16777216,
            ...     num_slots=4,
            ... )
        """
        self.total_size = total_size
        self.slot_size = slot_size
        self.num_slots = num_slots

        # Allocate the pinned memory pool
        self.pool = torch.empty(
            total_size,
            dtype=torch.uint8,
            pin_memory=True,
        )
        self.pool_ptr = self.pool.data_ptr()

        # Initialize free list (list of free blocks, each with start offset and size)
        # Initially, the entire pool is one free block
        self._free_blocks: List[Tuple[int, int]] = [(0, total_size)]
        self._allocated_slots: Dict[int, AllocSlot] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

        # Statistics
        self._alloc_count = 0
        self._free_count = 0

        logger.info(
            f"BuddyAllocator initialized: total_size={total_size} bytes, "
            f"slot_size={slot_size} bytes, num_slots={num_slots}"
        )

    def allocate(
        self, size: int, request_id: Optional[int] = None
    ) -> Optional[AllocSlot]:
        """Allocate memory from the pool.

        Uses buddy algorithm: finds the smallest free block that can fit,
        splitting larger blocks as needed.

        Algorithm:
        ---------
        1. Align size to next power of 2
        2. Find smallest free block that fits (best-fit)
        3. Split block if larger than needed
        4. Return allocated slot

        Args:
            size: Size in bytes to allocate.
                Will be aligned to next power of 2.
            request_id: Optional request ID for tracking.
                Used for debugging and memory accounting.

        Returns:
            AllocSlot if successful, None if insufficient memory.

        Example:
            >>> slot = allocator.allocate(1024, request_id=123)
            >>> print(f"Allocated at {slot.start}, size {slot.size}")
            Allocated at 0, size 2048

        Reference: RFC §3.2 "Dynamic Allocation (Split/Aggregate)"
        """
        with self._lock:
            # Align size to power of 2 for buddy allocation
            aligned_size = self._next_power_of_two(size)

            # Find the best fitting free block
            best_block_idx = -1
            best_block_size = float("inf")

            for i, (start, block_size) in enumerate(self._free_blocks):
                if block_size >= aligned_size and block_size < best_block_size:
                    best_block_idx = i
                    best_block_size = block_size

            if best_block_idx == -1:
                logger.warning(
                    f"BuddyAllocator: failed to allocate {size} bytes "
                    f"(aligned to {aligned_size}), insufficient memory"
                )
                return None

            # Get the block and potentially split it
            start, block_size = self._free_blocks.pop(best_block_idx)

            # Calculate how many slots this block represents
            slots_needed = (aligned_size + self.slot_size - 1) // self.slot_size
            allocated_size = slots_needed * self.slot_size

            # Split if the block is larger than needed
            remaining_size = block_size - allocated_size

            if remaining_size > 0:
                # Add remaining as a smaller free block
                # Try to merge with adjacent free blocks
                self._add_free_block(start + allocated_size, remaining_size)

            # Create the allocated slot
            slot_index = self._alloc_count
            self._alloc_count += 1

            alloc_slot = AllocSlot(
                slot_index=slot_index,
                start=start,
                size=allocated_size,
                allocated=True,
                request_id=request_id,
            )

            self._allocated_slots[slot_index] = alloc_slot

            logger.debug(
                f"BuddyAllocator: allocated {allocated_size} bytes "
                f"(slot {slot_index}, request_id={request_id})"
            )

            return alloc_slot

    def free(self, slot_index: int) -> bool:
        """Free an allocated slot.

        Uses coalesce algorithm to merge adjacent free blocks.

        Args:
            slot_index: Index of the slot to free

        Returns:
            True if successful, False if slot not found
        """
        with self._lock:
            if slot_index not in self._allocated_slots:
                logger.warning(f"BuddyAllocator: slot {slot_index} not found")
                return False

            slot = self._allocated_slots[slot_index]
            if not slot.allocated:
                logger.warning(f"BuddyAllocator: slot {slot_index} already freed")
                return False

            # Mark as freed
            slot.allocated = False
            slot.request_id = None

            # Add to free list and coalesce
            self._add_free_block(slot.start, slot.size)

            self._free_count += 1

            logger.debug(f"BuddyAllocator: freed slot {slot_index}")

            return True

    def _add_free_block(self, start: int, size: int) -> None:
        """Add a free block and attempt to coalesce with adjacent blocks.

        Reference: RFC §3.2 "Defragmentation (Coalesce)"
        """
        if size <= 0:
            return

        # Try to merge with existing free blocks
        merged = True
        current_start = start
        current_size = size

        while merged:
            merged = False
            new_free_blocks = []

            for i, (block_start, block_size) in enumerate(self._free_blocks):
                # Check if adjacent (before or after)
                if block_start + block_size == current_start:
                    # Merge: block is immediately before current
                    current_start = block_start
                    current_size += block_size
                    merged = True
                elif current_start + current_size == block_start:
                    # Merge: block is immediately after current
                    current_size += block_size
                    merged = True
                else:
                    new_free_blocks.append((block_start, block_size))

            if merged:
                new_free_blocks.append((current_start, current_size))
                self._free_blocks = new_free_blocks
            else:
                # No more merging possible, add the block
                self._free_blocks.append((current_start, current_size))

    def get_ptr(self, slot_index: int) -> Optional[int]:
        """Get the memory pointer for a slot.

        Args:
            slot_index: Index of the slot

        Returns:
            Memory pointer (offset from pool base), or None if not found
        """
        with self._lock:
            if slot_index not in self._allocated_slots:
                return None
            slot = self._allocated_slots[slot_index]
            if not slot.allocated:
                return None
            return self.pool_ptr + slot.start

    def get_free_slots_count(self) -> int:
        """Get the number of free slots.

        Returns:
            Number of available slots
        """
        with self._lock:
            # Approximate by counting free bytes / slot_size
            free_bytes = sum(size for _, size in self._free_blocks)
            return free_bytes // self.slot_size

    def get_stats(self) -> Dict[str, int]:
        """Get allocator statistics.

        Returns:
            Dictionary with allocation statistics
        """
        with self._lock:
            return {
                "total_size": self.total_size,
                "slot_size": self.slot_size,
                "num_slots": self.num_slots,
                "alloc_count": self._alloc_count,
                "free_count": self._free_count,
                "allocated_count": len(
                    [s for s in self._allocated_slots.values() if s.allocated]
                ),
                "free_bytes": sum(size for _, size in self._free_blocks),
            }

    @staticmethod
    def _next_power_of_two(n: int) -> int:
        """Get the next power of two >= n."""
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()


def create_buddy_allocator(
    num_slots: int,
    slot_size: int,
) -> BuddyAllocator:
    """Create a BuddyAllocator with given configuration.

    Args:
        num_slots: Number of slots
        slot_size: Size of each slot in bytes

    Returns:
        Configured BuddyAllocator instance
    """
    total_size = num_slots * slot_size
    return BuddyAllocator(
        total_size=total_size,
        slot_size=slot_size,
        num_slots=num_slots,
    )
