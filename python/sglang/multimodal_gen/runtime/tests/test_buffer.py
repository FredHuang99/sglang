"""Unit tests for TransferBuffer.

Tests the TransferTensorBuffer and TransferMetaBuffer classes
used in Diffusion PD disaggregation.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.cache.buffer import (
    DiffusionRole,
    TransferMetaBuffer,
    TransferPoll,
    TransferTensorBuffer,
)


class TestTransferTensorBuffer:
    """Test cases for TransferTensorBuffer."""

    def test_encoder_buffer_creation(self):
        """Test creating a buffer for Encoder role."""
        buffer = TransferTensorBuffer(
            role=DiffusionRole.ENCODER,
            concurrent_requests=4,
            latent_channels=16,
            dtype=torch.float16,
        )

        assert buffer.role == DiffusionRole.ENCODER
        assert buffer.latent_channels == 16
        assert buffer.dtype == torch.float16
        assert buffer.allocator is not None

    def test_denoising_buffer_creation(self):
        """Test creating a buffer for Denoising role."""
        buffer = TransferTensorBuffer(
            role=DiffusionRole.DENOISING,
            concurrent_requests=4,
            latent_channels=16,
            dtype=torch.float16,
        )

        assert buffer.role == DiffusionRole.DENOISING

    def test_decoder_buffer_creation(self):
        """Test creating a buffer for Decoder role."""
        buffer = TransferTensorBuffer(
            role=DiffusionRole.DECODER,
            concurrent_requests=4,
            latent_channels=16,
            dtype=torch.float16,
        )

        assert buffer.role == DiffusionRole.DECODER

    def test_allocate_for_request(self):
        """Test allocating buffer for a request."""
        buffer = TransferTensorBuffer(
            role=DiffusionRole.ENCODER,
            concurrent_requests=4,
        )

        # Allocate for 480p 5 second video
        slot = buffer.allocate_for_request(
            resolution=(832, 480),
            num_frames=81,
            request_id=1,
        )

        assert slot is not None
        assert slot.resolution == (832, 480)
        assert slot.num_frames == 81
        assert slot.ptr is not None

    def test_free_slot(self):
        """Test freeing an allocated slot."""
        buffer = TransferTensorBuffer(
            role=DiffusionRole.ENCODER,
            concurrent_requests=4,
        )

        # Allocate
        slot = buffer.allocate_for_request(
            resolution=(832, 480),
            num_frames=81,
            request_id=1,
        )
        assert slot is not None

        # Free
        result = buffer.free(slot.slot_index)
        assert result is True

    def test_multiple_allocations(self):
        """Test multiple allocations."""
        buffer = TransferTensorBuffer(
            role=DiffusionRole.ENCODER,
            concurrent_requests=4,
        )

        # Allocate for multiple requests
        slot1 = buffer.allocate_for_request(
            resolution=(832, 480),
            num_frames=81,
            request_id=1,
        )
        slot2 = buffer.allocate_for_request(
            resolution=(1280, 720),
            num_frames=161,
            request_id=2,
        )

        assert slot1 is not None
        assert slot2 is not None

        # Free one and allocate again
        buffer.free(slot1.slot_index)

        slot3 = buffer.allocate_for_request(
            resolution=(832, 480),
            num_frames=81,
            request_id=3,
        )
        assert slot3 is not None

    def test_tensor_view(self):
        """Test getting tensor view of allocated slot."""
        buffer = TransferTensorBuffer(
            role=DiffusionRole.ENCODER,
            concurrent_requests=4,
            latent_channels=16,
        )

        slot = buffer.allocate_for_request(
            resolution=(832, 480),
            num_frames=81,
            request_id=1,
        )

        tensor = buffer.get_tensor_view(slot.slot_index)
        assert tensor is not None
        # The view returns raw bytes (uint8), size should match
        assert tensor.numel() == slot.size_bytes

    def test_free_slots_count(self):
        """Test getting free slots count."""
        buffer = TransferTensorBuffer(
            role=DiffusionRole.ENCODER,
            concurrent_requests=4,
        )

        initial_free = buffer.get_free_slots_count()
        assert initial_free > 0

        # Allocate
        slot = buffer.allocate_for_request(
            resolution=(832, 480),
            num_frames=81,
            request_id=1,
        )
        assert slot is not None

        free_after = buffer.get_free_slots_count()
        assert free_after < initial_free

    def test_stats(self):
        """Test getting buffer statistics."""
        buffer = TransferTensorBuffer(
            role=DiffusionRole.ENCODER,
            concurrent_requests=4,
        )

        stats = buffer.get_stats()
        assert "role" in stats
        assert stats["role"] == DiffusionRole.ENCODER
        assert "total_size" in stats


class TestTransferMetaBuffer:
    """Test cases for TransferMetaBuffer."""

    def test_creation(self):
        """Test creating a metadata buffer."""
        buffer = TransferMetaBuffer(num_slots=256)

        assert buffer.num_slots == 256
        assert buffer.slot_size == 256
        assert buffer.storage is not None

    def test_allocate(self):
        """Test allocating a metadata slot."""
        buffer = TransferMetaBuffer(num_slots=256)

        slot = buffer.allocate(
            request_id=1,
            resolution=(832, 480),
            num_frames=81,
        )

        assert slot is not None
        assert slot.request_id == 1
        assert slot.resolution == (832, 480)
        assert slot.num_frames == 81
        assert slot.status == TransferPoll.Bootstrapping
        assert slot.bootstrap_room == 0

    def test_multiple_allocations(self):
        """Test multiple metadata allocations."""
        buffer = TransferMetaBuffer(num_slots=10)

        slots = []
        for i in range(10):
            slot = buffer.allocate(
                request_id=i,
                resolution=(832, 480),
                num_frames=81,
            )
            assert slot is not None
            slots.append(slot)

        # 11th allocation should fail
        with pytest.raises(RuntimeError):
            buffer.allocate(
                request_id=10,
                resolution=(832, 480),
                num_frames=81,
            )

    def test_free(self):
        """Test freeing a metadata slot."""
        buffer = TransferMetaBuffer(num_slots=256)

        slot = buffer.allocate(
            request_id=1,
            resolution=(832, 480),
            num_frames=81,
        )

        result = buffer.free(slot.slot_index)
        assert result is True

    def test_update_status(self):
        """Test updating transfer status."""
        buffer = TransferMetaBuffer(num_slots=256)

        slot = buffer.allocate(
            request_id=1,
            resolution=(832, 480),
            num_frames=81,
        )

        # Update status
        result = buffer.update_status(slot.slot_index, TransferPoll.Success)
        assert result is True

        # Check updated status
        updated_slot = buffer.get_slot(slot.slot_index)
        assert updated_slot.status == TransferPoll.Success

    def test_get_slot(self):
        """Test getting a slot by index."""
        buffer = TransferMetaBuffer(num_slots=256)

        slot = buffer.allocate(
            request_id=1,
            resolution=(832, 480),
            num_frames=81,
        )

        retrieved = buffer.get_slot(slot.slot_index)
        assert retrieved is not None
        assert retrieved.request_id == 1


class TestBufferIntegration:
    """Integration tests for buffers."""

    def test_end_to_end_flow(self):
        """Test complete allocation and transfer flow."""
        # Create buffers for Encoder role
        tensor_buffer = TransferTensorBuffer(
            role=DiffusionRole.ENCODER,
            concurrent_requests=4,
        )
        meta_buffer = TransferMetaBuffer(num_slots=256)

        # Allocate for a request
        tensor_slot = tensor_buffer.allocate_for_request(
            resolution=(832, 480),
            num_frames=81,
            request_id=1,
        )
        assert tensor_slot is not None

        meta_slot = meta_buffer.allocate(
            request_id=1,
            resolution=(832, 480),
            num_frames=81,
        )
        assert meta_slot is not None

        # Simulate transfer completion
        meta_buffer.update_status(meta_slot.slot_index, TransferPoll.Success)

        # Get updated status
        updated = meta_buffer.get_slot(meta_slot.slot_index)
        assert updated.status == TransferPoll.Success

        # Cleanup
        tensor_buffer.free(tensor_slot.slot_index)
        meta_buffer.free(meta_slot.slot_index)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
