"""Unit tests for TransferManager.

Tests the CommonTransferManager and MooncakeTransferManager classes
used in Diffusion PD disaggregation.
"""

import time

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed.transfer.base import TransferArgs
from sglang.multimodal_gen.runtime.distributed.transfer.manager import (
    CommonTransferManager,
)
from sglang.multimodal_gen.runtime.distributed.transfer.mooncake import (
    MooncakeTransferManager,
)
from sglang.multimodal_gen.runtime.distributed.transfer.utils import (
    DiffusionRole,
    RankRoutingInfo,
    TransferPoll,
    group_concurrent_contiguous,
)


class MockServerArgs:
    """Mock server args for testing."""

    def __init__(self):
        self.disaggregation_max_concurrent_requests = 4


class TestTransferArgs:
    """Test cases for TransferArgs."""

    def test_creation(self):
        """Test creating TransferArgs."""
        args = TransferArgs(
            engine_rank=0,
            pp_rank=0,
            system_dp_rank=0,
            latent_channels=16,
            max_resolution=(1920, 1080),
            max_frames=300,
        )

        assert args.engine_rank == 0
        assert args.pp_rank == 0
        assert args.latent_channels == 16
        assert args.max_resolution == (1920, 1080)
        assert args.max_frames == 300


class TestCommonTransferManager:
    """Test cases for CommonTransferManager."""

    def test_encoder_manager_creation(self):
        """Test creating a manager for Encoder role."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.ENCODER,
            server_args=server_args,
        )

        assert mgr.role == DiffusionRole.ENCODER
        assert mgr.is_sender is True
        assert mgr.is_receiver is False
        assert mgr.tensor_buffer is not None
        assert mgr.meta_buffer is not None

    def test_denoising_manager_creation(self):
        """Test creating a manager for Denoising role."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.DENOISING,
            server_args=server_args,
        )

        assert mgr.role == DiffusionRole.DENOISING
        assert mgr.is_sender is True
        assert mgr.is_receiver is True

    def test_decoder_manager_creation(self):
        """Test creating a manager for Decoder role."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.DECODER,
            server_args=server_args,
        )

        assert mgr.role == DiffusionRole.DECODER
        assert mgr.is_sender is False
        assert mgr.is_receiver is True

    def test_null_role_manager(self):
        """Test creating a manager for NULL role."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.NULL,
            server_args=server_args,
        )

        assert mgr.role == DiffusionRole.NULL
        assert mgr.is_sender is False
        assert mgr.is_receiver is False
        assert mgr.tensor_buffer is None
        assert mgr.meta_buffer is None

    def test_status_update(self):
        """Test updating transfer status."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.ENCODER,
            server_args=server_args,
        )

        bootstrap_room = 123

        # Initial status
        status = mgr.check_status(bootstrap_room)
        assert status == TransferPoll.Bootstrapping

        # Update to WaitingForInput
        mgr.update_status(bootstrap_room, TransferPoll.WaitingForInput)
        status = mgr.check_status(bootstrap_room)
        assert status == TransferPoll.WaitingForInput

        # Update to Transferring
        mgr.update_status(bootstrap_room, TransferPoll.Transferring)
        status = mgr.check_status(bootstrap_room)
        assert status == TransferPoll.Transferring

        # Update to Success
        mgr.update_status(bootstrap_room, TransferPoll.Success)
        status = mgr.check_status(bootstrap_room)
        assert status == TransferPoll.Success

    def test_status_transitions(self):
        """Test status transition logic."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.ENCODER,
            server_args=server_args,
        )

        bootstrap_room = 456

        # Failed should override any status
        mgr.update_status(bootstrap_room, TransferPoll.Success)
        mgr.update_status(bootstrap_room, TransferPoll.Failed)
        assert mgr.check_status(bootstrap_room) == TransferPoll.Failed

        # Failed should stay failed even with higher status
        mgr.update_status(bootstrap_room, TransferPoll.WaitingForInput)
        assert mgr.check_status(bootstrap_room) == TransferPoll.Failed

    def test_record_failure(self):
        """Test recording transfer failures."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.ENCODER,
            server_args=server_args,
        )

        bootstrap_room = 789

        # Record a failure
        mgr.record_failure(bootstrap_room, "Transfer timeout")

        # Check failure record
        failure = mgr.get_failure(bootstrap_room)
        assert failure == "Transfer timeout"

    def test_allocate_tensor_buffer(self):
        """Test allocating tensor buffer."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.ENCODER,
            server_args=server_args,
        )

        # Allocate
        slot = mgr.allocate_tensor_buffer(
            resolution=(832, 480),
            num_frames=81,
            request_id=1,
        )

        assert slot is not None
        assert slot.resolution == (832, 480)
        assert slot.num_frames == 81

        # Free
        result = mgr.free_tensor_buffer(slot.slot_index)
        assert result is True

    def test_allocate_meta_buffer(self):
        """Test allocating meta buffer."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.ENCODER,
            server_args=server_args,
        )

        # Allocate
        slot = mgr.allocate_meta_buffer(
            request_id=1,
            resolution=(832, 480),
            num_frames=81,
        )

        assert slot is not None
        assert slot.request_id == 1
        assert slot.resolution == (832, 480)
        assert slot.num_frames == 81

        # Free
        result = mgr.free_meta_buffer(slot.slot_index)
        assert result is True

    def test_free_buffer_slots(self):
        """Test getting free buffer slots count."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.ENCODER,
            server_args=server_args,
        )

        initial_free = mgr.get_free_buffer_slots()
        assert initial_free > 0

        # Allocate
        slot = mgr.allocate_tensor_buffer(
            resolution=(832, 480),
            num_frames=81,
            request_id=1,
        )
        assert slot is not None

        free_after = mgr.get_free_buffer_slots()
        assert free_after < initial_free


class TestDiffusionRole:
    """Test cases for DiffusionRole enum."""

    def test_is_sender(self):
        """Test is_sender method."""
        assert DiffusionRole.ENCODER.is_sender() is True
        assert DiffusionRole.DENOISING.is_sender() is True
        assert DiffusionRole.DECODER.is_sender() is False
        assert DiffusionRole.NULL.is_sender() is False

    def test_is_receiver(self):
        """Test is_receiver method."""
        assert DiffusionRole.ENCODER.is_receiver() is False
        assert DiffusionRole.DENOISING.is_receiver() is True
        assert DiffusionRole.DECODER.is_receiver() is True
        assert DiffusionRole.NULL.is_receiver() is False


class TestTransferPoll:
    """Test cases for TransferPoll status values."""

    def test_status_values(self):
        """Test TransferPoll status values."""
        assert TransferPoll.Failed == 0
        assert TransferPoll.Bootstrapping == 1
        assert TransferPoll.WaitingForInput == 2
        assert TransferPoll.Transferring == 3
        assert TransferPoll.Success == 4


class TestRankRoutingInfo:
    """Test cases for RankRoutingInfo dataclass."""

    def test_creation(self):
        """Test creating RankRoutingInfo."""
        info = RankRoutingInfo(
            instance_id="encoder_001",
            rank_id=0,
            ip="10.0.1.10",
            port=50001,
            tensor_ptr=0x7F8A1000,
            meta_ptr=0x7F8A2000,
            parallel_config={"dp": 1, "tp": 1, "sp": 1},
        )

        assert info.instance_id == "encoder_001"
        assert info.rank_id == 0
        assert info.ip == "10.0.1.10"
        assert info.port == 50001
        assert info.tensor_ptr == 0x7F8A1000
        assert info.parallel_config == {"dp": 1, "tp": 1, "sp": 1}

    def test_is_expired(self):
        """Test expiration checking."""
        # Fresh info should not be expired
        info = RankRoutingInfo(
            instance_id="encoder_001",
            rank_id=0,
            ip="10.0.1.10",
            port=50001,
            tensor_ptr=0x7F8A1000,
            meta_ptr=0x7F8A2000,
        )
        assert info.is_expired(max_age_seconds=300) is False

        # Old info should be expired
        old_info = RankRoutingInfo(
            instance_id="encoder_002",
            rank_id=0,
            ip="10.0.1.11",
            port=50002,
            tensor_ptr=0x7F8A3000,
            meta_ptr=0x7F8A4000,
            timestamp=time.time() - 400,  # 400 seconds ago
        )
        assert old_info.is_expired(max_age_seconds=300) is True


class TestRoutingTable:
    """Test cases for routing table in CommonTransferManager."""

    def test_routing_table_update(self):
        """Test updating routing table."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.DENOISING,
            server_args=server_args,
        )

        # Update routing table
        info = RankRoutingInfo(
            instance_id="encoder_001",
            rank_id=0,
            ip="10.0.1.10",
            port=50001,
            tensor_ptr=0x7F8A1000,
            meta_ptr=0x7F8A2000,
            parallel_config={"dp": 1, "tp": 1, "sp": 1},
        )

        mgr.update_routing_table("encoder_001", info)

        # Get from routing table
        retrieved = mgr.get_downstream_addr("encoder_001")
        assert retrieved is not None
        assert retrieved.instance_id == "encoder_001"
        assert retrieved.ip == "10.0.1.10"

    def test_routing_table_not_found(self):
        """Test getting non-existent routing entry."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.DENOISING,
            server_args=server_args,
        )

        # Should return None for non-existent entry
        retrieved = mgr.get_downstream_addr("nonexistent")
        assert retrieved is None

    def test_clear_routing_table(self):
        """Test clearing routing table."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = CommonTransferManager(
            args=args,
            role=DiffusionRole.DENOISING,
            server_args=server_args,
        )

        # Add entry
        info = RankRoutingInfo(
            instance_id="encoder_001",
            rank_id=0,
            ip="10.0.1.10",
            port=50001,
            tensor_ptr=0x7F8A1000,
            meta_ptr=0x7F8A2000,
        )
        mgr.update_routing_table("encoder_001", info)

        # Clear
        mgr.clear_routing_table()

        # Should return None now
        retrieved = mgr.get_downstream_addr("encoder_001")
        assert retrieved is None


class TestTransferSlice:
    """Test cases for transfer_slice in MooncakeTransferManager."""

    def test_check_parallel_mismatch(self):
        """Test parallel mismatch detection."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = MooncakeTransferManager(
            args=args,
            role=DiffusionRole.ENCODER,
            server_args=server_args,
        )

        # Same config - no mismatch
        src_config = {"tp": 1, "sp": 1}
        dst_config = {"tp": 1, "sp": 1}
        assert mgr.check_parallel_mismatch(src_config, dst_config) is False

        # Different TP - mismatch
        src_config = {"tp": 1, "sp": 1}
        dst_config = {"tp": 2, "sp": 1}
        assert mgr.check_parallel_mismatch(src_config, dst_config) is True

        # Different SP - mismatch
        src_config = {"tp": 1, "sp": 1}
        dst_config = {"tp": 1, "sp": 2}
        assert mgr.check_parallel_mismatch(src_config, dst_config) is True

    def test_compute_latent_slices(self):
        """Test latent slice computation."""
        args = TransferArgs()
        server_args = MockServerArgs()

        mgr = MooncakeTransferManager(
            args=args,
            role=DiffusionRole.ENCODER,
            server_args=server_args,
        )

        # Test with shape (16, 21, 60, 104) - C, T, H, W
        latent_shape = (16, 21, 60, 104)

        # Same parallel config - should return single slice
        src_config = {"tp": 1, "sp": 1}
        dst_config = {"tp": 1, "sp": 1}
        slices = mgr.compute_latent_slices(latent_shape, src_config, dst_config)
        assert len(slices) == 1
        # Single slice should cover full width
        assert slices[0][3].start == 0
        assert slices[0][3].stop == 104

        # Different SP config - should split
        src_config = {"tp": 1, "sp": 1}
        dst_config = {"tp": 1, "sp": 2}
        slices = mgr.compute_latent_slices(latent_shape, src_config, dst_config)
        # Should have 2 slices for 2 destination ranks
        assert len(slices) == 2


class TestGroupConcurrentContiguous:
    """Test cases for group_concurrent_contiguous."""

    def test_empty_input(self):
        """Test with empty input."""
        src_groups, dst_groups = group_concurrent_contiguous([], [])
        assert src_groups == []
        assert dst_groups == []

    def test_single_element(self):
        """Test with single element."""
        src_groups, dst_groups = group_concurrent_contiguous([0], [0])
        assert src_groups == [(0, 1)]
        assert dst_groups == [(0, 1)]

    def test_contiguous_indices(self):
        """Test with contiguous indices."""
        src_groups, dst_groups = group_concurrent_contiguous([0, 1, 2, 3], [0, 1, 2, 3])
        assert src_groups == [(0, 4)]
        assert dst_groups == [(0, 4)]

    def test_discontiguous_indices(self):
        """Test with discontiguous indices."""
        src_groups, dst_groups = group_concurrent_contiguous(
            [0, 1, 2, 5, 6, 7], [0, 1, 2, 5, 6, 7]
        )
        # Should be split into two groups
        assert len(src_groups) == 2
        assert src_groups[0] == (0, 3)  # 0, 1, 2
        assert src_groups[1] == (5, 3)  # 5, 6, 7

    def test_mixed_indices(self):
        """Test with mixed contiguous and discontiguous."""
        src_groups, dst_groups = group_concurrent_contiguous(
            [0, 1, 2, 5, 6, 7, 10], [0, 1, 2, 5, 6, 7, 10]
        )
        # Should be split into three groups
        assert len(src_groups) == 3
        assert src_groups[0] == (0, 3)
        assert src_groups[1] == (5, 3)
        assert src_groups[2] == (10, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
