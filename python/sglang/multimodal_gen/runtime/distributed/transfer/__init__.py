"""Transfer layer for Diffusion PD Disaggregation.

This module provides the transfer infrastructure for disaggregated diffusion,
enabling efficient tensor transfer between Encoder, Denoising, and Decoder roles.

Reference: RFC SGLang Diffusion Disaggregation §2: TransferManager
"""

from sglang.multimodal_gen.runtime.distributed.transfer.base import (
    BaseTransferManager,
    BaseTransferReceiver,
    BaseTransferSender,
    TransferArgs,
)
from sglang.multimodal_gen.runtime.distributed.transfer.manager import (
    CommonTransferManager,
)
from sglang.multimodal_gen.runtime.distributed.transfer.mooncake import (
    MooncakeTransferManager,
)
from sglang.multimodal_gen.runtime.distributed.transfer.utils import (
    DiffusionRole,
    RankRoutingInfo,
    TransferBackend,
    TransferPoll,
    align_to_power_of_two,
    calculate_latent_size,
    get_slot_config,
    group_concurrent_contiguous,
)

__all__ = [
    # Base classes
    "BaseTransferManager",
    "BaseTransferSender",
    "BaseTransferReceiver",
    "TransferArgs",
    # Implementations
    "CommonTransferManager",
    "MooncakeTransferManager",
    # Utilities
    "DiffusionRole",
    "RankRoutingInfo",
    "TransferBackend",
    "TransferPoll",
    "align_to_power_of_two",
    "calculate_latent_size",
    "get_slot_config",
    "group_concurrent_contiguous",
]
