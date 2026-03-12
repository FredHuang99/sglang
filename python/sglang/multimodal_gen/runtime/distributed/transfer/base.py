"""Abstract base classes for Diffusion PD Disaggregation Transfer Layer.

This module defines the core interfaces that must be implemented by
different transfer backends (e.g., Mooncake, NIXL, Mori).

Design Principles:
-----------------
1. Backend Agnostic: The abstract base classes define a common interface
   that allows different transfer backends to be plugged in seamlessly.
2. Sender/Receiver Pattern: Separate sender and receiver roles handle the
   bidirectional data flow in disaggregated diffusion.
3. State Machine: Transfer status follows a state machine (TransferPoll)
   to track the lifecycle of each transfer request.

File Structure:
--------------
- TransferArgs: Configuration dataclass for transfer setup
- BaseTransferManager: Abstract base for transfer coordination
- BaseTransferSender: Abstract base for sending tensors
- BaseTransferReceiver: Abstract base for receiving tensors

Reference:
- RFC SGLang Diffusion Disaggregation §2: Core Components per Role
- sglang/srt/disaggregation/base/conn.py for LLM PD patterns
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.distributed.transfer.utils import DiffusionRole
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


@dataclass
class TransferArgs:
    """Arguments for transfer configuration.

    This class is analogous to KVArgs in LLM PD disaggregation, but
    specifically for diffusion latent tensor transfer.

    Design Rationale:
    ----------------
    Different from LLM KV cache transfer, diffusion latent tensors have:
    - Variable sizes based on resolution and frame count
    - Different latency requirements (video generation is more latency-sensitive)
    - Different parallel strategy considerations (SP across video frames)

    Attributes:
        engine_rank: Rank ID within the current pipeline stage.
            Used to identify which rank within a distributed group is making the request.
        pp_rank: Pipeline parallel rank.
            Identifies the position in the pipeline for multi-stage setups.
        system_dp_rank: Data parallel rank across the system.
            Used for request routing in distributed inference.
        tensor_data_ptrs: List of memory pointers for tensor data.
            These are GPU memory addresses used for RDMA direct memory access.
        tensor_data_lens: List of tensor data lengths in bytes.
            Corresponding lengths for each tensor pointer.
        tensor_item_lens: List of tensor item lengths.
            Number of elements per tensor (for shape calculation).
        meta_data_ptrs: List of memory pointers for metadata.
            CPU pinned memory addresses for transfer metadata.
        meta_data_lens: List of metadata lengths in bytes.
            Corresponding lengths for each metadata pointer.
        meta_item_lens: List of metadata item lengths.
            Number of metadata items per pointer.
        ib_device: InfiniBand device for RDMA.
            Format: "mlx5_0:1" for RDMA-capable network interfaces.
        ib_traffic_class: InfiniBand traffic class/priority.
            Used for QoS in multi-tenant environments.
        gpu_id: GPU device ID for this rank.
            Specifies which GPU device to use for tensor operations.
        latent_channels: Number of channels in latent representation.
            Default 16 for most diffusion models (Wan, FLUX, etc.).
        max_resolution: Maximum supported (width, height) resolution.
            Used for buffer pre-allocation sizing.
        max_frames: Maximum supported number of frames.
            Used for buffer pre-allocation sizing.
        dtype_size: Size of data type in bytes.
            2 for fp16, 4 for bf16/fp32.

    Example:
        >>> args = TransferArgs(
        ...     engine_rank=0,
        ...     pp_rank=0,
        ...     system_dp_rank=0,
        ...     latent_channels=16,
        ...     max_resolution=(1920, 1080),
        ...     max_frames=300,
        ... )
        >>> print(args.latent_channels)
        16

    Reference: RFC §2 and sglang/srt/disaggregation/base/conn.py:KVArgs
    """

    # Engine information
    engine_rank: int = 0
    pp_rank: int = 0
    system_dp_rank: int = 0

    # Tensor data pointers (for latent transfer)
    tensor_data_ptrs: List[int] = field(default_factory=list)
    tensor_data_lens: List[int] = field(default_factory=list)
    tensor_item_lens: List[int] = field(default_factory=list)

    # Metadata pointers (for transfer metadata)
    meta_data_ptrs: List[int] = field(default_factory=list)
    meta_data_lens: List[int] = field(default_factory=list)
    meta_item_lens: List[int] = field(default_factory=list)

    # Hardware configuration
    ib_device: str = "mlx5_0:1"
    ib_traffic_class: str = "0"
    gpu_id: int = 0

    # Diffusion-specific configuration
    latent_channels: int = 16
    max_resolution: Tuple[int, int] = (1920, 1080)  # (width, height)
    max_frames: int = 300
    dtype_size: int = 2  # fp16 = 2 bytes


class BaseTransferManager(ABC):
    """Base class for managing tensor transfers in disaggregated diffusion.

    This class is analogous to BaseKVManager in LLM PD disaggregation.
    It manages the transfer state and coordinates between sender and receiver.

    Design Rationale:
    ----------------
    The TransferManager acts as the central coordinator for all transfer
    operations in a disaggregated diffusion pipeline. It provides:
    1. Buffer Management: Pre-allocated buffers for latent tensors and metadata
    2. Status Tracking: State machine for transfer progress
    3. Routing: Information about upstream/downstream peer locations
    4. Backend Abstraction: Pluggable backends (Mooncake, NIXL, etc.)

    Data Flow:
    ----------
    1. ENCODER -> DENOISING: Encoder sends latent tensor to Denoising
    2. DENOISING -> DECODER: Denoising sends denoised latent to Decoder

    Each transfer involves:
    a) Sender allocates buffer and notifies receiver address
    b) Receiver allocates buffer and sends back address
    c) Sender initiates RDMA transfer
    d) Receiver confirms transfer completion

    Attributes:
        role: The diffusion role (ENCODER, DENOISING, or DECODER)
        tensor_buffer: Buffer for latent tensor transfer
        meta_buffer: Buffer for metadata transfer

    Reference: RFC §2 "TransferManager" and sglang/srt/disaggregation/base/conn.py:BaseKVManager
    """

    @abstractmethod
    def __init__(
        self,
        args: TransferArgs,
        role: "DiffusionRole",
        server_args: "ServerArgs",
    ) -> None:
        """Initialize the transfer manager.

        Args:
            args: Transfer configuration arguments.
                Contains engine rank, parallel config, buffer pointers, etc.
            role: The diffusion role (ENCODER, DENOISING, or DECODER).
                Determines whether this manager sends, receives, or both.
            server_args: Server configuration.
                Contains disaggregation settings, concurrent request limits, etc.

        Raises:
            NotImplementedError: This is an abstract base class.
        """
        pass

    @abstractmethod
    def register_to_bootstrap(self) -> None:
        """Register this role to the bootstrap server.

        This is called during initialization to register the role's
        network address with the bootstrap server for peer discovery.

        Design Rationale:
        -----------------
        In a disaggregated setup, each role instance needs to discover
        its upstream and downstream peers. The bootstrap server acts as
        a registry where roles announce their presence and query peer locations.

        Process:
        1. This role sends registration message to bootstrap server
        2. Registration includes: role type, IP address, port, buffer pointers
        3. Bootstrap server maintains a routing table of all active roles
        4. Other roles can query this table to find their peers

        Reference: RFC §3 "Diffusion Server"
        """
        pass

    @abstractmethod
    def register_buffer_to_engine(self) -> None:
        """Register transfer buffer addresses to the transfer engine.

        This registers the pinned memory buffers with the transfer
        engine (e.g., Mooncake) for RDMA operations.

        Design Rationale:
        -----------------
        RDMA operations require memory to be registered with the network
        adapter. This registration:
        1. Provides the hardware with virtual-to-physical memory mappings
        2. Enables zero-copy transfers directly between GPU and network
        3. Must be done before any transfer can occur

        The registration typically includes:
        - Base pointer to the buffer pool
        - Total size of the buffer
        - Memory registration keys for authentication

        Reference: RFC §2 "Engine Setup"
        """
        pass

    def update_status(self, bootstrap_room: int, status: int) -> None:
        """Update the transfer status for a request.

        Args:
            bootstrap_room: The bootstrap room identifier.
                This is a unique identifier for a transfer session,
                typically derived from request ID and role combination.
            status: The new status (from TransferPoll).
                Valid values: Failed(0), Bootstrapping(1), WaitingForInput(2),
                Transferring(3), Success(4).

        Note:
            The base implementation does nothing. Subclasses should override
            to provide actual status tracking functionality.
        """
        pass

    def check_status(self, bootstrap_room: int) -> int:
        """Check the transfer status for a request.

        Args:
            bootstrap_room: The bootstrap room identifier.
                Unique identifier for the transfer session.

        Returns:
            The current status (from TransferPoll).
            Returns Bootstrapping(1) if the room is not found.

        Note:
            The base implementation returns Bootstrapping. Subclasses should
            override to provide actual status checking.
        """
        pass

    def get_free_buffer_slots(self) -> int:
        """Get the number of free buffer slots.

        Returns:
            Number of available buffer slots.
            Returns 0 in the base implementation.

        Note:
            Subclasses should override to return actual slot count
            from their buffer management system.
        """
        return 0


class BaseTransferSender(ABC):
    """Base class for sending tensors to downstream role.

    This class is analogous to BaseKVSender in LLM PD disaggregation.
    It handles the sender-side logic for tensor transfer.

    Design Rationale:
    -----------------
    The sender is responsible for:
    1. Initiating the transfer to downstream peers
    2. Managing the transfer lifecycle (init -> send -> poll)
    3. Handling parallel strategy differences (slice transfer)

    Data Flow:
    ----------
    1. init(): Receive tensor info from pipeline
    2. send(): Initiate RDMA transfer to receiver
    3. poll(): Check if transfer completed

    Attributes:
        mgr: The transfer manager for coordination
        bootstrap_room: Session identifier for this transfer
        bootstrap_addr: Address of bootstrap server
        dest_tp_ranks: List of destination TP ranks

    Reference: sglang/srt/disaggregation/base/conn.py:BaseKVSender
    """

    @abstractmethod
    def __init__(
        self,
        mgr: BaseTransferManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
    ) -> None:
        """Initialize the sender.

        Args:
            mgr: The transfer manager.
                Provides access to buffers, status tracking, and transfer engine.
            bootstrap_addr: Address of the bootstrap server.
                Used for peer discovery and coordination.
            bootstrap_room: The bootstrap room identifier.
                Unique session ID for this transfer operation.
            dest_tp_ranks: List of destination TP ranks.
                Specifies which ranks in the downstream role should receive the tensor.

        Raises:
            NotImplementedError: This is an abstract base class.
        """
        pass

    @abstractmethod
    def init(
        self,
        latent_shape: Tuple[int, ...],
        meta_index: int,
        tensor_buffer_slot,
    ) -> None:
        """Initialize the transfer with tensor information.

        This is called before send() to provide the necessary context
        for the transfer operation.

        Args:
            latent_shape: Shape of the latent tensor.
                Format: (batch, channels, temporal, height, width)
                Used to compute transfer size and validate buffer capacity.
            meta_index: Index in the metadata buffer.
                Points to the metadata slot containing transfer info.
            tensor_buffer_slot: The allocated tensor buffer slot.
                Contains the memory pointer and size for the transfer.

        Raises:
            NotImplementedError: This is an abstract base class.
        """
        pass

    @abstractmethod
    def send(self) -> int:
        """Initiate the tensor transfer.

        This method starts the actual RDMA transfer from local buffer
        to the downstream role's buffer.

        Returns:
            Status code from TransferPoll.
            Returns Transferring(3) if initiated successfully,
            Failed(0) if there was an error.

        Raises:
            NotImplementedError: This is an abstract base class.
        """
        pass

    @abstractmethod
    def poll(self) -> int:
        """Poll the transfer status.

        This method checks whether the transfer has completed.

        Returns:
            Current status from TransferPoll:
            - Bootstrapping(1): Transfer not yet initiated
            - WaitingForInput(2): Transfer initiated, waiting for receiver
            - Transferring(3): Transfer in progress
            - Success(4): Transfer completed successfully
            - Failed(0): Transfer failed

        Raises:
            NotImplementedError: This is an abstract base class.
        """
        pass


class BaseTransferReceiver(ABC):
    """Base class for receiving tensors from upstream role.

    This class is analogous to BaseKVReceiver in LLM PD disaggregation.
    It handles the receiver-side logic for tensor transfer.

    Design Rationale:
    -----------------
    The receiver is responsible for:
    1. Preparing buffer space for incoming tensors
    2. Notifying sender of buffer location
    3. Confirming transfer completion

    Data Flow:
    ----------
    1. init(): Set up expected tensor info and buffer
    2. poll(): Check if data has arrived
    3. clear(): Clean up after processing
    4. abort(): Handle transfer failure

    Attributes:
        mgr: The transfer manager for coordination
        bootstrap_room: Session identifier for this transfer
        bootstrap_addr: Address of bootstrap server

    Reference: sglang/srt/disaggregation/base/conn.py:BaseKVReceiver
    """

    @abstractmethod
    def __init__(
        self,
        mgr: BaseTransferManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ) -> None:
        """Initialize the receiver.

        Args:
            mgr: The transfer manager.
                Provides access to buffers, status tracking, and transfer engine.
            bootstrap_addr: Address of the bootstrap server.
                Used for peer discovery and coordination.
            bootstrap_room: Optional bootstrap room identifier.
                If provided, associates this receiver with a specific transfer session.

        Raises:
            NotImplementedError: This is an abstract base class.
        """
        pass

    @abstractmethod
    def init(
        self,
        latent_shape: Tuple[int, ...],
        tensor_buffer_slot,
        meta_index: int,
    ) -> None:
        """Initialize the receive with expected tensor information.

        This is called before polling to set up expectations for the incoming tensor.

        Args:
            latent_shape: Expected shape of the latent tensor.
                Format: (batch, channels, temporal, height, width)
                Used to validate incoming data size.
            tensor_buffer_slot: The allocated tensor buffer slot.
                Contains the memory pointer where data will be written.
            meta_index: Index in the metadata buffer.
                Points to the metadata slot for this transfer.

        Raises:
            NotImplementedError: This is an abstract base class.
        """
        pass

    @abstractmethod
    def poll(self) -> int:
        """Poll the transfer status.

        This method checks whether the tensor has been transferred.

        Returns:
            Current status from TransferPoll:
            - Bootstrapping(1): Not yet initialized
            - WaitingForInput(2): Initialized, waiting for data
            - Transferring(3): Data arriving
            - Success(4): Transfer completed successfully
            - Failed(0): Transfer failed

        Raises:
            NotImplementedError: This is an abstract base class.
        """
        pass

    def clear(self) -> None:
        """Clear any internal states.

        This is called after the tensor has been processed to clean up
        any receiver-specific state.

        Note:
            The base implementation does nothing. Subclasses should override
            to provide actual cleanup functionality.
        """
        pass

    def abort(self) -> None:
        """Abort the current transfer.

        This is called when the transfer fails or needs to be cancelled.
        It signals to the manager that the transfer has failed.

        Note:
            The base implementation does nothing. Subclasses should override
            to provide actual abort handling.
        """
        pass
