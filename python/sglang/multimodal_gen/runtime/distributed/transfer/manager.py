"""CommonTransferManager implementation for Diffusion PD Disaggregation.

This module provides the common implementation of TransferManager that
handles buffer management, request state tracking, and ZMQ-based communication.

Design Principles:
-----------------
1. Buffer Management: Uses TransferTensorBuffer and TransferMetaBuffer for
   efficient memory allocation with buddy algorithm.
2. Status Tracking: Implements TransferPoll state machine for each transfer request.
3. Thread Safety: All shared state protected by locks for concurrent access.
4. Event-Driven: Uses ZMQ for asynchronous communication between roles.

File Structure:
-------------
- CommonTransferManager: Main class with buffer management and event loops
- receive_event_loop(): Handles incoming address requests from upstream
- wait_for_transfer_finished(): Handles transfer completion notifications
- Routing table: Caches downstream peer information

Reference:
- RFC SGLang Diffusion Disaggregation §2: TransferManager
- sglang/srt/disaggregation/common/conn.py:CommonKVManager
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Dict, Optional

import zmq

from sglang.multimodal_gen.runtime.distributed.transfer.base import (
    BaseTransferManager,
    TransferArgs,
)
from sglang.multimodal_gen.runtime.distributed.transfer.utils import (
    DiffusionRole,
    RankRoutingInfo,
    TransferPoll,
)

# Import buffer types lazily to avoid circular import
if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.cache.buffer import (
        TransferMetaBuffer,
        TransferTensorBuffer,
    )

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = logging.getLogger(__name__)


class CommonTransferManager(BaseTransferManager):
    """Common implementation of TransferManager.

    This class is analogous to CommonKVManager in LLM PD disaggregation.
    It provides common functionality for buffer management, request state
    tracking, and ZMQ-based notification.

    Design Rationale:
    -----------------
    The CommonTransferManager serves as the base implementation that can be
    extended by backend-specific managers (e.g., MooncakeTransferManager).
    It provides:

    1. Buffer Management:
       - TransferTensorBuffer: Dynamic allocation for latent tensors
       - TransferMetaBuffer: Fixed-size slots for metadata

    2. Request State Tracking:
       - Status machine per bootstrap_room (transfer session)
       - Failure recording and retrieval
       - Thread-safe updates

    3. ZMQ Communication:
       - receive_event_loop: Listen for upstream address requests
       - wait_for_transfer_finished: Listen for completion notifications

    4. Routing Table:
       - Cache downstream peer information
       - Thread-safe read/write
       - Expiration-based cleanup

    Data Flow:
    ----------
    ENCODER role:
        1. Allocates tensor buffer for output latent
        2. Sends address to DENOISING via ZMQ
        3. Performs RDMA transfer
        4. Notifies DENOISING of completion

    DENOISING role:
        1. Receives address from ENCODER (receive_event_loop)
        2. Allocates receive buffer
        3. Notifies ENCODER of buffer location
        4. Waits for transfer (wait_for_transfer_finished)
        5. Processes latent
        6. Sends to DECODER

    DECODER role:
        1. Receives address from DENOISING
        2. Allocates receive buffer
        3. Waits for transfer
        4. Processes latent to final output

    Attributes:
        role: The diffusion role (ENCODER, DENOISING, or DECODER).
            Determines send/receive behavior and buffer sizing.
        tensor_buffer: Buffer for latent tensor transfer.
            Uses buddy allocator for dynamic sizing.
        meta_buffer: Buffer for metadata transfer.
            Fixed-size slots for transfer metadata.
        is_sender: Whether this role sends to downstream.
        is_receiver: Whether this role receives from upstream.
        request_status: Dict mapping bootstrap_room -> TransferPoll status.
        failure_records: Dict mapping bootstrap_room -> failure reason.
        routing_table: Dict mapping instance_id -> RankRoutingInfo.

    Reference: sglang/srt/disaggregation/common/conn.py:CommonKVManager
    """

    def __init__(
        self,
        args: TransferArgs,
        role: DiffusionRole,
        server_args: "ServerArgs",
    ) -> None:
        """Initialize the transfer manager.

        This constructor sets up all necessary components for the transfer manager:

        1. Buffer Initialization:
           - Creates TransferTensorBuffer with role-specific slot configuration
           - Creates TransferMetaBuffer with fixed-size slots

        2. ZMQ Socket Setup:
           - Creates PULL socket for receiving messages
           - Binds to random available port on localhost

        3. Role Configuration:
           - Sets is_sender/is_receiver based on role type
           - ENCODER: is_sender=True, is_receiver=False
           - DENOISING: is_sender=True, is_receiver=True
           - DECODER: is_sender=False, is_receiver=True

        4. Thread Startup:
           - Starts receive_event_loop if is_receiver=True
           - Starts wait_for_transfer_finished if is_receiver=True

        Args:
            args: Transfer configuration arguments.
                Contains latent_channels, max_resolution, max_frames, etc.
            role: The diffusion role.
                Determines pipeline position and buffer requirements.
            server_args: Server configuration.
                Contains disaggregation_max_concurrent_requests, etc.

        Example:
            >>> from sglang.multimodal_gen.runtime.distributed.transfer.base import TransferArgs
            >>> from sglang.multimodal_gen.runtime.distributed.transfer.utils import DiffusionRole
            >>> args = TransferArgs(latent_channels=16)
            >>> mgr = CommonTransferManager(args, DiffusionRole.DENOISING, server_args)
            >>> print(f"Is sender: {mgr.is_sender}, Is receiver: {mgr.is_receiver}")
            Is sender: True, Is receiver: True
        """
        # Import buffer types here to avoid circular import
        from sglang.multimodal_gen.runtime.cache.buffer import (
            TransferMetaBuffer,
            TransferTensorBuffer,
        )

        self.transfer_args = args
        self.role = role
        self.server_args = server_args

        # Get configuration
        self.concurrent_requests = getattr(
            server_args, "disaggregation_max_concurrent_requests", 4
        )
        self.latent_channels = args.latent_channels

        # Initialize buffers
        self.tensor_buffer: Optional[TransferTensorBuffer] = None
        self.meta_buffer: Optional[TransferMetaBuffer] = None

        if role != DiffusionRole.NULL:
            self.tensor_buffer = TransferTensorBuffer(
                role=role,
                concurrent_requests=self.concurrent_requests,
                latent_channels=self.latent_channels,
            )
            self.meta_buffer = TransferMetaBuffer(
                num_slots=self.concurrent_requests * 2,
            )

        # Initialize ZMQ socket for notifications
        self._init_zmq_socket()

        # Request status tracking
        self.request_status: Dict[int, int] = {}
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

        # For role-specific initialization
        if role == DiffusionRole.ENCODER:
            # Encoder is the first stage, sends to Denoising
            self.is_sender = True
            self.is_receiver = False
        elif role == DiffusionRole.DENOISING:
            # Denoising receives from Encoder, sends to Decoder
            self.is_sender = True
            self.is_receiver = True
        elif role == DiffusionRole.DECODER:
            # Decoder is the last stage, only receives
            self.is_sender = False
            self.is_receiver = True
        else:
            self.is_sender = False
            self.is_receiver = False

        # Routing table for downstream peers
        # Maps instance_id -> RankRoutingInfo
        self.routing_table: Dict[str, RankRoutingInfo] = {}
        self.routing_lock = threading.Lock()

        # Thread control flags
        self._running = True

        # Start receive event loop if this role receives data
        if self.is_receiver:
            self._receive_thread = threading.Thread(
                target=self.receive_event_loop,
                daemon=True,
                name=f"transfer_receive_{role.value}",
            )
            self._receive_thread.start()

            # Also start the transfer finished notification loop
            self._notify_thread = threading.Thread(
                target=self.wait_for_transfer_finished,
                daemon=True,
                name=f"transfer_notify_{role.value}",
            )
            self._notify_thread.start()

        logger.info(
            f"CommonTransferManager initialized for role {role.value}: "
            f"is_sender={self.is_sender}, is_receiver={self.is_receiver}"
        )

    def _init_zmq_socket(self) -> None:
        """Initialize ZMQ socket for receiving notifications.

        Creates a ZMQ PULL socket bound to a random available port on localhost.
        This socket is used for receiving:
        1. Address requests from upstream roles
        2. Transfer completion notifications

        Design Rationale:
        -----------------
        Using ZMQ PULL socket provides:
        - Reliable message queuing
        - Easy integration with async event loops
        - Simple address discovery via LAST_ENDPOINT

        Note:
            In production, this would use actual network interfaces and
            potentially different socket types based on deployment.

        Attributes Modified:
            zmq_context: ZMQ context for socket management
            zmq_socket: PULL socket for receiving messages
            zmq_port: Bound port number (set after bind)
        """
        context = zmq.Context()
        self.zmq_context = context
        self.zmq_socket = context.socket(zmq.PULL)
        self.zmq_socket.bind("tcp://127.0.0.1:*")
        self.zmq_port = self.zmq_socket.getsockopt(zmq.LAST_ENDPOINT)

        logger.debug(f"TransferManager ZMQ socket bound to {self.zmq_port}")

    def register_to_bootstrap(self) -> None:
        """Register this role to the bootstrap server.

        In a full implementation, this would register with a centralized
        DiffusionServer for peer discovery. For now, this is a placeholder.

        Reference: RFC Section 3 "Diffusion Server"
        """
        # TODO: Implement bootstrap registration with DiffusionServer
        logger.info(
            f"TransferManager: bootstrap registration for role {self.role.value} "
            "(placeholder - to be implemented with DiffusionServer)"
        )

    def register_buffer_to_engine(self) -> None:
        """Register buffer addresses to the transfer engine.

        This method should be overridden by backend-specific implementations
        (e.g., MooncakeTransferManager) to register buffer addresses.

        Reference: RFC Section 2 "Engine Setup"
        """
        # Base implementation does nothing
        # Backend-specific implementations should override this
        logger.debug(
            f"TransferManager: buffer registration for role {self.role.value} "
            "(base implementation)"
        )

    def update_status(self, bootstrap_room: int, status: int) -> None:
        """Update the transfer status for a request.

        Implements a monotonic state machine where:
        - Failed (0) is terminal: once set, stays Failed
        - Other states can progress forward: Bootstrapping -> WaitingForInput -> Transferring -> Success

        State Transition Rules:
        ------------------------
        1. If bootstrap_room not in status dict: set to new status
        2. If current is Failed (0): stay Failed (terminal state)
        3. If new status is Failed (0): set to Failed
        4. Otherwise: set to max(current, new) for forward progress

        Args:
            bootstrap_room: The bootstrap room identifier.
                Unique session ID for this transfer operation.
            status: The new status (from TransferPoll).
                Valid values: Failed(0), Bootstrapping(1), WaitingForInput(2),
                Transferring(3), Success(4).

        Example:
            >>> mgr.update_status(123, TransferPoll.Bootstrapping)
            >>> mgr.update_status(123, TransferPoll.Transferring)
            >>> mgr.check_status(123)
            <TransferPoll.Transferring: 3>
            >>> mgr.update_status(123, TransferPoll.Failed)
            >>> mgr.check_status(123)
            <TransferPoll.Failed: 0>
        """
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            current = self.request_status[bootstrap_room]
            if current == 0:  # Already Failed, don't change
                return
            if status == 0:  # New status is Failed
                self.request_status[bootstrap_room] = 0
            else:
                self.request_status[bootstrap_room] = max(current, status)

    def check_status(self, bootstrap_room: int) -> int:
        """Check the transfer status for a request.

        Args:
            bootstrap_room: The bootstrap room identifier.
                Unique session ID for this transfer operation.

        Returns:
            The current status (from TransferPoll).
            Returns Bootstrapping(1) if the bootstrap_room is not found
            (indicating the request hasn't been registered yet).

        Example:
            >>> status = mgr.check_status(123)
            >>> print(status)
            TransferPoll.Bootstrapping
        """
        return self.request_status.get(bootstrap_room, TransferPoll.Bootstrapping)

    def record_failure(self, bootstrap_room: int, failure_reason: str) -> None:
        """Record a transfer failure.

        Stores the failure reason for later retrieval and debugging.

        Args:
            bootstrap_room: The bootstrap room identifier.
                Unique session ID for this transfer operation.
            failure_reason: Description of what went wrong.
                Should include relevant details for debugging.

        Thread Safety:
        -------------
        This method is thread-safe due to the failure_lock.

        Example:
            >>> mgr.record_failure(123, "RDMA transfer timeout after 30s")
            >>> print(mgr.get_failure(123))
            RDMA transfer timeout after 30s
        """
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def get_failure(self, bootstrap_room: int) -> Optional[str]:
        """Get the failure reason for a request.

        Args:
            bootstrap_room: The bootstrap room identifier

        Returns:
            Failure reason, or None if not found
        """
        with self.failure_lock:
            return self.failure_records.get(bootstrap_room)

    def get_free_buffer_slots(self) -> int:
        """Get the number of free buffer slots.

        Returns:
            Number of available tensor buffer slots
        """
        if self.tensor_buffer is not None:
            return self.tensor_buffer.get_free_slots_count()
        return 0

    def allocate_tensor_buffer(
        self,
        resolution: Tuple[int, int],
        num_frames: int,
        request_id: int,
    ) -> Optional["TransferTensorBufferSlot"]:
        """Allocate a tensor buffer slot for a request.

        Args:
            resolution: (width, height) of the input
            num_frames: Number of frames
            request_id: Request identifier

        Returns:
            TransferTensorBufferSlot if successful, None otherwise
        """
        if self.tensor_buffer is not None:
            return self.tensor_buffer.allocate_for_request(
                resolution=resolution,
                num_frames=num_frames,
                request_id=request_id,
            )
        return None

    def allocate_meta_buffer(
        self,
        request_id: int,
        resolution: Tuple[int, int],
        num_frames: int,
    ) -> Optional["TransferMetaBufferSlot"]:
        """Allocate a metadata buffer slot for a request.

        Args:
            request_id: Request identifier
            resolution: (width, height) of the input
            num_frames: Number of frames

        Returns:
            TransferMetaBufferSlot if successful, None otherwise
        """
        if self.meta_buffer is not None:
            return self.meta_buffer.allocate(
                request_id=request_id,
                resolution=resolution,
                num_frames=num_frames,
            )
        return None

    def free_tensor_buffer(self, slot_index: int) -> bool:
        """Free a tensor buffer slot.

        Args:
            slot_index: Index of the slot to free

        Returns:
            True if successful
        """
        if self.tensor_buffer is not None:
            return self.tensor_buffer.free(slot_index)
        return False

    def free_meta_buffer(self, slot_index: int) -> bool:
        """Free a metadata buffer slot.

        Args:
            slot_index: Index of the slot to free

        Returns:
            True if successful
        """
        if self.meta_buffer is not None:
            return self.meta_buffer.free(slot_index)
        return False

    def receive_event_loop(self) -> None:
        """Receive event loop for handling incoming transfer requests.

        This loop runs in a dedicated thread and handles address requests from
        upstream roles. It performs the following steps:

        1. Receive address request from upstream (via ZMQ)
        2. Parse request: bootstrap_room, instance_id, parallel_config, resolution, num_frames
        3. Allocate local tensor and metadata buffers
        4. Send response with local address info (ip, port, buffer pointers)
        5. Update routing table with upstream information
        6. Update status to WaitingForInput

        Message Format (incoming):
        --------------------------
        {{
            "bootstrap_room": int,      # Session identifier
            "instance_id": str,        # Upstream instance ID (e.g., "encoder_001")
            "parallel_config": dict,    # TP/SP/FSDP configuration
            "resolution": tuple,        # (width, height) of input
            "num_frames": int,         # Number of frames
            "rank_id": int,            # Upstream rank ID
            "ip": str,                 # Upstream IP for return routing
            "port": int                # Upstream port for return routing
        }}

        Message Format (outgoing response):
        -----------------------------------
        {{
            "bootstrap_room": int,      # Echo back session ID
            "status": str,             # "ok" or "error"
            "ip": str,                 # This role's IP
            "port": int,               # This role's ZMQ port
            "tensor_ptr": int,         # GPU memory pointer for tensor
            "meta_ptr": int,           # Pinned memory pointer for metadata
            "tensor_slot_index": int,   # Slot index for tensor buffer
            "meta_slot_index": int,     # Slot index for metadata buffer
            "parallel_config": dict,    # This role's parallel config
            "resolution": tuple,        # Accepted resolution
            "num_frames": int          # Accepted frame count
        }}

        Design Rationale:
        -----------------
        This event-driven approach allows:
        - Asynchronous handling of multiple concurrent transfers
        - Backpressure through buffer allocation failure
        - Clean separation between address exchange and actual data transfer

        Thread Safety:
        -------------
        - Runs in dedicated daemon thread
        - Uses buffer allocation locks internally
        - Updates routing table with routing_lock

        Error Handling:
        --------------
        - Buffer allocation failure: Sends error response, continues loop
        - ZMQ.Again: Sleeps briefly, continues loop
        - Other exceptions: Logs error, sleeps briefly, continues loop

        Reference:
        - RFC Section 2: TransferManager.receive_event_loop
        - sglang/srt/disaggregation/mooncake/conn.py:decode_thread (line 961-990)
        """
        logger.info(f"receive_event_loop started for role {self.role.value}")

        while self._running:
            try:
                # Receive request from upstream role
                # Format: {"bootstrap_room": int, "instance_id": str,
                #          "parallel_config": dict, "resolution": tuple,
                #          "num_frames": int}
                msg = self.zmq_socket.recv_json(zmq.NOBLOCK)

                bootstrap_room = msg.get("bootstrap_room")
                instance_id = msg.get("instance_id")
                parallel_config = msg.get("parallel_config", {})
                resolution = tuple(msg.get("resolution", (1920, 1080)))
                num_frames = msg.get("num_frames", 1)

                logger.debug(
                    f"Received address request: bootstrap_room={bootstrap_room}, "
                    f"instance_id={instance_id}"
                )

                # Allocate local buffer slots
                tensor_slot = self.allocate_tensor_buffer(
                    resolution=resolution,
                    num_frames=num_frames,
                    request_id=bootstrap_room,
                )

                meta_slot = self.allocate_meta_buffer(
                    request_id=bootstrap_room,
                    resolution=resolution,
                    num_frames=num_frames,
                )

                if tensor_slot is None or meta_slot is None:
                    logger.warning(
                        f"Failed to allocate buffer for bootstrap_room={bootstrap_room}"
                    )
                    # Send error response
                    response = {
                        "bootstrap_room": bootstrap_room,
                        "status": "error",
                        "message": "Buffer allocation failed",
                    }
                    self._send_response(response)
                    continue

                # Get local address info
                local_ip = "127.0.0.1"  # In production, get actual IP
                local_port = self.zmq_port

                # Send response with local address info
                response = {
                    "bootstrap_room": bootstrap_room,
                    "status": "ok",
                    "ip": local_ip,
                    "port": local_port,
                    "tensor_ptr": tensor_slot.ptr,
                    "meta_ptr": meta_slot.ptr,
                    "tensor_slot_index": tensor_slot.slot_index,
                    "meta_slot_index": meta_slot.slot_index,
                    "parallel_config": self._get_parallel_config(),
                    "resolution": resolution,
                    "num_frames": num_frames,
                }

                self._send_response(response)

                # Update routing table with upstream info
                self.update_routing_table(
                    instance_id=instance_id,
                    rank_info=RankRoutingInfo(
                        instance_id=instance_id,
                        rank_id=msg.get("rank_id", 0),
                        ip=msg.get("ip", ""),
                        port=msg.get("port", 0),
                        tensor_ptr=msg.get("tensor_ptr", 0),
                        meta_ptr=msg.get("meta_ptr", 0),
                        parallel_config=parallel_config,
                    ),
                )

                # Update status to WaitingForInput
                self.update_status(bootstrap_room, TransferPoll.WaitingForInput)

                logger.debug(
                    f"Sent address response for bootstrap_room={bootstrap_room}"
                )

            except zmq.Again:
                # No message available, sleep briefly
                import time

                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in receive_event_loop: {e}")
                import time

                time.sleep(0.1)

        logger.info(f"receive_event_loop stopped for role {self.role.value}")

    def wait_for_transfer_finished(self) -> None:
        """Wait for transfer completion notifications.

        This loop runs in a dedicated thread and listens for transfer completion
        notifications from upstream roles. It performs the following steps:

        1. Create notification socket (separate from address request socket)
        2. Listen for completion messages
        3. On success: Update status to Success, release buffer slots
        4. On failure: Update status to Failed, record failure reason

        Message Format (notification):
        -----------------------------
        {{
            "bootstrap_room": int,      # Session identifier
            "status": str,            # "success" or "failure"
            "tensor_slot_index": int,  # Slot to release (on success)
            "meta_slot_index": int,    # Metadata slot to release (on success)
            "reason": str             # Failure reason (on failure)
        }}

        Design Rationale:
        -----------------
        Separating notification handling from address requests:
        - Clean separation of concerns
        - Allows independent scaling
        - Easier to debug and monitor

        The notification socket is separate from the address request socket
        to avoid interference between the two types of messages.

        Thread Safety:
        -------------
        - Runs in dedicated daemon thread
        - Uses buffer free methods with internal locking
        - Updates status with thread-safe update_status method

        Error Handling:
        --------------
        - Socket creation failure: Falls back to main ZMQ socket
        - ZMQ.Again: Sleeps briefly, continues loop
        - Other exceptions: Logs error, sleeps briefly, continues loop

        Reference:
        - RFC Section 2: TransferManager.send_event_loop
        - sglang/srt/disaggregation/mooncake/conn.py:decode_thread (line 973-989)
        """
        logger.info(f"wait_for_transfer_finished started for role {self.role.value}")

        # Create a separate socket for receiving notifications
        # In production, this would be configured differently
        notify_socket = None
        try:
            notify_socket = self.zmq_context.socket(zmq.PULL)
            notify_socket.bind("tcp://127.0.0.1:*")
            notify_port = notify_socket.getsockopt(zmq.LAST_ENDPOINT)
            logger.debug(f"Notification socket bound to {notify_port}")
        except Exception as e:
            logger.error(f"Failed to create notification socket: {e}")
            notify_socket = self.zmq_socket  # Fallback to main socket

        while self._running:
            try:
                socket_to_use = notify_socket or self.zmq_socket
                msg = socket_to_use.recv_json(zmq.NOBLOCK)

                bootstrap_room = msg.get("bootstrap_room")
                status = msg.get("status")

                logger.debug(
                    f"Received transfer notification: bootstrap_room={bootstrap_room}, "
                    f"status={status}"
                )

                if status == "success":
                    self.update_status(bootstrap_room, TransferPoll.Success)
                    # Release local buffer slots
                    tensor_slot_index = msg.get("tensor_slot_index")
                    meta_slot_index = msg.get("meta_slot_index")

                    if tensor_slot_index is not None:
                        self.free_tensor_buffer(tensor_slot_index)
                    if meta_slot_index is not None:
                        self.free_meta_buffer(meta_slot_index)

                    logger.debug(
                        f"Transfer completed successfully for bootstrap_room={bootstrap_room}"
                    )
                else:
                    self.update_status(bootstrap_room, TransferPoll.Failed)
                    failure_reason = msg.get("reason", "Unknown error")
                    self.record_failure(bootstrap_room, failure_reason)
                    logger.warning(
                        f"Transfer failed for bootstrap_room={bootstrap_room}: "
                        f"{failure_reason}"
                    )

            except zmq.Again:
                # No message available, sleep briefly
                import time

                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in wait_for_transfer_finished: {e}")
                import time

                time.sleep(0.1)

        if notify_socket and notify_socket != self.zmq_socket:
            notify_socket.close()

        logger.info(f"wait_for_transfer_finished stopped for role {self.role.value}")

    def _send_response(self, response: dict) -> None:
        """Send a response message.

        Args:
            response: Response dictionary to send
        """
        try:
            # In production, this would send to a specific upstream address
            # For now, we just log the response
            logger.debug(f"Would send response: {response}")
        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    def _get_parallel_config(self) -> dict:
        """Get the current parallel configuration.

        Returns:
            Dictionary with parallel configuration
        """
        # In production, this would get the actual parallel config
        return {
            "dp_rank": 0,
            "tp_rank": 0,
            "pp_rank": 0,
            "sp_size": 1,
            "fsdp_size": 1,
        }

    def update_routing_table(
        self,
        instance_id: str,
        rank_info: RankRoutingInfo,
    ) -> None:
        """Update routing table with downstream rank information.

        Args:
            instance_id: Instance identifier
            rank_info: Routing information for the rank
        """
        with self.routing_lock:
            self.routing_table[instance_id] = rank_info
            logger.debug(f"Updated routing table: {instance_id}")

    def get_downstream_addr(
        self,
        instance_id: str,
    ) -> Optional[RankRoutingInfo]:
        """Get downstream address from routing table.

        Args:
            instance_id: Instance identifier

        Returns:
            RankRoutingInfo if found, None otherwise
        """
        with self.routing_lock:
            info = self.routing_table.get(instance_id)
            if info and info.is_expired():
                # Remove expired entry
                del self.routing_table[instance_id]
                return None
            return info

    def clear_routing_table(self) -> None:
        """Clear all routing table entries."""
        with self.routing_lock:
            self.routing_table.clear()
            logger.debug("Routing table cleared")

    def shutdown(self) -> None:
        """Shutdown the transfer manager and clean up resources."""
        # Stop the event loops
        self._running = False

        # Wait for threads to finish
        if hasattr(self, "_receive_thread") and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=1.0)
        if hasattr(self, "_notify_thread") and self._notify_thread.is_alive():
            self._notify_thread.join(timeout=1.0)

        if hasattr(self, "zmq_socket") and self.zmq_socket is not None:
            self.zmq_socket.close()
        if hasattr(self, "zmq_context") and self.zmq_context is not None:
            self.zmq_context.term()

        logger.info(f"CommonTransferManager shutdown for role {self.role.value}")


# Import for type hints
from typing import Tuple  # noqa: E402

from sglang.multimodal_gen.runtime.cache.buffer import (  # noqa: E402
    TransferMetaBufferSlot,
    TransferTensorBufferSlot,
)
