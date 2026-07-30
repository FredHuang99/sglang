"""Strict world-size-one coordinator for PyTorch builds without C10d.

Some NVIDIA Jetson PyTorch builds expose a ``torch.distributed`` stub but are
compiled without C10d. SGLang Diffusion still needs group-shaped objects for
its world-size-one tensor, sequence, pipeline, CFG, data, and VAE-decode code
paths. This module provides only those identity semantics; it never emulates
communication between ranks.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform


class LocalSingleProcessGroupCoordinator:
    """GroupCoordinator-compatible identity operations for exactly one rank."""

    def __init__(
        self,
        group_ranks: list[list[int]] | None = None,
        local_rank: int = 0,
        torch_distributed_backend: str = "local",
        group_name: str | None = None,
        ulysses_group: Any | None = None,
        ring_group: Any | None = None,
        **_ignored: Any,
    ) -> None:
        group_ranks = group_ranks or [[0]]
        if group_ranks != [[0]]:
            raise RuntimeError(
                "The local SGLang Diffusion coordinator supports only rank 0 "
                f"with group_ranks=[[0]], got {group_ranks}."
            )
        if local_rank != 0:
            raise RuntimeError(
                "The local SGLang Diffusion coordinator requires local_rank=0, "
                f"got {local_rank}."
            )

        self.unique_name = f"{group_name or 'local'}:0"
        self.backend = torch_distributed_backend
        self.rank = 0
        self.ranks = [0]
        self.world_size = 1
        self.local_rank = 0
        self.rank_in_group = 0
        self.group_first_rank = 0
        self.group_last_rank = 0
        self.group_next_rank = 0
        self.group_prev_rank = 0
        self.device = current_platform.get_local_torch_device()

        # No ProcessGroup or communicator is created in local mode.
        self.device_group = None
        self.cpu_group = None
        self.device_communicator = None
        self.srt_custom_allreduce = None
        self.mq_broadcaster = None

        # Sequence-parallel callers expect these attributes even at SP=1.
        self.ulysses_group = ulysses_group or self
        self.ring_group = ring_group or self
        self.ulysses_world_size = 1
        self.ulysses_rank = 0
        self.ring_world_size = 1
        self.ring_rank = 0

    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        return input_

    def all_reduce(
        self,
        input_: torch.Tensor,
        op: Any | None = None,
        async_op: bool = False,
    ) -> torch.Tensor:
        del op, async_op
        return input_

    def all_gather(
        self,
        input_: torch.Tensor,
        dim: int = 0,
        separate_tensors: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        del dim
        return [input_] if separate_tensors else input_

    def all_to_all_4D(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 2,
        gather_dim: int = 1,
    ) -> torch.Tensor:
        del scatter_dim, gather_dim
        return input_

    def gather(
        self,
        input_: torch.Tensor,
        dst: int = 0,
        dim: int = -1,
    ) -> torch.Tensor:
        del dim
        self._validate_local_rank(dst, "destination")
        return input_

    def broadcast(
        self,
        input_: torch.Tensor,
        src: int = 0,
        async_op: bool = False,
    ) -> torch.Tensor:
        del async_op
        self._validate_local_rank(src, "source")
        return input_

    def broadcast_object(self, obj: Any = None, src: int = 0) -> Any:
        self._validate_local_rank(src, "source")
        return obj

    def broadcast_object_list(
        self,
        obj_list: list[Any],
        src: int = 0,
        group: Any | None = None,
    ) -> list[Any]:
        del group
        self._validate_local_rank(src, "source")
        return obj_list

    def broadcast_tensor_dict(
        self,
        tensor_dict: dict[str, Any] | None = None,
        src: int = 0,
        group: Any | None = None,
        metadata_group: Any | None = None,
    ) -> dict[str, Any] | None:
        del group, metadata_group
        self._validate_local_rank(src, "source")
        return tensor_dict

    def barrier(self) -> None:
        return None

    @contextmanager
    def graph_capture(self, graph_capture_context: Any | None = None):
        yield graph_capture_context

    def send_object(self, obj: Any, dst: int) -> None:
        del obj, dst
        self._raise_peer_to_peer()

    def recv_object(self, src: int) -> Any:
        del src
        self._raise_peer_to_peer()

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, Any],
        dst: int | None = None,
    ) -> None:
        del tensor_dict, dst
        self._raise_peer_to_peer()

    def recv_tensor_dict(self, src: int | None = None) -> dict[str, Any]:
        del src
        self._raise_peer_to_peer()

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        del tensor, dst
        self._raise_peer_to_peer()

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        src: int | None = None,
    ) -> torch.Tensor:
        del size, dtype, src
        self._raise_peer_to_peer()

    def destroy(self) -> None:
        return None

    @staticmethod
    def _validate_local_rank(rank: int, description: str) -> None:
        if rank != 0:
            raise RuntimeError(
                "The local SGLang Diffusion coordinator has only rank 0; "
                f"invalid {description} rank {rank}."
            )

    @staticmethod
    def _raise_peer_to_peer() -> None:
        raise RuntimeError(
            "Peer-to-peer communication is unavailable in the strict "
            "world-size-one local SGLang Diffusion backend."
        )


LocalPipelineGroupCoordinator = LocalSingleProcessGroupCoordinator
LocalSequenceParallelGroupCoordinator = LocalSingleProcessGroupCoordinator
