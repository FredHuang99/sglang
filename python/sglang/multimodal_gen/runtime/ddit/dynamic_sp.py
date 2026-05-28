"""Dynamic sequence-parallel process-group registry for DDiT."""

from __future__ import annotations

import contextlib
import itertools
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    SequenceParallelGroupCoordinator,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group
from sglang.multimodal_gen.runtime.platforms import current_platform

from .config import parse_allowed_gpu_counts, parse_local_ranks, resolve_ddit_sp_degrees


@dataclass(frozen=True)
class DynamicSPGroupSpec:
    ranks: tuple[int, ...]
    ulysses_degree: int
    ring_degree: int


def _rank_groups_covering_world(active_ranks: tuple[int, ...]) -> list[list[int]]:
    world_size = dist.get_world_size()
    active = set(active_ranks)
    groups: list[list[int]] = [list(active_ranks)]
    groups.extend([rank] for rank in range(world_size) if rank not in active)
    return sorted(groups, key=lambda group: (group[0], len(group)))


def _subgroups_for_degrees(
    ranks: list[int],
    *,
    ulysses_degree: int,
    ring_degree: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if len(ranks) == 1:
        return [ranks], [ranks]
    if ulysses_degree * ring_degree != len(ranks):
        raise ValueError(
            f"Invalid SP degrees for ranks {ranks}: "
            f"{ulysses_degree}x{ring_degree}"
        )
    ulysses_groups = []
    ring_groups = []
    for ring_idx in range(ring_degree):
        start = ring_idx * ulysses_degree
        ulysses_groups.append(ranks[start : start + ulysses_degree])
    for ulysses_idx in range(ulysses_degree):
        ring_groups.append(ranks[ulysses_idx::ulysses_degree])
    return ulysses_groups, ring_groups


class DynamicSPGroupRegistry:
    """Cache SequenceParallelGroupCoordinator objects by active rank tuple."""

    def __init__(self, server_args: Any):
        self.server_args = server_args
        self._cache: dict[DynamicSPGroupSpec, SequenceParallelGroupCoordinator] = {}

    def _build(self, spec: DynamicSPGroupSpec) -> SequenceParallelGroupCoordinator:
        backend = current_platform.get_torch_distributed_backend_str()
        rank = dist.get_rank()
        group_ranks = _rank_groups_covering_world(spec.ranks)
        ulysses_pg = None
        ring_pg = None

        for group in group_ranks:
            if tuple(group) == spec.ranks:
                ulysses_degree = spec.ulysses_degree
                ring_degree = spec.ring_degree
            else:
                ulysses_degree = 1
                ring_degree = 1
            ulysses_groups, ring_groups = _subgroups_for_degrees(
                group,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
            )
            for ranks in ulysses_groups:
                pg = dist.new_group(ranks, backend=backend)
                if rank in ranks:
                    ulysses_pg = pg
            for ranks in ring_groups:
                pg = dist.new_group(ranks, backend=backend)
                if rank in ranks:
                    ring_pg = pg

        if ulysses_pg is None or ring_pg is None:
            raise RuntimeError(f"Failed to build dynamic SP subgroup for rank {rank}")

        return SequenceParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=get_world_group().local_rank,
            torch_distributed_backend=backend,
            group_name="ddit_sp_group",
            ulysses_group=ulysses_pg,
            ring_group=ring_pg,
        )

    def get(self, ranks: tuple[int, ...]) -> SequenceParallelGroupCoordinator | None:
        if not dist.is_available() or not dist.is_initialized():
            return None
        ranks = tuple(sorted(int(rank) for rank in ranks))
        ulysses_degree, ring_degree = resolve_ddit_sp_degrees(
            len(ranks),
            getattr(self.server_args, "ddit_sp_degree_map", None),
            server_args=self.server_args,
        )
        spec = DynamicSPGroupSpec(
            ranks=ranks,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
        )
        if spec not in self._cache:
            self._cache[spec] = self._build(spec)
        return self._cache[spec]

    def prebuild(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        local_ranks = parse_local_ranks(
            getattr(self.server_args, "ddit_local_ranks", None), world_size
        )
        counts = parse_allowed_gpu_counts(
            getattr(self.server_args, "ddit_allowed_gpu_counts", None), len(local_ranks)
        )
        for count in counts:
            for ranks in itertools.combinations(local_ranks, count):
                self.get(tuple(ranks))
        dist.barrier()

    @contextlib.contextmanager
    def use(self, ranks: tuple[int, ...]) -> Iterator[None]:
        if not dist.is_available() or not dist.is_initialized():
            yield
            return

        import sglang.multimodal_gen.runtime.distributed.parallel_groups as parallel_groups
        import sglang.multimodal_gen.runtime.distributed.parallel_state as parallel_state

        new_sp = self.get(tuple(ranks))
        old_sp = parallel_state._SP
        old_ulysses_pg = parallel_groups.PROCESS_GROUP.ULYSSES_PG
        old_ring_pg = parallel_groups.PROCESS_GROUP.RING_PG
        parallel_state._SP = new_sp
        parallel_groups.PROCESS_GROUP.ULYSSES_PG = new_sp.ulysses_group
        parallel_groups.PROCESS_GROUP.RING_PG = new_sp.ring_group
        try:
            yield
        finally:
            parallel_state._SP = old_sp
            parallel_groups.PROCESS_GROUP.ULYSSES_PG = old_ulysses_pg
            parallel_groups.PROCESS_GROUP.RING_PG = old_ring_pg


_REGISTRY_CACHE: dict[int, DynamicSPGroupRegistry] = {}


def get_dynamic_sp_registry(server_args: Any) -> DynamicSPGroupRegistry:
    key = id(server_args)
    registry = _REGISTRY_CACHE.get(key)
    if registry is None:
        registry = DynamicSPGroupRegistry(server_args)
        _REGISTRY_CACHE[key] = registry
    return registry


@contextlib.contextmanager
def use_dynamic_sp_group(server_args: Any, ranks: tuple[int, ...]) -> Iterator[None]:
    registry = get_dynamic_sp_registry(server_args)
    with registry.use(tuple(ranks)):
        yield


def prebuild_dynamic_sp_groups(server_args: Any) -> None:
    if getattr(server_args, "enable_ddit", False) and getattr(
        server_args, "ddit_prebuild_sp_groups", True
    ):
        get_dynamic_sp_registry(server_args).prebuild()


def current_rank_in(ranks: tuple[int, ...]) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() in set(ranks)


def broadcast_tensor_from_rank(tensor: torch.Tensor, src: int) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    dist.broadcast(tensor, src=src, group=get_world_group().device_group)
    return tensor
