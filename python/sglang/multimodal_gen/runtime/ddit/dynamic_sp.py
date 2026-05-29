"""Dynamic sequence-parallel process-group registry for DDiT."""

from __future__ import annotations

import contextlib
import inspect
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator import (
    DistributedAutograd,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .config import (
    parse_allowed_gpu_counts,
    parse_local_ranks,
    parse_rank_list,
    parse_switch_plan,
    resolve_ddit_sp_degrees,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class DynamicSPGroupSpec:
    ranks: tuple[int, ...]
    ulysses_degree: int
    ring_degree: int


@dataclass
class DynamicSPBuildStats:
    created_process_groups: int = 0
    reused_process_groups: int = 0
    build_ms: float = 0.0
    new_group_ms: float = 0.0


@dataclass(frozen=True)
class DynamicSPEnsureResult:
    spec: DynamicSPGroupSpec
    group: Any | None
    created: bool
    stats: DynamicSPBuildStats = field(default_factory=DynamicSPBuildStats)


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


def _all_rank_tuples(
    local_ranks: tuple[int, ...], allowed_counts: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    tuples: list[tuple[int, ...]] = []
    for count in allowed_counts:
        tuples.extend(
            tuple(combo) for combo in itertools.combinations(local_ranks, count)
        )
    return tuple(tuples)


def _prebuild_rank_tuples(
    server_args: Any, world_size: int
) -> tuple[tuple[int, ...], ...]:
    local_ranks = parse_local_ranks(
        getattr(server_args, "ddit_local_ranks", None), world_size
    )
    allowed_counts = set(
        parse_allowed_gpu_counts(
            getattr(server_args, "ddit_allowed_gpu_counts", None), len(local_ranks)
        )
    )
    local_rank_set = set(local_ranks)
    rank_tuples: list[tuple[int, ...]] = []
    mode = str(
        getattr(server_args, "ddit_dynamic_sp_prebuild_mode", "auto") or "auto"
    ).lower()
    if mode == "off":
        return ()
    if mode == "all":
        return _all_rank_tuples(local_ranks, tuple(sorted(allowed_counts)))

    def add_candidate(ranks: tuple[int, ...] | list[int]) -> None:
        normalized = tuple(sorted(int(rank) for rank in ranks))
        if not normalized:
            return
        if len(normalized) not in allowed_counts:
            return
        if len(set(normalized)) != len(normalized):
            return
        if not set(normalized).issubset(local_rank_set):
            return
        rank_tuples.append(normalized)

    schedule_policy = str(getattr(server_args, "ddit_schedule_policy", "") or "")

    initial_ranks = parse_rank_list(getattr(server_args, "ddit_initial_ranks", None))
    if initial_ranks:
        add_candidate(initial_ranks)
    else:
        initial_count = max(
            1,
            min(int(getattr(server_args, "ddit_initial_gpus", 1)), len(local_ranks)),
        )
        add_candidate(local_ranks[:initial_count])

    baseline_ranks = parse_rank_list(getattr(server_args, "ddit_baseline_ranks", None))
    if baseline_ranks:
        add_candidate(baseline_ranks)
    else:
        baseline_gpus = getattr(server_args, "ddit_baseline_gpus", None)
        if baseline_gpus is not None:
            baseline_count = max(1, min(int(baseline_gpus), len(local_ranks)))
            add_candidate(local_ranks[:baseline_count])

    switch_plan = parse_switch_plan(getattr(server_args, "ddit_switch_plan", None))
    for event in switch_plan:
        add_candidate(event.ranks)

    vae_ranks = parse_rank_list(getattr(server_args, "ddit_vae_ranks", None))
    if vae_ranks:
        add_candidate(vae_ranks)
    else:
        vae_gpus = getattr(server_args, "ddit_vae_gpus", None)
        if vae_gpus is not None:
            vae_count = max(1, min(int(vae_gpus), len(local_ranks)))
            add_candidate(local_ranks[:vae_count])

    # Keep auto prebuild bounded. Forced-switch correctness usually knows its
    # explicit plan up front; E2E policies need only canonical startup coverage.
    needs_canonical_defaults = mode == "canonical" or (
        mode == "auto"
        and (schedule_policy != "forced_switch" or not switch_plan)
    )
    if needs_canonical_defaults:
        for count in sorted(allowed_counts):
            add_candidate(local_ranks[:count])

    return tuple(sorted(set(rank_tuples), key=lambda ranks: (len(ranks), ranks)))


def prebuild_rank_tuples_for_server(
    server_args: Any, world_size: int
) -> tuple[tuple[int, ...], ...]:
    return _prebuild_rank_tuples(server_args, world_size)


class LightweightDynamicSPCoordinator:
    """Minimal SP process-group wrapper for DDiT dynamic rank sets."""

    def __init__(
        self,
        *,
        ranks: tuple[int, ...],
        local_rank: int,
        device_group: Any | None,
        ulysses_group: Any | None,
        ring_group: Any | None,
        ulysses_ranks: tuple[int, ...] | None,
        ring_ranks: tuple[int, ...] | None,
    ):
        self.rank = dist.get_rank()
        self.local_rank = local_rank
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.rank_in_group = self.ranks.index(self.rank) if self.rank in ranks else -1
        self.device_group = device_group
        self.cpu_group = None
        self.ulysses_group = ulysses_group
        self.ring_group = ring_group
        self.ulysses_world_size = len(ulysses_ranks or ())
        self.ulysses_rank = (
            tuple(ulysses_ranks).index(self.rank)
            if ulysses_ranks is not None and self.rank in ulysses_ranks
            else -1
        )
        self.ring_world_size = len(ring_ranks or ())
        self.ring_rank = (
            tuple(ring_ranks).index(self.rank)
            if ring_ranks is not None and self.rank in ring_ranks
            else -1
        )

    def _require_member(self) -> None:
        if self.rank_in_group < 0 or self.device_group is None:
            raise RuntimeError(
                f"Rank {self.rank} is not a member of dynamic SP ranks {self.ranks}"
            )

    def all_to_all_4D(
        self, input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        self._require_member()
        return DistributedAutograd.AllToAll4D.apply(
            self.device_group, input_, self.world_size, scatter_dim, gather_dim
        )

    def all_reduce(
        self,
        input_: torch.Tensor,
        op=torch._C._distributed_c10d.ReduceOp.SUM,
        async_op: bool = False,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        self._require_member()
        dist.all_reduce(input_, op=op, group=self.device_group, async_op=async_op)
        return input_

    def all_gather(
        self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        world_size = self.world_size
        if world_size == 1:
            return [input_] if separate_tensors else input_
        self._require_member()
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            dim += input_.dim()
        input_size = list(input_.size())
        input_size[0] *= world_size
        output_tensor = torch.empty(
            input_size, dtype=input_.dtype, device=input_.device
        )
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        if dim != 0:
            input_size[0] //= world_size
            output_tensor = output_tensor.reshape([world_size] + input_size)
            output_tensor = output_tensor.movedim(0, dim)

        if separate_tensors:
            return [
                output_tensor.reshape(-1)
                .narrow(0, input_.numel() * i, input_.numel())
                .view_as(input_)
                for i in range(world_size)
            ]

        input_size = list(input_.size())
        input_size[dim] = input_size[dim] * world_size
        return output_tensor.reshape(input_size)

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        world_size = self.world_size
        if world_size == 1:
            return input_
        self._require_member()
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            dim += input_.dim()
        gather_list = (
            [torch.empty_like(input_) for _ in range(world_size)]
            if self.rank_in_group == dst
            else None
        )
        dist.gather(input_, gather_list, dst=self.ranks[dst], group=self.device_group)
        if self.rank_in_group == dst:
            return torch.cat(gather_list, dim=dim)
        return None

    def broadcast(
        self, input_: torch.Tensor, src: int = 0, async_op: bool = False
    ) -> torch.Tensor:
        assert src < self.world_size, f"Invalid src rank ({src})"
        if self.world_size == 1:
            return input_
        self._require_member()
        dist.broadcast(
            input_, src=self.ranks[src], group=self.device_group, async_op=async_op
        )
        return input_

    def destroy(self) -> None:
        return None


class DynamicSPGroupRegistry:
    """Cache lightweight dynamic SP coordinators by active rank tuple."""

    def __init__(self, server_args: Any):
        self.server_args = server_args
        self._cache: dict[DynamicSPGroupSpec, Any] = {}
        self._device_pg_cache: dict[tuple[int, ...], Any] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def resolve_spec(self, ranks: tuple[int, ...]) -> DynamicSPGroupSpec:
        ranks = tuple(sorted(int(rank) for rank in ranks))
        ulysses_degree, ring_degree = resolve_ddit_sp_degrees(
            len(ranks),
            getattr(self.server_args, "ddit_sp_degree_map", None),
            server_args=self.server_args,
        )
        return DynamicSPGroupSpec(
            ranks=ranks,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
        )

    def has(self, ranks: tuple[int, ...]) -> bool:
        spec = self.resolve_spec(ranks)
        return spec in self._cache or self._static_sp_group_for_spec(spec) is not None

    def ensure(self, ranks: tuple[int, ...]) -> DynamicSPEnsureResult:
        spec = self.resolve_spec(ranks)
        if not dist.is_available() or not dist.is_initialized():
            return DynamicSPEnsureResult(spec=spec, group=None, created=False)
        group = self._cache.get(spec)
        if group is not None:
            return DynamicSPEnsureResult(spec=spec, group=group, created=False)
        static_group = self._static_sp_group_for_spec(spec)
        if static_group is not None:
            self._cache[spec] = static_group
            return DynamicSPEnsureResult(
                spec=spec,
                group=static_group,
                created=False,
                stats=DynamicSPBuildStats(reused_process_groups=3),
            )
        started = time.perf_counter()
        group, stats = self._build(spec)
        stats = DynamicSPBuildStats(
            created_process_groups=stats.created_process_groups,
            reused_process_groups=stats.reused_process_groups,
            build_ms=(time.perf_counter() - started) * 1000.0,
            new_group_ms=stats.new_group_ms,
        )
        self._cache[spec] = group
        return DynamicSPEnsureResult(spec=spec, group=group, created=True, stats=stats)

    def _build(
        self, spec: DynamicSPGroupSpec
    ) -> tuple[LightweightDynamicSPCoordinator, DynamicSPBuildStats]:
        backend = current_platform.get_torch_distributed_backend_str()
        rank = dist.get_rank()
        stats = DynamicSPBuildStats()
        active_group = list(spec.ranks)
        ulysses_groups, ring_groups = _subgroups_for_degrees(
            active_group,
            ulysses_degree=spec.ulysses_degree,
            ring_degree=spec.ring_degree,
        )
        active_pg = self._get_or_create_device_pg(spec.ranks, backend, stats)
        ulysses_pg = None
        ring_pg = None
        current_ulysses_ranks = None
        current_ring_ranks = None
        for ranks in ulysses_groups:
            ranks_tuple = tuple(ranks)
            pg = self._get_or_create_device_pg(ranks_tuple, backend, stats)
            if rank in ranks_tuple:
                ulysses_pg = pg
                current_ulysses_ranks = ranks_tuple
        for ranks in ring_groups:
            ranks_tuple = tuple(ranks)
            pg = self._get_or_create_device_pg(ranks_tuple, backend, stats)
            if rank in ranks_tuple:
                ring_pg = pg
                current_ring_ranks = ranks_tuple

        if rank in spec.ranks:
            if current_ulysses_ranks is None or current_ring_ranks is None:
                raise RuntimeError(
                    f"Failed to build dynamic SP subgroup for rank {rank}"
                )
            group = LightweightDynamicSPCoordinator(
                ranks=spec.ranks,
                local_rank=get_world_group().local_rank,
                device_group=active_pg,
                ulysses_group=ulysses_pg,
                ring_group=ring_pg,
                ulysses_ranks=current_ulysses_ranks,
                ring_ranks=current_ring_ranks,
            )
        else:
            # Preserve the previous inactive-rank semantics without creating
            # one-rank NCCL/Gloo groups: non-members locally observe SP size 1.
            group = LightweightDynamicSPCoordinator(
                ranks=(rank,),
                local_rank=get_world_group().local_rank,
                device_group=None,
                ulysses_group=None,
                ring_group=None,
                ulysses_ranks=(rank,),
                ring_ranks=(rank,),
            )
        return group, stats

    def _get_or_create_device_pg(
        self,
        ranks: tuple[int, ...],
        backend: str,
        stats: DynamicSPBuildStats,
    ) -> Any | None:
        ranks = tuple(sorted(int(rank) for rank in ranks))
        if len(ranks) <= 1:
            return None
        if ranks in self._device_pg_cache:
            cached = self._device_pg_cache[ranks]
            stats.reused_process_groups += 1
            return cached

        kwargs = {"ranks": list(ranks), "backend": backend}
        kwargs.update(_new_group_optional_kwargs())
        started = time.perf_counter()
        pg = dist.new_group(**kwargs)
        stats.new_group_ms += (time.perf_counter() - started) * 1000.0
        stats.created_process_groups += 1
        self._device_pg_cache[ranks] = pg
        return pg

    def _static_sp_group_for_spec(self, spec: DynamicSPGroupSpec) -> Any | None:
        if not dist.is_available() or not dist.is_initialized():
            return None
        try:
            import sglang.multimodal_gen.runtime.distributed.parallel_state as parallel_state

            static_group = parallel_state.get_sp_group()
        except (AssertionError, RuntimeError):
            return None
        if tuple(sorted(int(rank) for rank in static_group.ranks)) != spec.ranks:
            return None
        if int(static_group.ulysses_world_size) != spec.ulysses_degree:
            return None
        if int(static_group.ring_world_size) != spec.ring_degree:
            return None
        return static_group

    def get(self, ranks: tuple[int, ...]) -> Any | None:
        return self.ensure(ranks).group

    def prebuild(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_ranks = parse_local_ranks(
            getattr(self.server_args, "ddit_local_ranks", None), world_size
        )
        counts = parse_allowed_gpu_counts(
            getattr(self.server_args, "ddit_allowed_gpu_counts", None), len(local_ranks)
        )
        rank_tuples = _prebuild_rank_tuples(self.server_args, world_size)
        total_groups = len(rank_tuples)
        if rank == 0:
            mode = str(
                getattr(self.server_args, "ddit_dynamic_sp_prebuild_mode", "auto")
                or "auto"
            ).lower()
            if mode == "plan" and not parse_switch_plan(
                getattr(self.server_args, "ddit_switch_plan", None)
            ):
                logger.warning(
                    "DDiT dynamic SP prebuild mode=plan has no server-level "
                    "ddit_switch_plan; request-level plans will be built lazily"
                )
            if str(
                getattr(self.server_args, "ddit_dynamic_sp_prebuild_mode", "auto")
                or "auto"
            ).lower() == "all":
                logger.warning(
                    "DDiT dynamic SP prebuild mode=all is for debugging only; "
                    "it may create many NCCL communicators and exhaust resources."
                )
            logger.info(
                "DDiT dynamic SP prebuild start: role=%s, world_size=%s, "
                "local_ranks=%s, counts=%s, total_rank_tuples=%s, "
                "rank_tuples=%s, degree_map=%s, mode=%s",
                _ddit_role_value(self.server_args),
                world_size,
                local_ranks,
                counts,
                total_groups,
                rank_tuples,
                getattr(self.server_args, "ddit_sp_degree_map", None),
                getattr(self.server_args, "ddit_dynamic_sp_prebuild_mode", "auto"),
            )

        for idx, ranks in enumerate(rank_tuples, start=1):
            if rank == 0:
                logger.info(
                    "DDiT dynamic SP prebuild progress: %s/%s ranks=%s",
                    idx,
                    total_groups,
                    ranks,
                )
            self.get(tuple(ranks))

        if total_groups == 0 and rank == 0:
            logger.warning(
                "DDiT dynamic SP prebuild found no rank tuples; "
                "runtime full-rank ensure will build groups lazily"
            )

        if dist.is_available() and dist.is_initialized():
            if rank == 0:
                logger.info("DDiT dynamic SP prebuild entering final barrier")
            dist.barrier()
        if rank == 0:
            logger.info(
                "DDiT dynamic SP prebuild done: cache_size=%s, "
                "process_group_cache_size=%s, total_rank_tuples=%s, "
                "allowed_counts=%s",
                self.cache_size,
                len(self._device_pg_cache),
                total_groups,
                counts,
            )

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
    if should_prebuild_dynamic_sp_groups(server_args):
        get_dynamic_sp_registry(server_args).prebuild()


def _new_group_optional_kwargs() -> dict[str, Any]:
    try:
        parameters = inspect.signature(dist.new_group).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, Any] = {}
    # Dynamic DDiT builds subset groups through a full-rank ordered ensure wave.
    # Keep global synchronization semantics so non-member ranks participate in
    # the same new_group ordering; local synchronization can deadlock here.
    if (
        "device_id" in parameters
        and current_platform.is_cuda_alike()
        and torch.cuda.is_available()
    ):
        kwargs["device_id"] = torch.device("cuda", torch.cuda.current_device())
    return kwargs


def _ddit_role_value(server_args: Any) -> str:
    role = getattr(server_args, "disagg_role", None)
    if role is None:
        return "monolithic"
    return getattr(role, "value", str(role)).lower()


def should_prebuild_dynamic_sp_groups(server_args: Any) -> bool:
    if not getattr(server_args, "enable_ddit", False):
        _log_prebuild_skip(server_args, "enable_ddit=false")
        return False
    mode = str(
        getattr(server_args, "ddit_dynamic_sp_prebuild_mode", "auto") or "auto"
    ).lower()
    if mode not in {"off", "auto", "plan", "canonical", "all"}:
        raise ValueError(
            "--ddit-dynamic-sp-prebuild-mode must be one of "
            "{off,auto,plan,canonical,all}, got "
            f"{mode!r}"
        )
    if mode == "off":
        _log_prebuild_skip(server_args, "mode=off")
        return False
    if not getattr(server_args, "ddit_prebuild_sp_groups", True):
        _log_prebuild_skip(server_args, "disabled_by_arg")
        return False

    role_value = _ddit_role_value(server_args)
    if role_value in {"encoder", "server"}:
        _log_prebuild_skip(server_args, f"role={role_value}")
        return False
    return True


def _log_prebuild_skip(server_args: Any, reason: str) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    logger.info(
        "Skipping DDiT dynamic SP prebuild: reason=%s, role=%s, "
        "enable_ddit=%s, ddit_prebuild_sp_groups=%s, mode=%s",
        reason,
        _ddit_role_value(server_args),
        getattr(server_args, "enable_ddit", False),
        getattr(server_args, "ddit_prebuild_sp_groups", True),
        getattr(server_args, "ddit_dynamic_sp_prebuild_mode", "auto"),
    )


def current_rank_in(ranks: tuple[int, ...]) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() in set(ranks)


def broadcast_tensor_from_rank(tensor: torch.Tensor, src: int) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    dist.broadcast(tensor, src=src, group=get_world_group().device_group)
    return tensor
