"""Configuration parsing helpers for elastic DDiT execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DDiTSwitchEvent:
    """A rank-set transition after a completed denoising step."""

    after_step: int
    ranks: tuple[int, ...]
    reason: str = "switch_plan"


@dataclass(frozen=True)
class DDiTExecutionPlan:
    """Normalized per-request DDiT rank plan."""

    initial_ranks: tuple[int, ...]
    switches: tuple[DDiTSwitchEvent, ...]

    @property
    def enabled(self) -> bool:
        return bool(self.switches)

    def switch_after(self, completed_step: int) -> DDiTSwitchEvent | None:
        for event in self.switches:
            if event.after_step == completed_step:
                return event
        return None


def _as_extra_value(batch: Any, key: str, default: Any = None) -> Any:
    if hasattr(batch, key):
        return getattr(batch, key)
    extra = getattr(batch, "extra", None)
    if isinstance(extra, dict):
        return extra.get(key, default)
    return default


def _unique_sorted_ranks(ranks: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    parsed = tuple(sorted(int(rank) for rank in ranks))
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"DDiT ranks must be unique, got {list(ranks)}")
    if any(rank < 0 for rank in parsed):
        raise ValueError(f"DDiT ranks must be non-negative, got {list(ranks)}")
    return parsed


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def parse_rank_list(value: Any) -> tuple[int, ...]:
    """Parse a rank list from list/tuple/int/string forms."""
    if value is None or value == "":
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)):
        return _unique_sorted_ranks([int(v) for v in value])
    if isinstance(value, str):
        tokens = [token for token in value.replace("|", ",").split(",") if token]
        return _unique_sorted_ranks([int(token.strip()) for token in tokens])
    raise TypeError(f"Unsupported rank list type: {type(value)!r}")


def _expand_count_transition(spec: str) -> tuple[int, ...]:
    """Expand '1->4' into ranks [0, 1, 2, 3]."""
    _old, new_count = spec.split("->", 1)
    count = int(new_count.strip())
    if count <= 0:
        raise ValueError(f"DDiT transition target must be positive, got {spec!r}")
    return tuple(range(count))


def parse_switch_plan(value: Any) -> tuple[DDiTSwitchEvent, ...]:
    """Parse switch plans from strings or JSON-like objects.

    Supported string examples:
      - "15:0,1;30:0,1,2,3;45:0,1,2,3,4,5,6,7"
      - "15:1->2;30:2->4;45:4->8"
    Supported list example:
      - [{"after_step": 15, "ranks": [0, 1]}]
    """
    if value is None or value == "":
        return ()

    events: list[DDiTSwitchEvent] = []
    if isinstance(value, str):
        for raw_event in [part.strip() for part in value.split(";") if part.strip()]:
            if ":" not in raw_event:
                raise ValueError(f"DDiT switch event must contain ':': {raw_event!r}")
            step_text, ranks_text = raw_event.split(":", 1)
            after_step = int(step_text.strip())
            ranks = (
                _expand_count_transition(ranks_text)
                if "->" in ranks_text
                else parse_rank_list(ranks_text)
            )
            events.append(DDiTSwitchEvent(after_step=after_step, ranks=ranks))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, DDiTSwitchEvent):
                events.append(item)
                continue
            if not isinstance(item, dict):
                raise TypeError(f"DDiT switch event must be a dict, got {type(item)!r}")
            after_step = int(item.get("after_step", item.get("step")))
            ranks = parse_rank_list(item.get("ranks", item.get("new_ranks")))
            reason = str(item.get("reason", "switch_plan"))
            events.append(
                DDiTSwitchEvent(after_step=after_step, ranks=ranks, reason=reason)
            )
    else:
        raise TypeError(f"Unsupported DDiT switch plan type: {type(value)!r}")

    normalized = []
    seen_steps = set()
    for event in sorted(events, key=lambda e: e.after_step):
        if event.after_step <= 0:
            raise ValueError(f"DDiT after_step must be positive, got {event.after_step}")
        if event.after_step in seen_steps:
            raise ValueError(f"Duplicate DDiT switch step: {event.after_step}")
        if not event.ranks:
            raise ValueError(f"DDiT switch at step {event.after_step} has no ranks")
        seen_steps.add(event.after_step)
        normalized.append(
            DDiTSwitchEvent(
                after_step=event.after_step,
                ranks=_unique_sorted_ranks(event.ranks),
                reason=event.reason,
            )
        )
    return tuple(normalized)


def _first_n_ranks(count: int, world_size: int) -> tuple[int, ...]:
    count = max(1, min(int(count), int(world_size)))
    return tuple(range(count))


def validate_node_local_power_of_two(
    ranks: tuple[int, ...],
    *,
    local_world_size: int,
    allowed_gpu_counts: tuple[int, ...],
) -> None:
    if len(ranks) not in allowed_gpu_counts:
        raise ValueError(
            f"DDiT rank count must be one of {allowed_gpu_counts}, got {len(ranks)}"
        )
    if not is_power_of_two(len(ranks)):
        raise ValueError(f"DDiT rank count must be a power of two, got {len(ranks)}")
    bad = [rank for rank in ranks if rank >= local_world_size]
    if bad:
        raise ValueError(
            f"DDiT ranks must stay within the local node world size "
            f"{local_world_size}, got {bad}"
        )


def parse_allowed_gpu_counts(value: Any, world_size: int) -> tuple[int, ...]:
    if value is None or value == "":
        counts = [1, 2, 4, 8]
    elif isinstance(value, str):
        counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        counts = [int(part) for part in value]
    counts = sorted({count for count in counts if 1 <= count <= world_size})
    return tuple(counts or [1])


def parse_local_ranks(value: Any, world_size: int) -> tuple[int, ...]:
    ranks = parse_rank_list(value)
    if not ranks:
        ranks = tuple(range(world_size))
    bad = [rank for rank in ranks if rank >= world_size]
    if bad:
        raise ValueError(f"DDiT local ranks exceed world size {world_size}: {bad}")
    return ranks


def parse_sp_degree_map(value: Any) -> dict[int, tuple[int, int]]:
    """Parse '1=1x1,2=2x1,4=2x2' into {k: (ulysses, ring)}."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        parsed = {}
        for key, degree in value.items():
            if isinstance(degree, str):
                ulysses, ring = degree.lower().split("x", 1)
                parsed[int(key)] = (int(ulysses), int(ring))
            else:
                parsed[int(key)] = (int(degree[0]), int(degree[1]))
        return parsed
    parsed = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        key, degree = item.split("=", 1)
        ulysses, ring = degree.lower().split("x", 1)
        parsed[int(key.strip())] = (int(ulysses.strip()), int(ring.strip()))
    return parsed


def resolve_sp_degrees(rank_count: int, degree_map: Any = None) -> tuple[int, int]:
    parsed_map = parse_sp_degree_map(degree_map)
    ulysses, ring = parsed_map.get(rank_count, (rank_count, 1))
    if ulysses * ring != rank_count:
        raise ValueError(
            f"DDiT SP degree product must match rank count: "
            f"{ulysses}x{ring} != {rank_count}"
        )
    return ulysses, ring


def build_execution_plan(
    server_args: Any,
    batch: Any,
    *,
    world_size: int,
) -> DDiTExecutionPlan:
    allowed = parse_allowed_gpu_counts(
        getattr(server_args, "ddit_allowed_gpu_counts", None), world_size
    )
    local_rank_tuple = parse_local_ranks(
        getattr(server_args, "ddit_local_ranks", None), world_size
    )
    local_ranks = set(local_rank_tuple)
    initial_ranks = parse_rank_list(_as_extra_value(batch, "ddit_initial_ranks"))
    if not initial_ranks:
        initial_ranks = parse_rank_list(getattr(server_args, "ddit_initial_ranks", None))
    if not initial_ranks:
        count = max(
            1,
            min(int(getattr(server_args, "ddit_initial_gpus", 1)), len(local_rank_tuple)),
        )
        initial_ranks = tuple(local_rank_tuple[:count])

    switch_plan = parse_switch_plan(_as_extra_value(batch, "ddit_switch_plan"))
    if not switch_plan:
        switch_plan = parse_switch_plan(getattr(server_args, "ddit_switch_plan", None))

    validate_node_local_power_of_two(
        initial_ranks, local_world_size=world_size, allowed_gpu_counts=allowed
    )
    if not set(initial_ranks).issubset(local_ranks):
        raise ValueError(
            f"DDiT initial ranks {initial_ranks} must stay within local ranks "
            f"{tuple(sorted(local_ranks))}"
        )
    for event in switch_plan:
        validate_node_local_power_of_two(
            event.ranks, local_world_size=world_size, allowed_gpu_counts=allowed
        )
        if not set(event.ranks).issubset(local_ranks):
            raise ValueError(
                f"DDiT switch ranks {event.ranks} must stay within local ranks "
                f"{tuple(sorted(local_ranks))}"
            )
    return DDiTExecutionPlan(initial_ranks=initial_ranks, switches=switch_plan)


def resolve_vae_ranks(
    server_args: Any,
    batch: Any,
    *,
    world_size: int,
    final_dit_ranks: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    explicit_ranks = parse_rank_list(_as_extra_value(batch, "ddit_vae_ranks"))
    if explicit_ranks:
        return explicit_ranks

    request_k = _as_extra_value(batch, "ddit_vae_k")
    vae_k = int(request_k if request_k is not None else getattr(server_args, "ddit_vae_gpus", 1))
    local_rank_tuple = parse_local_ranks(
        getattr(server_args, "ddit_local_ranks", None), world_size
    )
    if final_dit_ranks and len(final_dit_ranks) >= vae_k:
        ranks = tuple(sorted(final_dit_ranks))[:vae_k]
    else:
        count = max(1, min(vae_k, len(local_rank_tuple)))
        ranks = tuple(local_rank_tuple[:count])

    allowed = parse_allowed_gpu_counts(
        getattr(server_args, "ddit_allowed_gpu_counts", None), world_size
    )
    local_ranks = set(local_rank_tuple)
    validate_node_local_power_of_two(
        ranks, local_world_size=world_size, allowed_gpu_counts=allowed
    )
    if not set(ranks).issubset(local_ranks):
        raise ValueError(
            f"DDiT VAE ranks {ranks} must stay within local ranks "
            f"{tuple(sorted(local_ranks))}"
        )
    return ranks


def resolve_resolution_key(batch: Any) -> str:
    explicit = _as_extra_value(batch, "resolution_key")
    if explicit is None:
        explicit = _as_extra_value(batch, "ddit_resolution_key")
    if explicit:
        return str(explicit)
    height = getattr(batch, "height", None)
    width = getattr(batch, "width", None)
    if height and width:
        short_edge = min(int(height), int(width))
        return f"{short_edge}p"
    return "unknown"
