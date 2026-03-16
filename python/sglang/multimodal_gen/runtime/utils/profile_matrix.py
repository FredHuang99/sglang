"""Helpers for building and validating profiling matrices."""

from __future__ import annotations


def _positive_divisors(value: int) -> list[int]:
    if value < 1:
        raise ValueError(f"value must be >= 1, got {value}")

    divisors = set()
    for candidate in range(1, int(value**0.5) + 1):
        if value % candidate == 0:
            divisors.add(candidate)
            divisors.add(value // candidate)
    return sorted(divisors)


def build_ulysses_ring_pairs(
    sp_degree: int,
    *,
    num_heads: int | None = None,
) -> list[tuple[int, int]]:
    """Return legal `(ulysses, ring)` pairs for a given SP degree.

    The list is ordered to match the usual profiling sweep, preferring larger
    Ulysses groups first:
    - 8 -> [(8, 1), (4, 2), (2, 4), (1, 8)]
    - 6 -> [(6, 1), (3, 2), (2, 3), (1, 6)]
    """

    pairs = []
    for ulysses_degree in sorted(_positive_divisors(sp_degree), reverse=True):
        if num_heads is not None and num_heads % ulysses_degree != 0:
            continue
        pairs.append((ulysses_degree, sp_degree // ulysses_degree))
    return pairs


def validate_sp_topology(
    *,
    num_gpus: int,
    sp_degree: int,
    ulysses_degree: int,
    ring_degree: int,
    num_heads: int | None = None,
) -> None:
    """Validate a DiT SP topology against current SGLang constraints."""

    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if sp_degree < 1:
        raise ValueError(f"sp_degree must be >= 1, got {sp_degree}")
    if ulysses_degree < 1:
        raise ValueError(f"ulysses_degree must be >= 1, got {ulysses_degree}")
    if ring_degree < 1:
        raise ValueError(f"ring_degree must be >= 1, got {ring_degree}")
    if ulysses_degree * ring_degree != sp_degree:
        raise ValueError(
            "ulysses_degree * ring_degree must equal sp_degree, "
            f"got {ulysses_degree} * {ring_degree} != {sp_degree}"
        )
    if sp_degree > num_gpus:
        raise ValueError(
            f"sp_degree must be <= num_gpus, got {sp_degree} > {num_gpus}"
        )
    if num_gpus % sp_degree != 0:
        raise ValueError(
            "num_gpus must be divisible by sp_degree, "
            f"got {num_gpus} % {sp_degree} != 0"
        )
    if num_heads is not None and num_heads % ulysses_degree != 0:
        raise ValueError(
            "ulysses_degree must divide the model attention head count, "
            f"got {num_heads} % {ulysses_degree} != 0"
        )
