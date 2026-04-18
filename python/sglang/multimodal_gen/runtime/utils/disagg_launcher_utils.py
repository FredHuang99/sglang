"""Helpers for disaggregated launcher defaults."""

from __future__ import annotations

import json

# NOTE: This mapping follows the uploaded H200 topo information for the two
# known machines used in the cluster. Mooncake can consume a JSON GPU->IB map,
# which lets each role rank automatically pick the NIC nearest to its physical
# GPU id.
_KNOWN_HOST_TO_IB_MAP = {
    "10.3.4.3": {
        0: "mlx5_0",
        1: "mlx5_1",
        2: "mlx5_2",
        3: "mlx5_3",
        4: "mlx5_6",
        5: "mlx5_7",
        6: "mlx5_8",
        7: "mlx5_9",
    },
    "10.3.4.2": {
        0: "mlx5_0",
        1: "mlx5_1",
        2: "mlx5_2",
        3: "mlx5_3",
        4: "mlx5_6",
        5: "mlx5_7",
        6: "mlx5_8",
        7: "mlx5_9",
    },
}

_HOST_ALIASES = {
    "ac-h200-gpu03": "10.3.4.3",
    "10.3.4.3": "10.3.4.3",
    "ac-h200-gpu02": "10.3.4.2",
    "10.3.4.2": "10.3.4.2",
}


def normalize_disagg_host(host: str | None) -> str | None:
    if host is None:
        return None
    normalized = host.strip().lower()
    if not normalized:
        return None
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    return _HOST_ALIASES.get(normalized, normalized)


def build_auto_ib_device_map(host: str | None) -> dict[int, str] | None:
    normalized = normalize_disagg_host(host)
    if normalized is None:
        return None
    ib_map = _KNOWN_HOST_TO_IB_MAP.get(normalized)
    if ib_map is None:
        return None
    return dict(ib_map)


def resolve_disagg_ib_device(
    requested: str | None,
    *,
    host: str | None,
) -> str | None:
    if requested is None:
        return None

    stripped = requested.strip()
    if not stripped:
        return None
    if stripped.lower() != "auto":
        return stripped

    ib_map = build_auto_ib_device_map(host)
    if ib_map is None:
        return None
    return json.dumps({str(k): v for k, v in ib_map.items()}, separators=(",", ":"))
