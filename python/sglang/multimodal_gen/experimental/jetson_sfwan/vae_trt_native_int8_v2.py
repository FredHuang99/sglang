"""Manifest and graph contracts for the isolated Native INT8 V2 VAE path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .vae_trt_fusion import sha256_file
from .vae_trt_native_int8 import (
    EXPECTED_CALL_SITES_PER_GRAPH,
    EXPECTED_FP16_CACHE_SLOTS,
    EXPECTED_INT8_CACHE_SLOTS,
    EXPECTED_RESIDUAL_BLOCKS,
    EXPECTED_SIGNATURES,
    load_native_int8_manifest,
    validate_native_int8_manifest,
)

NATIVE_INT8_V2_VARIANT = "native_int8_v2"
NATIVE_INT8_V2_SCHEMA_VERSION = 2
NATIVE_INT8_V2_SUBDIRECTORY = "native_int8_v2"
NATIVE_INT8_V2_LEVELS = ("p1", "p2", "p3")
NATIVE_INT8_V2_PLUGIN_LIBRARY_FILE = "libsfwan_vae_native_int8_v2.so"
NATIVE_INT8_V2_PLUGIN_NAME = "SfWanNativeInt8V2ResidualBlockPlugin"
NATIVE_INT8_V2_PLUGIN_NAMESPACE = "sglang.sfwan"
NATIVE_INT8_V2_PLUGIN_VERSION = "1"
NATIVE_INT8_V2_PLUGIN_INIT_SYMBOL = "initSfWanVaeNativeInt8V2Plugins"
NATIVE_INT8_V2_CUTLASS_COMMIT = "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
NATIVE_INT8_V2_LEVEL_OFFSETS = {"p1": 0, "p2": 6, "p3": 12}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rewrite_native_int8_v2_graph(
    *, source_path: str | Path, destination_path: str | Path, level: str
) -> dict[str, Any]:
    """Rebind a SHA-validated V1 plugin graph to one V2 optimization level."""

    if level not in NATIVE_INT8_V2_LEVELS:
        raise ValueError(f"unsupported Native INT8 V2 level: {level}")
    try:
        import onnx
        from onnx import helper
    except ImportError as exc:
        raise RuntimeError("Native INT8 V2 graph rewrite requires ONNX") from exc

    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    model = onnx.load(str(source), load_external_data=True)
    offset = NATIVE_INT8_V2_LEVEL_OFFSETS[level]
    rebound: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type != "SfWanNativeInt8ResidualBlockPlugin":
            continue
        attrs = {attribute.name: attribute for attribute in node.attribute}
        for key in ("tile1", "tile2"):
            attribute = attrs.get(key)
            if attribute is None:
                raise ValueError(f"V1 plugin node {node.name!r} has no {key}")
            base_tile = int(helper.get_attribute_value(attribute))
            if not 0 <= base_tile < 6:
                raise ValueError(
                    f"V1 plugin node {node.name!r} has invalid {key}={base_tile}"
                )
            attribute.i = base_tile + offset
        old_name = node.name
        node.op_type = NATIVE_INT8_V2_PLUGIN_NAME
        node.name = old_name.replace("native_int8/", f"native_int8_v2/{level}/", 1)
        rebound.append(
            {
                "v1_name": old_name,
                "v2_name": node.name,
                "tile1": int(helper.get_attribute_value(attrs["tile1"])),
                "tile2": int(helper.get_attribute_value(attrs["tile2"])),
                "profile_id": int(
                    helper.get_attribute_value(attrs["profile_id"])
                ),
            }
        )
    expected = EXPECTED_RESIDUAL_BLOCKS * 3
    if len(rebound) != expected:
        raise ValueError(
            f"Native INT8 V2 expected {expected} residual plugins, got {len(rebound)}"
        )
    onnx.checker.check_model(model, full_check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    onnx.save(model, str(temporary))
    temporary.replace(destination)
    return {
        "schema_version": NATIVE_INT8_V2_SCHEMA_VERSION,
        "variant": NATIVE_INT8_V2_VARIANT,
        "level": level,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "destination": str(destination),
        "destination_sha256": sha256_file(destination),
        "plugin_node_count": len(rebound),
        "replaced_target_conv_call_site_count": len(rebound) * 2,
        "plugins": rebound,
    }


def load_native_int8_v2_manifest(
    engine_dir: str | Path, *, level: str
) -> dict[str, Any]:
    if level not in NATIVE_INT8_V2_LEVELS:
        raise ValueError(f"unsupported Native INT8 V2 level: {level}")
    path = (
        Path(engine_dir).expanduser().resolve()
        / NATIVE_INT8_V2_SUBDIRECTORY
        / level
        / "manifest.json"
    )
    if not path.is_file():
        raise ValueError(f"Native INT8 V2 {level} manifest does not exist: {path}")
    return _load_json(path, label=f"Native INT8 V2 {level} manifest")


def validate_native_int8_v2_manifest(
    manifest: Mapping[str, Any],
    *,
    engine_dir: str | Path,
    base_manifest: Mapping[str, Any],
    level: str,
    verify_hashes: bool,
) -> dict[str, Any]:
    """Fail closed before a V2 plan or mixed cache bank reaches runtime."""

    if level not in NATIVE_INT8_V2_LEVELS:
        raise ValueError(f"unsupported Native INT8 V2 level: {level}")
    if (
        manifest.get("schema_version") != NATIVE_INT8_V2_SCHEMA_VERSION
        or manifest.get("variant") != NATIVE_INT8_V2_VARIANT
        or manifest.get("level") != level
    ):
        raise ValueError("Native INT8 V2 manifest identity changed")
    root = (
        Path(engine_dir).expanduser().resolve()
        / NATIVE_INT8_V2_SUBDIRECTORY
    )
    level_root = root / level
    v1_manifest = load_native_int8_manifest(engine_dir)
    v1_validated = validate_native_int8_manifest(
        v1_manifest,
        engine_dir=engine_dir,
        base_manifest=base_manifest,
        verify_hashes=verify_hashes,
    )
    if manifest.get("base_native_int8_v1_identity") != _identity(v1_manifest):
        raise ValueError("Native INT8 V2/V1 manifest identity mismatch")

    plugin = manifest.get("plugin")
    audit_record = manifest.get("audit")
    engines = manifest.get("engines")
    cache = manifest.get("cache")
    if not all(
        isinstance(value, Mapping)
        for value in (plugin, audit_record, engines, cache)
    ):
        raise ValueError("Native INT8 V2 manifest is incomplete")
    if (
        plugin.get("plugin_name") != NATIVE_INT8_V2_PLUGIN_NAME
        or plugin.get("plugin_namespace") != NATIVE_INT8_V2_PLUGIN_NAMESPACE
        or plugin.get("plugin_version") != NATIVE_INT8_V2_PLUGIN_VERSION
        or plugin.get("cutlass_commit") != NATIVE_INT8_V2_CUTLASS_COMMIT
    ):
        raise ValueError("Native INT8 V2 plugin ABI identity changed")
    plugin_path = root / str(plugin.get("file"))
    audit_path = level_root / str(audit_record.get("file"))
    if verify_hashes:
        if sha256_file(plugin_path) != plugin.get("sha256"):
            raise ValueError("Native INT8 V2 plugin SHA256 mismatch")
        if sha256_file(audit_path) != audit_record.get("sha256"):
            raise ValueError("Native INT8 V2 audit SHA256 mismatch")
    audit = _load_json(audit_path, label=f"Native INT8 V2 {level} audit")
    expected_workspace = {
        "p1": {"accumulator2_workspace_bytes": 0, "direct_causal_iterator": False},
        "p2": {
            "accumulator2_workspace_bytes": 0,
            "temporal_window_bytes": 0,
            "direct_causal_iterator": True,
        },
        "p3": {
            "accumulator1_workspace_bytes": 0,
            "accumulator2_workspace_bytes": 0,
            "temporal_window_bytes": 0,
            "direct_causal_iterator": True,
            "persistent_block_count": EXPECTED_RESIDUAL_BLOCKS * 3 * 2,
        },
    }[level]
    if (
        audit.get("passed") is not True
        or audit.get("errors") != []
        or audit.get("initial_call_site_count") != EXPECTED_CALL_SITES_PER_GRAPH
        or audit.get("steady_call_site_count") != EXPECTED_CALL_SITES_PER_GRAPH
        or audit.get("signature_count") != EXPECTED_SIGNATURES
    ):
        raise ValueError(f"Native INT8 V2 {level} audit did not pass")
    for key, expected in expected_workspace.items():
        if audit.get(key) != expected:
            raise ValueError(
                f"Native INT8 V2 {level} audit {key}={audit.get(key)!r}, "
                f"expected {expected!r}"
            )
    validated_engines: dict[str, Any] = {}
    for kind in ("initial", "steady"):
        record = engines.get(kind)
        if not isinstance(record, Mapping):
            raise ValueError(f"Native INT8 V2 {level} {kind} engine is missing")
        plan = level_root / str(record.get("file"))
        inspector = level_root / str(record.get("inspector_file"))
        if verify_hashes:
            if sha256_file(plan) != record.get("sha256"):
                raise ValueError(f"Native INT8 V2 {level} {kind} plan SHA mismatch")
            if sha256_file(inspector) != record.get("inspector_sha256"):
                raise ValueError(
                    f"Native INT8 V2 {level} {kind} inspector SHA mismatch"
                )
        if audit.get("plan_sha256", {}).get(kind) != record.get("sha256"):
            raise ValueError(f"Native INT8 V2 {level} {kind} audit/plan mismatch")
        validated_engines[kind] = {
            **dict(record),
            "path": str(plan),
            "inspector": json.loads(inspector.read_text(encoding="utf-8")),
        }
    bindings = cache.get("bindings")
    if bindings != v1_manifest.get("cache", {}).get("bindings"):
        raise ValueError("Native INT8 V2 changed the validated V1 mixed-cache ABI")
    if (
        cache.get("int8_slot_count") != EXPECTED_INT8_CACHE_SLOTS
        or cache.get("fp16_slot_count") != EXPECTED_FP16_CACHE_SLOTS
    ):
        raise ValueError("Native INT8 V2 cache dtype counts changed")
    return {
        "root": root,
        "level_root": level_root,
        "plugin_path": plugin_path,
        "engines": validated_engines,
        "audit": audit,
        "cache_bindings": [dict(value) for value in bindings],
        "profile_ids": dict(manifest.get("kernel_profile_ids", {})),
        "v1_validated": v1_validated,
    }


__all__ = [
    "NATIVE_INT8_V2_CUTLASS_COMMIT",
    "NATIVE_INT8_V2_LEVELS",
    "NATIVE_INT8_V2_LEVEL_OFFSETS",
    "NATIVE_INT8_V2_PLUGIN_INIT_SYMBOL",
    "NATIVE_INT8_V2_PLUGIN_LIBRARY_FILE",
    "NATIVE_INT8_V2_PLUGIN_NAME",
    "NATIVE_INT8_V2_PLUGIN_NAMESPACE",
    "NATIVE_INT8_V2_PLUGIN_VERSION",
    "NATIVE_INT8_V2_SCHEMA_VERSION",
    "NATIVE_INT8_V2_SUBDIRECTORY",
    "NATIVE_INT8_V2_VARIANT",
    "load_native_int8_v2_manifest",
    "rewrite_native_int8_v2_graph",
    "validate_native_int8_v2_manifest",
]
