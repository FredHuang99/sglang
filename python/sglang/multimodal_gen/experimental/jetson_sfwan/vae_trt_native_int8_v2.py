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
NATIVE_INT8_V2_SCHEMA_VERSION = 3
NATIVE_INT8_V2_P1_SCHEMA_VERSION = 2
NATIVE_INT8_V2_SUBDIRECTORY = "native_int8_v2"
NATIVE_INT8_V2_WINDOWED_SUBDIRECTORY = "windowed_v3"
NATIVE_INT8_V2_LEVELS = ("p1", "p2", "p3")
NATIVE_INT8_V2_PLUGIN_LIBRARY_FILE = "libsfwan_vae_native_int8_v2.so"
NATIVE_INT8_V2_PLUGIN_NAME = "SfWanNativeInt8V2ResidualBlockPlugin"
NATIVE_INT8_V2_PLUGIN_NAMESPACE = "sglang.sfwan"
NATIVE_INT8_V2_PLUGIN_VERSION = "1"
NATIVE_INT8_V2_PLUGIN_INIT_SYMBOL = "initSfWanVaeNativeInt8V2Plugins"
NATIVE_INT8_V2_CUTLASS_COMMIT = "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
NATIVE_INT8_V2_LEVEL_OFFSETS = {"p1": 0, "p2": 6, "p3": 12}
NATIVE_INT8_V2_P1_ALGORITHM = "cutlass_implicit_gemm_fused_epilogue"
NATIVE_INT8_V2_P1_KERNEL_REVISION = "cutlass_fused_epilogue_v2"
NATIVE_INT8_V2_P2_ALGORITHM = "cutlass_windowed_register_producer"
NATIVE_INT8_V2_P2_KERNEL_REVISION = "windowed_register_producer_v1"
NATIVE_INT8_V2_P3_ALGORITHM = "cutlass_windowed_conv1_fp16_epilogue"
NATIVE_INT8_V2_P3_KERNEL_REVISION = "windowed_conv1_fp16_epilogue_v1"
NATIVE_INT8_V2_SERIALIZATION_ABI_REVISION = 1


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


def native_int8_v2_level_root(engine_dir: str | Path, *, level: str) -> Path:
    if level not in NATIVE_INT8_V2_LEVELS:
        raise ValueError(f"unsupported Native INT8 V2 level: {level}")
    root = (
        Path(engine_dir).expanduser().resolve() / NATIVE_INT8_V2_SUBDIRECTORY
    )
    if level == "p1":
        return root / level
    return root / NATIVE_INT8_V2_WINDOWED_SUBDIRECTORY / level


def _manifest_schema(level: str) -> int:
    return (
        NATIVE_INT8_V2_P1_SCHEMA_VERSION
        if level == "p1"
        else NATIVE_INT8_V2_SCHEMA_VERSION
    )


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
        "schema_version": _manifest_schema(level),
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
    path = native_int8_v2_level_root(engine_dir, level=level) / "manifest.json"
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
        manifest.get("schema_version") != _manifest_schema(level)
        or manifest.get("variant") != NATIVE_INT8_V2_VARIANT
        or manifest.get("level") != level
    ):
        raise ValueError("Native INT8 V2 manifest identity changed")
    root = (
        Path(engine_dir).expanduser().resolve()
        / NATIVE_INT8_V2_SUBDIRECTORY
    )
    level_root = native_int8_v2_level_root(engine_dir, level=level)
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
            "direct_causal_iterator": False,
        },
        "p3": {
            "accumulator1_workspace_bytes": 0,
            "accumulator2_workspace_bytes": 0,
            "direct_causal_iterator": False,
            "persistent_block_count": 0,
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
    if level == "p2" and int(audit.get("accumulator1_workspace_bytes", 0)) <= 0:
        raise ValueError("Native INT8 V2 P2 lost its Conv1 accumulator workspace")
    if level == "p3" and int(audit.get("conv1_mid_workspace_bytes", 0)) <= 0:
        raise ValueError("Native INT8 V2 P3 has no FP16 Conv1-mid workspace")
    if level in {"p2", "p3"} and int(audit.get("temporal_window_bytes", 0)) <= 0:
        raise ValueError(f"Native INT8 V2 {level} has no temporal-window workspace")
    if level == "p1":
        contract = audit.get("p1_kernel_contract")
        expected_contract = {
            "tensor_core_int8": True,
            "accumulator2_global_store": False,
            "separate_residual_kernel": False,
            "direct_conv_kernel_used_by_p1": False,
            "serialization_abi_revision": (
                NATIVE_INT8_V2_SERIALIZATION_ABI_REVISION
            ),
        }
        if (
            audit.get("p1_algorithm") != NATIVE_INT8_V2_P1_ALGORITHM
            or audit.get("p1_kernel_revision")
            != NATIVE_INT8_V2_P1_KERNEL_REVISION
            or contract != expected_contract
            or manifest.get("p1_kernel_revision")
            != NATIVE_INT8_V2_P1_KERNEL_REVISION
        ):
            raise ValueError("Native INT8 V2 P1 CUTLASS kernel contract changed")
    else:
        expected_algorithms = {
            "p2": (
                NATIVE_INT8_V2_P2_ALGORITHM,
                NATIVE_INT8_V2_P2_KERNEL_REVISION,
                {
                    "tensor_core_int8": True,
                    "direct_causal_iterator": False,
                    "legacy_direct_causal_used": False,
                    "temporal_window_materialized": True,
                    "entry_value_global_loads": 1,
                    "mid_accumulator_global_loads": 1,
                    "accumulator1_global_store": True,
                    "accumulator2_global_store": False,
                    "conv2_fused_residual_epilogue": True,
                },
            ),
            "p3": (
                NATIVE_INT8_V2_P3_ALGORITHM,
                NATIVE_INT8_V2_P3_KERNEL_REVISION,
                {
                    "tensor_core_int8": True,
                    "direct_causal_iterator": False,
                    "legacy_persistent_wmma": False,
                    "temporal_window_materialized": True,
                    "entry_value_global_loads": 1,
                    "mid_value_global_loads": 1,
                    "accumulator1_global_store": False,
                    "accumulator2_global_store": False,
                    "conv1_output_dtype_fp16": True,
                    "conv1_fused_dequant_bias": True,
                    "conv2_fused_residual_epilogue": True,
                    "persistent_block_count": 0,
                },
            ),
        }
        algorithm, revision, contract = expected_algorithms[level]
        if (
            audit.get("native_int8_v2_algorithm") != algorithm
            or audit.get("native_int8_v2_kernel_revision") != revision
            or audit.get("native_int8_v2_legacy_wmma_used") is not False
            or audit.get("native_int8_v2_kernel_contract") != contract
            or manifest.get("native_int8_v2_algorithm") != algorithm
            or manifest.get("native_int8_v2_kernel_revision") != revision
            or manifest.get("native_int8_v2_legacy_wmma_used") is not False
        ):
            raise ValueError(
                f"Native INT8 V2 {level} CUTLASS kernel contract changed"
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
    "NATIVE_INT8_V2_P1_ALGORITHM",
    "NATIVE_INT8_V2_P1_KERNEL_REVISION",
    "NATIVE_INT8_V2_P1_SCHEMA_VERSION",
    "NATIVE_INT8_V2_SCHEMA_VERSION",
    "NATIVE_INT8_V2_SERIALIZATION_ABI_REVISION",
    "NATIVE_INT8_V2_SUBDIRECTORY",
    "NATIVE_INT8_V2_WINDOWED_SUBDIRECTORY",
    "NATIVE_INT8_V2_VARIANT",
    "load_native_int8_v2_manifest",
    "native_int8_v2_level_root",
    "rewrite_native_int8_v2_graph",
    "validate_native_int8_v2_manifest",
]
