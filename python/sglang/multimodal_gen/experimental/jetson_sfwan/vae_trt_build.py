"""Build fixed-shape FP16 and explicit-INT8 SFWan VAE TensorRT engines.

This command must run on the target Jetson AGX Orin.  TensorRT plans are tied
to the CUDA/TensorRT/SM environment that builds them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .vae_trt_qdq import (
    EXPECTED_CALL_SITES,
    EXPECTED_LOGICAL_CONVS,
    QDQ_OPSET,
    rewrite_onnx_with_int8_qdq,
)
from .vae_trt_runtime import (
    TRT_VAE_CACHE_BANK_BYTES,
    TRT_VAE_CACHE_COUNT,
    TRT_VAE_CACHE_TOTAL_ELEMENTS,
    TRT_VAE_HEIGHT,
    TRT_VAE_LATENT_SHAPE,
    TRT_VAE_MANIFEST_SCHEMA_VERSION,
    TRT_VAE_WIDTH,
)

EXPECTED_EXPORT_NEAREST_UPSAMPLES = 3


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_quantized_residual_conv_names(vae: Any) -> tuple[str, ...]:
    """Return exactly the 28 Wan residual-block conv1/conv2 module names."""

    from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanResidualBlock

    targets: list[str] = []
    for block_name, block in vae.decoder.named_modules():
        if not isinstance(block, WanResidualBlock):
            continue
        for attribute in ("conv1", "conv2"):
            convolution = getattr(block, attribute)
            if tuple(convolution.kernel_size) != (3, 3, 3):
                raise ValueError(
                    f"{block_name}.{attribute} is not a 3x3x3 causal Conv3d"
                )
            if tuple(convolution.stride) != (1, 1, 1):
                raise ValueError(
                    f"{block_name}.{attribute} must have unit stride for V1"
                )
            if int(convolution.groups) != 1:
                raise ValueError(f"{block_name}.{attribute} must not be grouped")
            targets.append(f"decoder.{block_name}.{attribute}")
    if len(targets) != EXPECTED_LOGICAL_CONVS:
        raise ValueError(
            f"SFWan TensorRT VAE expects {EXPECTED_LOGICAL_CONVS} residual "
            f"Conv3d modules, found {len(targets)}: {targets}"
        )
    return tuple(targets)


def _decoder_cache_slot_count(vae: Any) -> int:
    from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
        SpatialParallelCausalConv3d,
        WanCausalConv3d,
    )

    return sum(
        isinstance(module, (WanCausalConv3d, SpatialParallelCausalConv3d))
        for module in vae.decoder.modules()
    )


@contextmanager
def _portable_causal_pad_for_export() -> Any:
    """Force traceable Torch cat/pad instead of a CUDA or Triton custom op."""

    from sglang.multimodal_gen.runtime.layers import parallel_conv

    original = parallel_conv.fused_causal_conv3d_cat_pad
    parallel_conv.fused_causal_conv3d_cat_pad = None
    try:
        yield
    finally:
        parallel_conv.fused_causal_conv3d_cat_pad = original


def _identity_conv3d_input_format(x: Any, _weight: Any) -> Any:
    """Preserve logical values while omitting a PyTorch-only layout hint."""

    return x


@contextmanager
def _portable_conv3d_layout_for_export() -> Any:
    """Remove channels-last memory-format ops only while tracing ONNX.

    SGLang converts Wan VAE Conv3d weights to ``channels_last_3d`` for native
    PyTorch execution.  The causal Conv3d wrappers consequently insert an
    ``aten::contiguous(memory_format=channels_last_3d)`` operation, which the
    legacy ONNX exporter in NVIDIA's Jetson PyTorch build cannot represent.
    ONNX tensors do not carry PyTorch stride metadata, so bypassing this
    layout-only conversion preserves the graph's mathematical values.
    """

    from sglang.multimodal_gen.runtime.layers import parallel_conv
    from sglang.multimodal_gen.runtime.models.vaes import wanvae as wanvae_module

    original_wan_match = wanvae_module.match_conv3d_input_format
    original_parallel_match = getattr(
        parallel_conv,
        "_match_conv3d_input_format",
    )
    wanvae_module.match_conv3d_input_format = _identity_conv3d_input_format
    setattr(
        parallel_conv,
        "_match_conv3d_input_format",
        _identity_conv3d_input_format,
    )
    try:
        yield
    finally:
        wanvae_module.match_conv3d_input_format = original_wan_match
        setattr(
            parallel_conv,
            "_match_conv3d_input_format",
            original_parallel_match,
        )


def _assert_nearest_2x_export_equivalence() -> None:
    """Fail closed unless this Torch build gives identical 2x nearest results."""

    import torch

    sample = torch.arange(
        15,
        dtype=torch.float32,
    ).reshape(1, 1, 3, 5)
    nearest_exact = torch.nn.functional.interpolate(
        sample,
        scale_factor=(2.0, 2.0),
        mode="nearest-exact",
    )
    nearest = torch.nn.functional.interpolate(
        sample,
        scale_factor=(2.0, 2.0),
        mode="nearest",
    )
    if not torch.equal(nearest_exact, nearest):
        raise RuntimeError(
            "nearest-exact and nearest are not identical for fixed 2x "
            "upsampling in this Torch build"
        )


@contextmanager
def _portable_nearest_upsample_for_export(wrapper: Any) -> Any:
    """Use ONNX-exportable nearest only for fixed, equivalent 2x upsampling."""

    from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanUpsample

    upsamplers = [
        module
        for module in wrapper.modules()
        if isinstance(module, WanUpsample) and module.mode == "nearest-exact"
    ]
    if len(upsamplers) != EXPECTED_EXPORT_NEAREST_UPSAMPLES:
        raise ValueError(
            "SFWan TensorRT ONNX export expects "
            f"{EXPECTED_EXPORT_NEAREST_UPSAMPLES} nearest-exact upsamplers, "
            f"found {len(upsamplers)}"
        )

    for module in upsamplers:
        scale_factor = module.scale_factor
        if isinstance(scale_factor, (int, float)):
            normalized_scale = (float(scale_factor), float(scale_factor))
        else:
            normalized_scale = tuple(float(value) for value in scale_factor)
        if (
            module.size is not None
            or normalized_scale != (2.0, 2.0)
            or module.align_corners is not None
            or module.recompute_scale_factor not in (None, False)
        ):
            raise ValueError(
                "nearest-exact export substitution only supports size=None, "
                "scale_factor=(2, 2), align_corners=None, and "
                "recompute_scale_factor unset"
            )

    _assert_nearest_2x_export_equivalence()
    original_modes = [module.mode for module in upsamplers]
    try:
        for module in upsamplers:
            module.mode = "nearest"
        yield
    finally:
        for module, original_mode in zip(
            upsamplers,
            original_modes,
            strict=True,
        ):
            module.mode = original_mode


def _normalize_export_cache_tensors(
    *,
    cache: list[Any],
    active_cache_indices: tuple[int, ...],
    reference_tensor: Any,
) -> tuple[Any, ...]:
    """Make the chunk-boundary cache match the TensorRT FP16 binding ABI."""

    import torch

    if len(active_cache_indices) != TRT_VAE_CACHE_COUNT:
        raise ValueError(
            f"TensorRT export requires {TRT_VAE_CACHE_COUNT} active cache "
            f"indices, got {len(active_cache_indices)}"
        )
    if not isinstance(reference_tensor, torch.Tensor):
        raise TypeError("TensorRT export cache reference must be a tensor")

    supported_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }
    normalized: list[Any] = []
    for binding_index, source_index in enumerate(active_cache_indices):
        if source_index < 0 or source_index >= len(cache):
            raise ValueError(
                f"cache binding {binding_index} has invalid source slot {source_index}"
            )
        tensor = cache[source_index]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"cache binding {binding_index} from source slot {source_index} "
                f"is {type(tensor).__name__}, not a tensor"
            )
        if tensor.dtype not in supported_dtypes:
            raise TypeError(
                f"cache binding {binding_index} from source slot {source_index} "
                f"has unsupported dtype {tensor.dtype}"
            )
        if tensor.device != reference_tensor.device:
            raise ValueError(
                f"cache binding {binding_index} from source slot {source_index} "
                f"is on {tensor.device}, expected {reference_tensor.device}"
            )
        tensor = tensor.to(dtype=torch.float16).contiguous()
        cache[source_index] = tensor
        normalized.append(tensor)
    return tuple(normalized)


def _run_decoder_chunk(
    *,
    post_quant_conv: Any,
    decoder: Any,
    patch_size: Any,
    latent: Any,
    cache: list[Any],
    first_request_chunk: bool,
) -> Any:
    import torch

    from sglang.multimodal_gen.runtime.models.vaes import wanvae as wanvae_module

    x = post_quant_conv(latent)
    outputs = []
    with (
        wanvae_module.disable_spatial_parallel_decode(),
        wanvae_module.forward_context(
            feat_cache_arg=cache,
            feat_idx_arg=0,
        ),
    ):
        for latent_index in range(3):
            wanvae_module.feat_idx.set(0)
            wanvae_module.first_chunk.set(first_request_chunk and latent_index == 0)
            outputs.append(decoder(x[:, :, latent_index : latent_index + 1]))
    output = torch.cat(outputs, dim=2)
    if patch_size is not None:
        output = wanvae_module.unpatchify(output, patch_size=patch_size)
    return output


def _active_cache_indices(cache: list[Any]) -> tuple[int, ...]:
    import torch

    invalid = [
        (index, type(value).__name__)
        for index, value in enumerate(cache)
        if value is not None and not isinstance(value, torch.Tensor)
    ]
    if invalid:
        raise ValueError(f"causal cache contains non-tensor terminal states: {invalid}")
    indices = tuple(
        index for index, value in enumerate(cache) if isinstance(value, torch.Tensor)
    )
    if len(indices) != TRT_VAE_CACHE_COUNT:
        raise ValueError(
            f"SFWan TensorRT VAE expects {TRT_VAE_CACHE_COUNT} active cache slots, "
            f"found {indices}"
        )
    return indices


def _make_export_wrappers(
    *,
    vae: Any,
    cache_slot_count: int,
    active_cache_indices: tuple[int, ...],
) -> tuple[Any, Any]:
    import torch

    patch_size = vae.config.patch_size

    class InitialChunkWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.post_quant_conv = vae.post_quant_conv
            self.decoder = vae.decoder

        def forward(self, latent: Any) -> tuple[Any, ...]:
            cache: list[Any] = [None] * cache_slot_count
            rgb = _run_decoder_chunk(
                post_quant_conv=self.post_quant_conv,
                decoder=self.decoder,
                patch_size=patch_size,
                latent=latent,
                cache=cache,
                first_request_chunk=True,
            )
            cache_outputs = _normalize_export_cache_tensors(
                cache=cache,
                active_cache_indices=active_cache_indices,
                reference_tensor=latent,
            )
            return (rgb, *cache_outputs)

    class SteadyChunkWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.post_quant_conv = vae.post_quant_conv
            self.decoder = vae.decoder

        def forward(self, latent: Any, *cache_inputs: Any) -> tuple[Any, ...]:
            if len(cache_inputs) != len(active_cache_indices):
                raise ValueError("steady VAE wrapper received the wrong cache count")
            cache: list[Any] = [None] * cache_slot_count
            for cache_index, tensor in zip(
                active_cache_indices,
                cache_inputs,
                strict=True,
            ):
                cache[cache_index] = tensor
            rgb = _run_decoder_chunk(
                post_quant_conv=self.post_quant_conv,
                decoder=self.decoder,
                patch_size=patch_size,
                latent=latent,
                cache=cache,
                first_request_chunk=False,
            )
            cache_outputs = _normalize_export_cache_tensors(
                cache=cache,
                active_cache_indices=active_cache_indices,
                reference_tensor=latent,
            )
            return (rgb, *cache_outputs)

    return InitialChunkWrapper().eval(), SteadyChunkWrapper().eval()


def _dummy_denormalized_chunks(
    *,
    vae: Any,
    device: Any,
    seed: int,
) -> list[Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    normalized_fp32 = torch.randn(
        (1, 16, 21, 60, 104),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    normalized_bf16 = normalized_fp32.to(torch.bfloat16)
    mean = torch.tensor(
        vae.config.latents_mean,
        device=device,
        dtype=torch.float32,
    ).view(1, 16, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std,
        device=device,
        dtype=torch.float32,
    ).view(1, 16, 1, 1, 1)
    chunks = []
    for index in range(7):
        normalized = normalized_bf16[:, :, index * 3 : (index + 1) * 3].to(
            device=device
        )
        chunks.append((normalized.float() * std + mean).half().contiguous())
    return chunks


def _collect_activation_scales(
    *,
    vae: Any,
    target_module_names: tuple[str, ...],
    latent_chunks: list[Any],
    cache_slot_count: int,
) -> dict[str, dict[str, float]]:
    import torch

    module_map = dict(vae.named_modules())
    absmax: dict[str, dict[str, Any | None]] = {
        "initial": {name: None for name in target_module_names},
        "steady": {name: None for name in target_module_names},
    }
    current_graph = ["initial"]
    handles = []

    def _make_hook(module_name: str) -> Any:
        def _hook(_module: Any, args: tuple[Any, ...]) -> None:
            value = args[0].detach().abs().amax().float()
            graph_values = absmax[current_graph[0]]
            previous = graph_values[module_name]
            graph_values[module_name] = (
                value if previous is None else torch.maximum(previous, value)
            )

        return _hook

    for module_name in target_module_names:
        module = module_map.get(module_name)
        if module is None:
            raise ValueError(f"VAE module {module_name!r} does not exist")
        handles.append(module.register_forward_pre_hook(_make_hook(module_name)))

    cache: list[Any] = [None] * cache_slot_count
    try:
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ),
            _portable_causal_pad_for_export(),
        ):
            for chunk_index, latent in enumerate(latent_chunks):
                current_graph[0] = "initial" if chunk_index == 0 else "steady"
                output = _run_decoder_chunk(
                    post_quant_conv=vae.post_quant_conv,
                    decoder=vae.decoder,
                    patch_size=vae.config.patch_size,
                    latent=latent,
                    cache=cache,
                    first_request_chunk=chunk_index == 0,
                )
                del output
    finally:
        for handle in handles:
            handle.remove()

    scales: dict[str, dict[str, float]] = {}
    for graph_kind, values in absmax.items():
        collected = {
            name: (None if value is None else float(value.item()))
            for name, value in values.items()
        }
        missing = [
            name for name, value in collected.items() if value is None or value <= 0
        ]
        if missing:
            raise ValueError(
                f"{graph_kind} activation collection produced no samples for {missing}"
            )
        scales[graph_kind] = {
            name: max(float(value) / 127.0, 1.0e-8)
            for name, value in collected.items()
            if value is not None
        }
    return scales


def _validate_model_contract(vae: Any) -> None:
    if not bool(vae.use_feature_cache):
        raise ValueError("TensorRT SFWan VAE requires feature cache")
    if int(vae.config.z_dim) != 16:
        raise ValueError("TensorRT SFWan VAE requires 16 latent channels")
    if int(vae.config.out_channels) != 3:
        raise ValueError("TensorRT SFWan VAE requires three RGB output channels")
    spatial_factor = 2 ** (len(vae.config.dim_mult) - 1)
    temporal_factor = 2 ** sum(bool(value) for value in vae.config.temperal_downsample)
    if spatial_factor != 8 or temporal_factor != 4:
        raise ValueError("TensorRT SFWan VAE requires spatial=8 and temporal=4")
    if len(vae.config.latents_mean) != 16 or len(vae.config.latents_std) != 16:
        raise ValueError("TensorRT SFWan VAE requires 16-channel mean/std")


def _validate_onnx_fp16_io_contract(
    *,
    model: Any,
    path: Path,
    expected_input_shapes: dict[str, tuple[int, ...]],
    expected_output_shapes: dict[str, tuple[int, ...]],
) -> None:
    import onnx

    def _validate_values(
        *,
        values: Any,
        expected_shapes: dict[str, tuple[int, ...]],
        kind: str,
    ) -> None:
        actual_names = [value.name for value in values]
        expected_names = list(expected_shapes)
        if actual_names != expected_names:
            raise ValueError(
                f"{path.name} ONNX {kind} names are {actual_names}, "
                f"expected {expected_names}"
            )
        for value in values:
            tensor_type = value.type.tensor_type
            if int(tensor_type.elem_type) != int(onnx.TensorProto.FLOAT16):
                raise ValueError(
                    f"{path.name} ONNX {kind} {value.name!r} must be float16, "
                    f"got elem_type={tensor_type.elem_type}"
                )
            dimensions = tensor_type.shape.dim
            if any(
                bool(dimension.dim_param) or int(dimension.dim_value) <= 0
                for dimension in dimensions
            ):
                raise ValueError(
                    f"{path.name} ONNX {kind} {value.name!r} must have a "
                    "fully static positive shape"
                )
            actual_shape = tuple(int(dimension.dim_value) for dimension in dimensions)
            expected_shape = expected_shapes[value.name]
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{path.name} ONNX {kind} {value.name!r} shape is "
                    f"{actual_shape}, expected {expected_shape}"
                )

    _validate_values(
        values=model.graph.input,
        expected_shapes=expected_input_shapes,
        kind="input",
    )
    _validate_values(
        values=model.graph.output,
        expected_shapes=expected_output_shapes,
        kind="output",
    )


def _export_onnx(
    *,
    wrapper: Any,
    arguments: tuple[Any, ...],
    input_names: list[str],
    output_names: list[str],
    output_shapes: list[tuple[int, ...]],
    path: Path,
) -> None:
    import onnx
    import torch

    if len(arguments) != len(input_names):
        raise ValueError("ONNX export argument and input-name counts differ")
    if len(output_shapes) != len(output_names):
        raise ValueError("ONNX export output-shape and output-name counts differ")
    if not arguments:
        raise ValueError("ONNX export requires at least one tensor input")
    expected_device = arguments[0].device
    for name, argument in zip(input_names, arguments, strict=True):
        if not isinstance(argument, torch.Tensor):
            raise TypeError(f"ONNX export input {name!r} is not a tensor")
        if argument.dtype != torch.float16:
            raise TypeError(
                f"ONNX export input {name!r} must be float16, got {argument.dtype}"
            )
        if argument.device != expected_device:
            raise ValueError(
                f"ONNX export input {name!r} is on {argument.device}, "
                f"expected {expected_device}"
            )
        if not argument.is_contiguous():
            raise ValueError(f"ONNX export input {name!r} must be contiguous")

    with (
        torch.inference_mode(),
        _portable_causal_pad_for_export(),
        _portable_conv3d_layout_for_export(),
        _portable_nearest_upsample_for_export(wrapper),
    ):
        torch.onnx.export(
            wrapper,
            arguments,
            str(path),
            export_params=True,
            opset_version=QDQ_OPSET,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            keep_initializers_as_inputs=False,
            dynamo=False,
        )
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    _validate_onnx_fp16_io_contract(
        model=model,
        path=path,
        expected_input_shapes={
            name: tuple(int(dimension) for dimension in argument.shape)
            for name, argument in zip(input_names, arguments, strict=True)
        },
        expected_output_shapes=dict(zip(output_names, output_shapes, strict=True)),
    )


def _trt_logger(trt: Any) -> Any:
    return trt.Logger(trt.Logger.INFO)


def _network_flags(trt: Any) -> int:
    flags = 0
    for flag_name in ("EXPLICIT_BATCH", "STRONGLY_TYPED"):
        flag = getattr(trt.NetworkDefinitionCreationFlag, flag_name, None)
        if flag is not None:
            flags |= 1 << int(flag)
    return flags


def _profiling_verbosity(trt: Any, value: str) -> Any:
    if value == "none":
        return trt.ProfilingVerbosity.NONE
    if value == "detailed":
        return trt.ProfilingVerbosity.DETAILED
    raise ValueError(f"unsupported TensorRT profiling verbosity: {value}")


def _engine_io_contract(engine: Any) -> list[dict[str, Any]]:
    records = []
    for index in range(int(engine.num_io_tensors)):
        name = engine.get_tensor_name(index)
        records.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)).split(".")[-1].lower(),
                "dtype": str(engine.get_tensor_dtype(name)).split(".")[-1].lower(),
                "shape": [int(value) for value in engine.get_tensor_shape(name)],
            }
        )
    return records


def _build_engine_bytes(
    *,
    trt: Any,
    onnx_path: Path,
    workspace_gib: float,
    profiling_verbosity: str,
) -> tuple[bytes, list[dict[str, Any]], str]:
    logger = _trt_logger(trt)
    builder = trt.Builder(logger)
    network = builder.create_network(_network_flags(trt))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [
            str(parser.get_error(index)) for index in range(int(parser.num_errors))
        ]
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gib * 1024**3),
    )
    config.profiling_verbosity = _profiling_verbosity(trt, profiling_verbosity)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT could not build {onnx_path}")
    plan = bytes(serialized)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError(
            f"TensorRT could not deserialize the new plan for {onnx_path}"
        )
    inspector = engine.create_engine_inspector()
    inspector_json = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    return plan, _engine_io_contract(engine), inspector_json


def _inspector_layers(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [record for record in value if isinstance(record, dict)]
    if not isinstance(value, dict):
        return []
    for key in ("Layers", "layers"):
        records = value.get(key)
        if isinstance(records, list):
            return [record for record in records if isinstance(record, dict)]
    return [value]


def _inspector_layer_name(record: dict[str, Any]) -> str:
    for key in ("Name", "name", "LayerName", "layer_name"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _precision_evidence(record: dict[str, Any]) -> str:
    evidence: list[str] = []

    def _visit(value: Any, key: str = "") -> None:
        normalized_key = key.lower().replace("_", "")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                _visit(child_value, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                _visit(child, key)
            return
        if any(
            marker in normalized_key
            for marker in ("format", "datatype", "precision", "tactic")
        ):
            evidence.append(str(value))

    _visit(record)
    return " ".join(evidence).lower()


def _audit_tensorrt_tactics(
    *,
    graph_kind: str,
    call_site_names: list[str],
    inspector_json: str,
) -> dict[str, Any]:
    try:
        decoded = json.loads(inspector_json)
    except json.JSONDecodeError:
        decoded = {}
    layers = _inspector_layers(decoded)
    matches: dict[str, Any] = {}
    unmapped: list[str] = []
    non_int8: list[str] = []
    for call_site in call_site_names:
        matched = [
            layer for layer in layers if call_site in _inspector_layer_name(layer)
        ]
        if not matched:
            unmapped.append(call_site)
            continue
        evidence = [_precision_evidence(layer) for layer in matched]
        has_int8 = any(
            "int8" in value or "kint8" in value or "imma" in value for value in evidence
        )
        if not has_int8:
            non_int8.append(call_site)
        matches[call_site] = {
            "layer_names": [_inspector_layer_name(layer) for layer in matched],
            "int8_evidence": has_int8,
        }
    errors = []
    if len(call_site_names) != EXPECTED_CALL_SITES:
        errors.append(
            f"Q/DQ report contains {len(call_site_names)} call sites, "
            f"expected {EXPECTED_CALL_SITES}"
        )
    if unmapped:
        errors.append(f"{len(unmapped)} call sites could not be mapped")
    if non_int8:
        errors.append(f"{len(non_int8)} call sites have no INT8 tactic evidence")
    return {
        "graph_kind": graph_kind,
        "passed": not errors,
        "errors": errors,
        "mapped_count": len(matches),
        "unmapped_call_sites": unmapped,
        "non_int8_call_sites": non_int8,
        "matches": matches,
        "inspector": decoded,
    }


def _engine_record(
    *,
    output_dir: Path,
    plan_path: Path,
    io_contract: list[dict[str, Any]],
    rgb_frames: int,
) -> dict[str, Any]:
    return {
        "file": str(plan_path.relative_to(output_dir)),
        "sha256": _sha256_file(plan_path),
        "rgb_shape": [1, 3, rgb_frames, TRT_VAE_HEIGHT, TRT_VAE_WIDTH],
        "io_tensors": io_contract,
    }


def build_engines(
    *,
    model_path: str,
    output_dir: str | Path,
    height: int,
    width: int,
    seed: int,
    workspace_gib: float,
    profiling_verbosity: str,
    device_index: int,
) -> dict[str, Any]:
    if (height, width) != (TRT_VAE_HEIGHT, TRT_VAE_WIDTH):
        raise ValueError("TensorRT SFWan VAE V1 supports only 480x832")
    if workspace_gib <= 0:
        raise ValueError("--workspace-gib must be positive")

    import onnx
    import tensorrt as trt
    import torch

    from .model import ModelLoadConfig, VAE_COMPONENT_NAMES, _ComponentSet

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    load_config = ModelLoadConfig(
        model_path=model_path,
        device_index=device_index,
        vae_precision="fp16",
        text_encoder_cpu_offload=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        enable_profile=False,
        enable_nvtx=False,
    )
    components = _ComponentSet(
        load_config=load_config,
        component_names=VAE_COMPONENT_NAMES,
    )
    vae = (
        components.modules["vae"]
        .eval()
        .to(
            device=components.device,
            dtype=torch.float16,
        )
    )
    _validate_model_contract(vae)
    targets = find_quantized_residual_conv_names(vae)
    cache_slot_count = _decoder_cache_slot_count(vae)
    if cache_slot_count != 33:
        raise ValueError(
            f"SFWan TensorRT VAE expects 33 allocated decoder slots, "
            f"found {cache_slot_count}"
        )

    latent_chunks = _dummy_denormalized_chunks(
        vae=vae,
        device=components.device,
        seed=seed,
    )
    activation_scales = _collect_activation_scales(
        vae=vae,
        target_module_names=targets,
        latent_chunks=latent_chunks,
        cache_slot_count=cache_slot_count,
    )

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ),
        _portable_causal_pad_for_export(),
    ):
        initial_cache: list[Any] = [None] * cache_slot_count
        initial_rgb = _run_decoder_chunk(
            post_quant_conv=vae.post_quant_conv,
            decoder=vae.decoder,
            patch_size=vae.config.patch_size,
            latent=latent_chunks[0],
            cache=initial_cache,
            first_request_chunk=True,
        )
        active_indices = _active_cache_indices(initial_cache)
        initial_cache_tensors = _normalize_export_cache_tensors(
            cache=initial_cache,
            active_cache_indices=active_indices,
            reference_tensor=latent_chunks[0],
        )
        cache_shapes = [tuple(tensor.shape) for tensor in initial_cache_tensors]
        cache_elements = sum(math.prod(shape) for shape in cache_shapes)
        if cache_elements != TRT_VAE_CACHE_TOTAL_ELEMENTS:
            raise ValueError(
                f"SFWan steady cache has {cache_elements} elements, "
                f"expected {TRT_VAE_CACHE_TOTAL_ELEMENTS}"
            )
        if tuple(initial_rgb.shape) != (1, 3, 9, height, width):
            raise ValueError(f"initial VAE output shape is {tuple(initial_rgb.shape)}")

        steady_cache: list[Any] = [None] * cache_slot_count
        for index, tensor in zip(active_indices, initial_cache_tensors, strict=True):
            steady_cache[index] = tensor
        steady_rgb = _run_decoder_chunk(
            post_quant_conv=vae.post_quant_conv,
            decoder=vae.decoder,
            patch_size=vae.config.patch_size,
            latent=latent_chunks[1],
            cache=steady_cache,
            first_request_chunk=False,
        )
        steady_indices = _active_cache_indices(steady_cache)
        steady_cache_tensors = _normalize_export_cache_tensors(
            cache=steady_cache,
            active_cache_indices=steady_indices,
            reference_tensor=latent_chunks[1],
        )
        steady_shapes = [tuple(tensor.shape) for tensor in steady_cache_tensors]
        if steady_indices != active_indices or steady_shapes != cache_shapes:
            raise ValueError("initial and steady VAE cache contracts do not match")
        if tuple(steady_rgb.shape) != (1, 3, 12, height, width):
            raise ValueError(f"steady VAE output shape is {tuple(steady_rgb.shape)}")

    initial_wrapper, steady_wrapper = _make_export_wrappers(
        vae=vae,
        cache_slot_count=cache_slot_count,
        active_cache_indices=active_indices,
    )
    cache_input_names = [f"cache_in_{index:03d}" for index in range(32)]
    cache_output_names = [f"cache_out_{index:03d}" for index in range(32)]
    initial_fp16_path = root / "initial_fp16.onnx"
    steady_fp16_path = root / "steady_fp16.onnx"
    _export_onnx(
        wrapper=initial_wrapper,
        arguments=(latent_chunks[0],),
        input_names=["latent"],
        output_names=["rgb", *cache_output_names],
        output_shapes=[(1, 3, 9, height, width), *cache_shapes],
        path=initial_fp16_path,
    )
    _export_onnx(
        wrapper=steady_wrapper,
        arguments=(latent_chunks[1], *initial_cache_tensors),
        input_names=["latent", *cache_input_names],
        output_names=["rgb", *cache_output_names],
        output_shapes=[(1, 3, 12, height, width), *cache_shapes],
        path=steady_fp16_path,
    )

    initial_int8_path = root / "initial_int8_qdq.onnx"
    steady_int8_path = root / "steady_int8_qdq.onnx"
    qdq_reports = {
        "initial": rewrite_onnx_with_int8_qdq(
            source_path=initial_fp16_path,
            destination_path=initial_int8_path,
            graph_kind="initial",
            target_module_names=targets,
            activation_scales=activation_scales["initial"],
        ),
        "steady": rewrite_onnx_with_int8_qdq(
            source_path=steady_fp16_path,
            destination_path=steady_int8_path,
            graph_kind="steady",
            target_module_names=targets,
            activation_scales=activation_scales["steady"],
        ),
    }
    _write_json(
        root / "quant_scales.json",
        {
            "seed": seed,
            "activation": activation_scales,
            "weight": {
                kind: report["weight_scales"] for kind, report in qdq_reports.items()
            },
        },
    )

    onnx_paths = {
        "fp16": {"initial": initial_fp16_path, "steady": steady_fp16_path},
        "int8": {"initial": initial_int8_path, "steady": steady_int8_path},
    }
    engines: dict[str, dict[str, Any]] = {"fp16": {}, "int8": {}}
    tactic_audits: dict[str, Any] = {}
    for precision in ("fp16", "int8"):
        for kind in ("initial", "steady"):
            plan, io_contract, inspector_json = _build_engine_bytes(
                trt=trt,
                onnx_path=onnx_paths[precision][kind],
                workspace_gib=workspace_gib,
                profiling_verbosity=profiling_verbosity,
            )
            plan_path = root / f"{kind}_{precision}.plan"
            plan_path.write_bytes(plan)
            engines[precision][kind] = _engine_record(
                output_dir=root,
                plan_path=plan_path,
                io_contract=io_contract,
                rgb_frames=9 if kind == "initial" else 12,
            )
            if precision != "int8":
                continue
            detailed_inspector = inspector_json
            if profiling_verbosity != "detailed":
                _audit_plan, _audit_io, detailed_inspector = _build_engine_bytes(
                    trt=trt,
                    onnx_path=onnx_paths[precision][kind],
                    workspace_gib=workspace_gib,
                    profiling_verbosity="detailed",
                )
                del _audit_plan, _audit_io
            tactic_audits[kind] = _audit_tensorrt_tactics(
                graph_kind=kind,
                call_site_names=qdq_reports[kind]["call_site_names"],
                inspector_json=detailed_inspector,
            )

    audit_errors = [
        error
        for kind in ("initial", "steady")
        for error in tactic_audits[kind]["errors"]
    ]
    int8_audit = {
        "passed": not audit_errors,
        "errors": audit_errors,
        "target_conv_call_sites_per_graph": EXPECTED_CALL_SITES,
        "structural": {
            kind: {
                key: value
                for key, value in report.items()
                if key
                not in {
                    "weight_scales",
                    "activation_scales",
                }
            }
            for kind, report in qdq_reports.items()
        },
        "tactics": tactic_audits,
    }
    _write_json(root / "int8_audit.json", int8_audit)
    if not int8_audit["passed"]:
        raise RuntimeError(
            "TensorRT built the graph but the INT8 tactic audit failed; "
            f"see {root / 'int8_audit.json'}"
        )

    manifest = {
        "schema_version": TRT_VAE_MANIFEST_SCHEMA_VERSION,
        "model_id": model_path,
        "resolved_model_path": str(components.model_path),
        "batch_size": 1,
        "height": height,
        "width": width,
        "latent_shape": list(TRT_VAE_LATENT_SHAPE),
        "latent_dtype": "float16",
        "latents_mean": [float(value) for value in vae.config.latents_mean],
        "latents_std": [float(value) for value in vae.config.latents_std],
        "cache": {
            "allocated_slot_count": cache_slot_count,
            "active_slot_indices": list(active_indices),
            "bindings": [
                {
                    "index": index,
                    "source_slot": active_indices[index],
                    "input_name": cache_input_names[index],
                    "output_name": cache_output_names[index],
                    "shape": list(cache_shapes[index]),
                    "dtype": "float16",
                }
                for index in range(TRT_VAE_CACHE_COUNT)
            ],
            "total_elements": cache_elements,
            "single_bank_bytes": TRT_VAE_CACHE_BANK_BYTES,
            "double_bank_bytes": TRT_VAE_CACHE_BANK_BYTES * 2,
        },
        "engines": engines,
        "quantization": {
            "scheme": "explicit_qdq_signed_int8",
            "target_module_names": list(targets),
            "logical_conv_count": EXPECTED_LOGICAL_CONVS,
            "call_sites_per_graph": EXPECTED_CALL_SITES,
            "activation": "per_tensor_symmetric_static",
            "weight": "per_output_channel_symmetric_axis_0",
            "calibration": {
                "kind": "deterministic_dummy_speed_only",
                "seed": seed,
                "num_video_frames": 81,
                "num_latent_chunks": 7,
            },
        },
        "build": {
            "compute_capability": list(
                torch.cuda.get_device_capability(components.device)
            ),
            "device_name": torch.cuda.get_device_name(components.device),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "tensorrt_version": str(trt.__version__),
            "onnx_version": str(onnx.__version__),
            "workspace_gib": workspace_gib,
            "profiling_verbosity": profiling_verbosity,
            "source_onnx_sha256": {
                kind: {
                    precision: _sha256_file(onnx_paths[precision][kind])
                    for precision in ("fp16", "int8")
                }
                for kind in ("initial", "steady")
            },
        },
        "int8_audit": {
            "passed": True,
            "target_conv_call_sites_per_graph": EXPECTED_CALL_SITES,
            "report_file": "int8_audit.json",
            "report_sha256": _sha256_file(root / "int8_audit.json"),
        },
    }
    if manifest["build"]["compute_capability"] != [8, 7]:
        raise ValueError("TensorRT SFWan VAE plans must be built on Jetson Orin SM87")
    _write_json(root / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, default=TRT_VAE_HEIGHT)
    parser.add_argument("--width", type=int, default=TRT_VAE_WIDTH)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument(
        "--profiling-verbosity",
        choices=("none", "detailed"),
        default="none",
    )
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_engines(
        model_path=args.model_path,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        seed=args.seed,
        workspace_gib=args.workspace_gib,
        profiling_verbosity=args.profiling_verbosity,
        device_index=args.device_index,
    )
    print(
        json.dumps(
            {
                "engine_dir": str(Path(args.output_dir).expanduser().resolve()),
                "manifest_schema": manifest["schema_version"],
                "int8_audit_passed": manifest["int8_audit"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
