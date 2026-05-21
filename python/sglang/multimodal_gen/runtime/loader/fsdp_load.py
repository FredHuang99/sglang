# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune
# Copyright 2024 The TorchTune Authors.
# Copyright 2025 The sglang-diffusion Authors.

import time
from collections.abc import Callable, Generator
from contextlib import nullcontext
from itertools import chain
from typing import Any

import torch
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed._tensor import distribute_tensor
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.nn.modules.module import _IncompatibleKeys

from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    resolve_safetensors_read_backend,
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.loader.weight_broadcast import (
    broadcast_module_tensors,
    broadcast_rank0_load_status,
    confirm_rank0_broadcast_entry,
    confirm_tensor_broadcast_ready,
    log_broadcast_stage,
    materialize_empty_model_state_dict,
    offload_model_tensors_to_cpu,
    resolve_rank0_broadcast_decision,
    set_profile_load_mode,
)
from sglang.multimodal_gen.runtime.loader.weight_staging import (
    maybe_stage_weight_iterator,
    normalize_weight_staging_mode,
)
from sglang.multimodal_gen.runtime.loader.weight_warm_pool import (
    build_weight_warm_pool_key,
    get_weight_warm_pool_entry,
    is_weight_warm_pool_enabled,
    make_weight_warm_pool_entry,
    normalize_weight_warm_pool_mode,
    put_weight_warm_pool_entry,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.weight_load_profiler import (
    WEIGHT_LOAD_CPU_MATERIALIZE_MS,
    WEIGHT_LOAD_D2H_OR_OFFLOAD_MS,
    WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS,
    WEIGHT_LOAD_PIN_MEMORY_MS,
    WEIGHT_LOAD_READ_SAFETENSORS_MS,
    DiffusionWeightLoadProfiler,
)
from sglang.multimodal_gen.utils import set_mixed_precision_policy
from sglang.srt.utils import is_npu

_is_npu = is_npu()

logger = init_logger(__name__)


def _make_param_like(
    actual_param: torch.nn.Parameter, tensor: torch.Tensor
) -> torch.nn.Parameter:
    cls = actual_param.__class__
    # nn.Parameter defaults to requires_grad=True, which is illegal for non-floating/complex dtypes (e.g., int8/FP8
    # quantized weights).
    try:
        new_param = cls.__new__(cls, tensor, requires_grad=False)
    except TypeError:
        new_param = cls.__new__(cls, tensor)
    new_param.__dict__.update(actual_param.__dict__)
    new_param.requires_grad = False
    return new_param


# TODO(PY): add compile option
def maybe_load_fsdp_model(
    model_cls: type[nn.Module],
    init_params: dict[str, Any],
    weight_dir_list: list[str],
    device: torch.device,
    hsdp_replicate_dim: int,
    hsdp_shard_dim: int,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    fsdp_inference: bool = False,
    output_dtype: torch.dtype | None = None,
    pin_cpu_memory: bool = True,
    strict: bool = True,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
    weight_staging_mode: str = "none",
    weight_load_mode: str = "default",
    weight_broadcast_components: list[str] | tuple[str, ...] | str | None = None,
    weight_component: str | None = None,
    weight_tp_size: int | None = 1,
    weight_model_path: str | None = None,
    weight_model_class: str | None = None,
    weight_warm_pool_mode: str = "disabled",
    weight_warm_pool_components: list[str] | tuple[str, ...] | str | None = None,
    weight_warm_pool_max_gb: float = 0.0,
) -> torch.nn.Module:
    """Load a model with optional FSDP (Fully Sharded Data Parallel) support.

    Args:
        param_dtype: Data type for model parameters, also used for:
            - Model initialization context (set_default_torch_dtype)
            - FSDP mixed precision policy
            - Weight loading and casting
        reduce_dtype: Data type for gradient reduction in FSDP mixed precision.
        strict: If True, enforce strict state dict loading (all keys must match).
    """
    # NOTE(will): cast_forward_inputs=True shouldn't be needed as we are
    # manually casting the inputs to the model
    default_torch_dtype = param_dtype if param_dtype else torch.bfloat16
    mp_policy = MixedPrecisionPolicy(
        default_torch_dtype, reduce_dtype, output_dtype, cast_forward_inputs=False
    )

    set_mixed_precision_policy(
        param_dtype=default_torch_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        mp_policy=mp_policy,
    )

    with set_default_torch_dtype(default_torch_dtype), torch.device("meta"):
        model = model_cls(**init_params)

    # Check if we should use FSDP
    use_fsdp = fsdp_inference

    # Disable FSDP for MPS as it's not compatible
    if current_platform.is_mps():
        use_fsdp = False
        logger.info("Disabling FSDP for MPS platform as it's not compatible")

    if use_fsdp:
        world_size = hsdp_replicate_dim * hsdp_shard_dim
        if not fsdp_inference:
            hsdp_replicate_dim = world_size
            hsdp_shard_dim = 1

        device_mesh = init_device_mesh(
            current_platform.device_type,
            # (Replicate(), Shard(dim=0))
            mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
            mesh_dim_names=("replicate", "shard"),
        )
        shard_model(
            model,
            cpu_offload=cpu_offload,
            reshard_after_forward=True,
            mp_policy=mp_policy,
            mesh=device_mesh,
            fsdp_shard_conditions=model._fsdp_shard_conditions,
            pin_cpu_memory=pin_cpu_memory,
        )

    broadcast_decision = resolve_rank0_broadcast_decision(
        load_mode=weight_load_mode,
        broadcast_components=weight_broadcast_components,
        component_name=weight_component,
        tp_size=weight_tp_size,
        fsdp_inference=use_fsdp,
    )
    set_profile_load_mode(weight_load_profile, broadcast_decision)
    if broadcast_decision.requested_mode == "rank0-broadcast":
        logger.info(
            "Diffusion weight load mode requested=%s effective=%s "
            "component=%s sp_rank=%s sp_world_size=%s reason=%s",
            broadcast_decision.requested_mode,
            broadcast_decision.effective_mode,
            weight_component,
            broadcast_decision.sp_rank,
            broadcast_decision.sp_world_size,
            broadcast_decision.reason,
            main_process_only=False,
            local_main_process_only=False,
        )

    def _load_weights_on_current_rank(*, force_device_resident: bool = False) -> None:
        warm_pool_requested = normalize_weight_warm_pool_mode(weight_warm_pool_mode)
        warm_pool_key = None
        warm_pool_entry = None
        preloaded_custom_param_sd = None
        preloaded_reverse_param_names_mapping = None

        if weight_load_profile is not None:
            weight_load_profile.set_warm_pool_requested(warm_pool_requested)
            weight_load_profile.set_warm_pool_effective("disabled")
            weight_load_profile.set_warm_pool_hit(False)

        staging_mode = normalize_weight_staging_mode(weight_staging_mode)
        warm_pool_enabled = (
            broadcast_decision.enabled
            and broadcast_decision.sp_rank == 0
            and staging_mode == "pageable"
            and is_weight_warm_pool_enabled(
                mode=warm_pool_requested,
                components=weight_warm_pool_components,
                component=weight_component,
            )
        )
        if (
            broadcast_decision.enabled
            and broadcast_decision.sp_rank == 0
            and warm_pool_requested != "disabled"
            and staging_mode != "pageable"
            and weight_load_profile is not None
        ):
            weight_load_profile.set_warm_pool_error(
                f"requires_pageable_staging:got_{staging_mode}"
            )

        if warm_pool_enabled:
            if weight_load_profile is not None:
                weight_load_profile.set_warm_pool_effective("pageable")
            try:
                warm_pool_key = build_weight_warm_pool_key(
                    component=weight_component or "unknown",
                    model_path=weight_model_path or "",
                    safetensors_files=weight_dir_list,
                    dtype=param_dtype,
                    component_class=weight_model_class or model_cls.__name__,
                )
                warm_pool_entry = get_weight_warm_pool_entry(warm_pool_key)
                if warm_pool_entry is not None:
                    preloaded_custom_param_sd = warm_pool_entry.tensors
                    preloaded_reverse_param_names_mapping = (
                        warm_pool_entry.reverse_param_names_mapping
                    )
                    if weight_load_profile is not None:
                        weight_load_profile.set_warm_pool_hit(True)
            except Exception as exc:
                warm_pool_key = None
                if weight_load_profile is not None:
                    weight_load_profile.set_warm_pool_effective("disabled")
                    weight_load_profile.set_warm_pool_error(
                        f"{type(exc).__name__}: {exc}"
                    )

        def _store_warm_pool_entry(
            custom_param_sd: dict[str, torch.Tensor],
            reverse_param_names_mapping: dict[str, Any],
        ) -> None:
            if warm_pool_key is None or warm_pool_entry is not None:
                return
            try:
                entry = make_weight_warm_pool_entry(
                    custom_param_sd,
                    reverse_param_names_mapping=reverse_param_names_mapping,
                )
                stored = put_weight_warm_pool_entry(
                    warm_pool_key,
                    entry,
                    max_gb=weight_warm_pool_max_gb,
                )
                if weight_load_profile is not None:
                    weight_load_profile.set_warm_pool_store_bytes(
                        entry.bytes if stored else 0
                    )
            except Exception as exc:
                if weight_load_profile is not None:
                    weight_load_profile.set_warm_pool_error(
                        f"{type(exc).__name__}: {exc}"
                    )

        use_runai_model_streamer = None
        if broadcast_decision.enabled and broadcast_decision.sp_rank == 0:
            # RunAI model streamer can synchronize across ranks while non-rank0
            # waits for rank0 load status in rank0-broadcast mode.
            use_runai_model_streamer = False

        if preloaded_custom_param_sd is None:
            if weight_load_profile is not None:
                read_backend = resolve_safetensors_read_backend(
                    use_runai_model_streamer
                )
                if (
                    broadcast_decision.enabled
                    and broadcast_decision.sp_rank == 0
                    and use_runai_model_streamer is False
                ):
                    read_backend = "rank0-broadcast-no-runai"
                weight_load_profile.set_read_backend(read_backend)
            weight_iterator = safetensors_weights_iterator(
                weight_dir_list,
                use_runai_model_streamer=use_runai_model_streamer,
            )
            if weight_load_profile is not None:
                weight_iterator = weight_load_profile.profile_safetensors_iterator(
                    weight_iterator
                )
            weight_iterator = maybe_stage_weight_iterator(
                weight_iterator,
                staging_mode=staging_mode,
                weight_load_profile=weight_load_profile,
            )
        else:
            if weight_load_profile is not None:
                weight_load_profile.set_read_backend("warm-pool")
            weight_iterator = iter(())

        param_names_mapping_fn = get_param_names_mapping(model.param_names_mapping)
        load_model_from_full_model_state_dict(
            model,
            weight_iterator,
            device,
            param_dtype,
            strict=strict,
            cpu_offload=(cpu_offload and not force_device_resident),
            param_names_mapping=param_names_mapping_fn,
            weight_load_profile=weight_load_profile,
            preloaded_custom_param_sd=preloaded_custom_param_sd,
            preloaded_reverse_param_names_mapping=preloaded_reverse_param_names_mapping,
            on_custom_param_sd_materialized=_store_warm_pool_entry
            if warm_pool_enabled
            else None,
        )

    if broadcast_decision.enabled:
        confirm_rank0_broadcast_entry(
            broadcast_decision.sp_group,
            component_name=weight_component,
            weight_load_profile=weight_load_profile,
        )
        rank0_load_exc: Exception | None = None
        rank0_load_status: dict[str, Any] | None = None
        local_ready_for_tensor_broadcast = True
        local_ready_error: str | None = None
        if broadcast_decision.sp_rank == 0:
            log_broadcast_stage(
                "rank0_load_start",
                broadcast_decision.sp_group,
                component_name=weight_component,
                detail=f"file_count={len(weight_dir_list)}",
                weight_load_profile=weight_load_profile,
            )
            try:
                _load_weights_on_current_rank(force_device_resident=cpu_offload)
            except Exception as exc:
                rank0_load_exc = exc
                error = f"rank0_load_error:{type(exc).__name__}: {exc}"
                rank0_load_status = {"ok": False, "error": error}
                if weight_load_profile is not None:
                    weight_load_profile.set_broadcast_error(error)
                log_broadcast_stage(
                    "rank0_load_error",
                    broadcast_decision.sp_group,
                    component_name=weight_component,
                    detail=error,
                    weight_load_profile=weight_load_profile,
                )
            else:
                rank0_load_status = {"ok": True, "error": None}
                log_broadcast_stage(
                    "rank0_load_done",
                    broadcast_decision.sp_group,
                    component_name=weight_component,
                    weight_load_profile=weight_load_profile,
                )
        else:
            if weight_load_profile is not None:
                weight_load_profile.set_read_backend("rank0-broadcast-receive")
            log_broadcast_stage(
                "nonrank_empty_materialize_start",
                broadcast_decision.sp_group,
                component_name=weight_component,
                weight_load_profile=weight_load_profile,
            )
            try:
                materialize_empty_model_state_dict(model, device=device, strict=strict)
            except Exception as exc:
                local_ready_for_tensor_broadcast = False
                local_ready_error = (
                    f"nonrank_empty_materialize_error:"
                    f"{type(exc).__name__}: {exc}"
                )
                if weight_load_profile is not None:
                    weight_load_profile.set_broadcast_error(local_ready_error)
                log_broadcast_stage(
                    "nonrank_empty_materialize_error",
                    broadcast_decision.sp_group,
                    component_name=weight_component,
                    detail=local_ready_error,
                    weight_load_profile=weight_load_profile,
                )
            else:
                log_broadcast_stage(
                    "nonrank_empty_materialize_done",
                    broadcast_decision.sp_group,
                    component_name=weight_component,
                    weight_load_profile=weight_load_profile,
                )

        received_status = broadcast_rank0_load_status(
            broadcast_decision.sp_group,
            rank0_status=rank0_load_status,
            component_name=weight_component,
            weight_load_profile=weight_load_profile,
        )
        if not received_status.get("ok"):
            error = str(received_status.get("error") or "rank0 load failed")
            if weight_load_profile is not None:
                weight_load_profile.set_broadcast_error(error)
            if rank0_load_exc is not None:
                raise rank0_load_exc
            raise RuntimeError(error)
        confirm_tensor_broadcast_ready(
            broadcast_decision.sp_group,
            local_ok=local_ready_for_tensor_broadcast,
            component_name=weight_component,
            local_error=local_ready_error,
            weight_load_profile=weight_load_profile,
        )

        broadcast_module_tensors(
            model,
            broadcast_decision.sp_group,
            component_name=weight_component,
            weight_load_profile=weight_load_profile,
        )
        if cpu_offload:
            log_broadcast_stage(
                "cpu_offload_start",
                broadcast_decision.sp_group,
                component_name=weight_component,
                weight_load_profile=weight_load_profile,
            )
            with (
                weight_load_profile.timing_scope(WEIGHT_LOAD_D2H_OR_OFFLOAD_MS)
                if weight_load_profile is not None
                else nullcontext()
            ):
                offload_model_tensors_to_cpu(model)
            log_broadcast_stage(
                "cpu_offload_done",
                broadcast_decision.sp_group,
                component_name=weight_component,
                weight_load_profile=weight_load_profile,
            )
    else:
        _load_weights_on_current_rank()

    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and hasattr(
            quant_method, "process_weights_after_loading"
        ):
            if _is_npu and not isinstance(quant_method, UnquantizedLinearMethod):
                # Activate the NZ format for storing weights,
                # which is a specific optimization for Ascend NPU
                torch.npu.config.allow_internal_format = True
            quant_method.process_weights_after_loading(module)
            if _is_npu:
                torch.npu.empty_cache()

    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
        # Avoid unintended computation graph accumulation during inference
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad = False
    return model


def shard_model(
    model,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = MixedPrecisionPolicy(),  # noqa
    mesh: DeviceMesh | None = None,
    fsdp_shard_conditions: list[Callable[[str, nn.Module], bool]] = [],  # noqa
    pin_cpu_memory: bool = True,
) -> None:
    """
    Utility to shard a model with FSDP using the PyTorch Distributed fully_shard API.

    This method will over the model's named modules from the bottom-up and apply shard modules
    based on whether they meet any of the criteria from shard_conditions.

    Args:
        model (TransformerDecoder): Model to shard with FSDP.
        cpu_offload (bool): If set to True, FSDP will offload parameters, gradients, and optimizer
            states to CPU.
        reshard_after_forward (bool): Whether to reshard parameters and buffers after
            the forward pass. Setting this to True corresponds to the FULL_SHARD sharding strategy
            from FSDP1, while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.
        mesh (Optional[DeviceMesh]): Device mesh to use for FSDP sharding under multiple parallelism.
            Default to None.
        fsdp_shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP.
        pin_cpu_memory (bool): If set to True, FSDP will pin the CPU memory of the offloaded parameters.

    """
    if fsdp_shard_conditions is None or len(fsdp_shard_conditions) == 0:
        logger.warning(
            "The FSDP shard condition list is empty or None. No modules will be sharded in %s",
            type(model).__name__,
        )
        return

    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": mesh,
        "mp_policy": mp_policy,
    }
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=pin_cpu_memory)

    # iterating in reverse to start with
    # lowest-level modules first
    num_layers_sharded = 0
    # TODO(will): don't reshard after forward for the last layer to save on the
    # all-gather that will immediately happen Shard the model with FSDP,
    for n, m in reversed(list(model.named_modules())):
        if any([shard_condition(n, m) for shard_condition in fsdp_shard_conditions]):  # type: ignore
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)


# TODO(PY): device mesh for cfg parallel
def load_model_from_full_model_state_dict(
    model: FSDPModule | torch.nn.Module,
    full_sd_iterator: Generator[tuple[str, torch.Tensor], None, None],
    device: torch.device,
    param_dtype: torch.dtype | None,
    strict: bool = False,
    cpu_offload: bool = False,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
    preloaded_custom_param_sd: dict[str, torch.Tensor] | None = None,
    preloaded_reverse_param_names_mapping: dict[str, Any] | None = None,
    on_custom_param_sd_materialized: Callable[
        [dict[str, torch.Tensor], dict[str, Any]], None
    ]
    | None = None,
) -> _IncompatibleKeys:
    """
    Converting full state dict into a sharded state dict
    and loading it into FSDP model (if training) or normal huggingface model
    Args:
        model (Union[FSDPModule, torch.nn.Module]): Model to generate fully qualified names for cpu_state_dict
        full_sd_iterator (Generator): an iterator yielding (param_name, tensor) pairs
        device (torch.device): device used to move full state dict tensors
        param_dtype (torch.dtype): dtype used to move full state dict tensors. If none, respect original dtype from checkpoint
        strict (bool): flag to check if to load the model in strict mode
        cpu_offload (bool): flag to check if FSDP offload is enabled
        param_names_mapping (Optional[Callable[[str], str]]): a function that maps full param name to sharded param name
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    """
    meta_sd = model.state_dict()
    param_dict = dict(model.named_parameters())

    if preloaded_custom_param_sd is None:
        # map names from checkpoint to customized names
        materialize_start = time.perf_counter()
        read_before_ms = (
            weight_load_profile.get_ms(WEIGHT_LOAD_READ_SAFETENSORS_MS)
            if weight_load_profile is not None
            else 0.0
        )
        pin_before_ms = (
            weight_load_profile.get_ms(WEIGHT_LOAD_PIN_MEMORY_MS)
            if weight_load_profile is not None
            else 0.0
        )
        custom_param_sd, reverse_param_names_mapping = hf_to_custom_state_dict(
            full_sd_iterator, param_names_mapping
        )  # type: ignore
        if weight_load_profile is not None:
            elapsed_ms = (time.perf_counter() - materialize_start) * 1000.0
            read_delta_ms = (
                weight_load_profile.get_ms(WEIGHT_LOAD_READ_SAFETENSORS_MS)
                - read_before_ms
            )
            pin_delta_ms = (
                weight_load_profile.get_ms(WEIGHT_LOAD_PIN_MEMORY_MS) - pin_before_ms
            )
            weight_load_profile.add_ms(
                WEIGHT_LOAD_CPU_MATERIALIZE_MS,
                max(0.0, elapsed_ms - read_delta_ms - pin_delta_ms),
            )
        if on_custom_param_sd_materialized is not None:
            on_custom_param_sd_materialized(
                custom_param_sd,
                reverse_param_names_mapping,
            )
    else:
        custom_param_sd = preloaded_custom_param_sd
        reverse_param_names_mapping = dict(
            preloaded_reverse_param_names_mapping or {}
        )

    is_fsdp_model = isinstance(model, FSDPModule) or any(
        hasattr(p, "device_mesh") for p in meta_sd.values()
    )

    # sort parameter names to ensure all ranks process parameters in the same order
    sorted_param_names = sorted(custom_param_sd.keys())

    sharded_sd = {}
    skipped_checkpoint_keys: list[str] = []

    # shard from loaded state_dict, custom_param_sd -> sharded_sd
    for target_param_name in sorted_param_names:
        full_tensor = custom_param_sd[target_param_name]
        meta_sharded_param = meta_sd.get(target_param_name)

        if meta_sharded_param is None:
            # For FSDP models, ensure all ranks process parameters consistently
            if strict or is_fsdp_model:
                raise ValueError(
                    f"Parameter {target_param_name} not found in custom model state dict. The hf to custom mapping may be incorrect."
                )
            else:
                skipped_checkpoint_keys.append(target_param_name)
                continue

        # use meta param dtype so quantized params (e.g. FP8) keep their dtype;
        # for non-quantized models meta dtype equals param_dtype anyway
        if meta_sharded_param is None:
            # for nunchaku, some scales are patched later
            target_dtype = full_tensor.dtype
        else:
            target_dtype = meta_sharded_param.dtype

        if not hasattr(meta_sharded_param, "device_mesh"):
            with (
                weight_load_profile.timing_scope(WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS)
                if weight_load_profile is not None
                else nullcontext()
            ):
                full_tensor = full_tensor.to(device=device, dtype=target_dtype)
                actual_param = param_dict.get(target_param_name)
                weight_loader = (
                    getattr(actual_param, "weight_loader", None)
                    if actual_param is not None
                    else None
                )
                if weight_loader is not None:
                    assert actual_param is not None
                    sharded_tensor = torch.empty_like(
                        meta_sharded_param, device=device, dtype=target_dtype
                    )
                    # Preserve requires_grad flag to avoid errors with non-floating dtypes
                    requires_grad = getattr(meta_sharded_param, "requires_grad", False)
                    temp_param = _make_param_like(actual_param, sharded_tensor)
                    if not (
                        sharded_tensor.is_floating_point()
                        or sharded_tensor.is_complex()
                    ):
                        requires_grad = False
                    temp_param.requires_grad = requires_grad
                    weight_loader(temp_param, full_tensor)
                    sharded_tensor = temp_param.data
                else:
                    # In cases where parts of the model aren't sharded, some parameters will be plain tensors
                    sharded_tensor = full_tensor

            # Important: `cpu_offload` is intended for FSDP-managed parameter movement.
            # If a parameter is not sharded into a DTensor (i.e., no `device_mesh`), FSDP
            # will NOT manage it. Offloading it here would leave CPU parameters that
            # later participate in GPU kernels (e.g., conv/embedding), causing device/dtype
            # mismatches like "Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type)".
            #
            # Therefore:
            # - For non-FSDP models, keep the historical behavior (allow CPU offload).
            # - For FSDP models, do NOT offload non-sharded parameters here.
            if cpu_offload and not is_fsdp_model:
                with (
                    weight_load_profile.timing_scope(WEIGHT_LOAD_D2H_OR_OFFLOAD_MS)
                    if weight_load_profile is not None
                    else nullcontext()
                ):
                    sharded_tensor = sharded_tensor.cpu()
        else:
            with (
                weight_load_profile.timing_scope(WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS)
                if weight_load_profile is not None
                else nullcontext()
            ):
                full_tensor = full_tensor.to(device=device, dtype=target_dtype)
                sharded_tensor = distribute_tensor(
                    full_tensor,
                    meta_sharded_param.device_mesh,
                    meta_sharded_param.placements,
                )
            if cpu_offload:
                with (
                    weight_load_profile.timing_scope(WEIGHT_LOAD_D2H_OR_OFFLOAD_MS)
                    if weight_load_profile is not None
                    else nullcontext()
                ):
                    sharded_tensor = sharded_tensor.to("cpu")

        requires_grad = False
        sharded_sd[target_param_name] = nn.Parameter(
            sharded_tensor, requires_grad=requires_grad
        )

    model.reverse_param_names_mapping = reverse_param_names_mapping

    if skipped_checkpoint_keys:
        logger.warning(
            "Checkpoint keys not loaded (no matching model parameter) %s",
            (
                skipped_checkpoint_keys[:20]
                if len(skipped_checkpoint_keys) > 20
                else skipped_checkpoint_keys
            ),
        )
        if len(skipped_checkpoint_keys) > 20:
            logger.warning(
                "... and %d more skipped keys.",
                len(skipped_checkpoint_keys) - 20,
            )

    # parameters in nn.Module that doesn't exist in safetensor files
    unused_keys = set(meta_sd.keys()) - set(sharded_sd.keys())
    if unused_keys:
        logger.warning("Found unloaded parameters in meta state dict: %s", unused_keys)

    # for nunchaku; norm_q/norm_k for SANA QK normalization layers
    ALLOWED_NEW_PARAM_PATTERNS = [
        "gate_compress",
        "wcscales",
        "wtscale",
        "bias",
        "norm_q",
        "norm_k",
    ]
    for new_param_name in unused_keys:
        if not any(pattern in new_param_name for pattern in ALLOWED_NEW_PARAM_PATTERNS):
            logger.error(
                "Unsupported new parameter: %s. Allowed patterns: %s",
                new_param_name,
                ALLOWED_NEW_PARAM_PATTERNS,
            )
            raise ValueError(
                f"New parameter '{new_param_name}' is not supported. "
                f"Currently only parameters containing {ALLOWED_NEW_PARAM_PATTERNS} are allowed."
            )

        meta_sharded_param = meta_sd.get(new_param_name)
        meta_sharded_param_dtype = meta_sharded_param.dtype

        if any(
            p in new_param_name for p in ("wcscales", "wtscale", "norm_q", "norm_k")
        ):
            init_like = torch.ones_like
        else:
            init_like = torch.zeros_like

        if not hasattr(meta_sharded_param, "device_mesh"):
            with (
                weight_load_profile.timing_scope(WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS)
                if weight_load_profile is not None
                else nullcontext()
            ):
                sharded_tensor = init_like(
                    meta_sharded_param, device=device, dtype=meta_sharded_param_dtype
                )
            if cpu_offload and not is_fsdp_model:
                with (
                    weight_load_profile.timing_scope(WEIGHT_LOAD_D2H_OR_OFFLOAD_MS)
                    if weight_load_profile is not None
                    else nullcontext()
                ):
                    sharded_tensor = sharded_tensor.cpu()
        else:
            with (
                weight_load_profile.timing_scope(WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS)
                if weight_load_profile is not None
                else nullcontext()
            ):
                full_tensor = init_like(
                    meta_sharded_param, device=device, dtype=meta_sharded_param_dtype
                )
                sharded_tensor = distribute_tensor(
                    full_tensor,
                    meta_sharded_param.device_mesh,
                    meta_sharded_param.placements,
                )
            if cpu_offload:
                with (
                    weight_load_profile.timing_scope(WEIGHT_LOAD_D2H_OR_OFFLOAD_MS)
                    if weight_load_profile is not None
                    else nullcontext()
                ):
                    sharded_tensor = sharded_tensor.cpu()
        sharded_sd[new_param_name] = nn.Parameter(sharded_tensor)

    # choose `assign=True` since we cannot call `copy_` on meta tensor
    with (
        weight_load_profile.timing_scope(WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS)
        if weight_load_profile is not None
        else nullcontext()
    ):
        return model.load_state_dict(sharded_sd, strict=strict, assign=True)
