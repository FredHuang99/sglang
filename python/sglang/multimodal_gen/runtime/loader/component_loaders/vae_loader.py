import importlib.util
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.loader.weight_broadcast import (
    Rank0BroadcastDecision,
    broadcast_module_tensors,
    broadcast_rank0_load_status,
    confirm_rank0_broadcast_entry,
    confirm_tensor_broadcast_ready,
    log_broadcast_stage,
    resolve_rank0_broadcast_decision,
    set_profile_load_mode,
)
from sglang.multimodal_gen.runtime.loader.weight_staging import (
    maybe_stage_weight_iterator,
    normalize_weight_staging_mode,
    should_stage_weights_on_current_rank,
)
from sglang.multimodal_gen.runtime.loader.weight_warm_pool import (
    build_weight_warm_pool_key,
    get_weight_warm_pool_entry,
    is_weight_warm_pool_enabled,
    make_weight_warm_pool_entry,
    normalize_weight_warm_pool_mode,
    put_weight_warm_pool_entry,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.weight_load_profiler import (
    WEIGHT_LOAD_CPU_MATERIALIZE_MS,
    WEIGHT_LOAD_DISCOVER_FILES_MS,
    WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS,
    WEIGHT_LOAD_READ_SAFETENSORS_MS,
    DiffusionWeightLoadProfiler,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _load_vae_state_dict_from_safetensors(
    safetensors_list: list[str],
    server_args: ServerArgs,
    weight_load_profile: DiffusionWeightLoadProfiler,
) -> dict[str, torch.Tensor]:
    loaded: dict[str, torch.Tensor] = {}
    stage_weights = should_stage_weights_on_current_rank(
        server_args.diffusion_weight_staging
    )
    for sf_path in safetensors_list:
        with weight_load_profile.timing_scope(WEIGHT_LOAD_READ_SAFETENSORS_MS):
            tensors = safetensors_load_file(sf_path)
        weight_load_profile.add_tensor_collection_bytes(tensors.values())

        if stage_weights:
            weight_iterator = maybe_stage_weight_iterator(
                tensors.items(),
                staging_mode=server_args.diffusion_weight_staging,
                weight_load_profile=weight_load_profile,
            )
            for name, tensor in weight_iterator:
                with weight_load_profile.timing_scope(WEIGHT_LOAD_CPU_MATERIALIZE_MS):
                    loaded[name] = tensor
        else:
            with weight_load_profile.timing_scope(WEIGHT_LOAD_CPU_MATERIALIZE_MS):
                loaded.update(tensors)
    return loaded


def _get_vae_key_mismatches(
    vae: nn.Module, loaded_keys: set[str]
) -> tuple[list[str], list[str]]:
    state_keys = set(vae.state_dict().keys())
    missing_keys = sorted(state_keys - loaded_keys)
    unexpected_keys = sorted(loaded_keys - state_keys)
    return missing_keys, unexpected_keys


def _warn_vae_key_mismatches(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    should_log: bool,
) -> None:
    if not should_log:
        return
    if missing_keys:
        logger.warning("VAE missing keys: %s", missing_keys)
    if unexpected_keys:
        logger.warning("VAE unexpected keys: %s", unexpected_keys)


def _load_vae_weights_default(
    vae: nn.Module,
    safetensors_list: list[str],
    server_args: ServerArgs,
    weight_load_profile: DiffusionWeightLoadProfiler,
    preloaded_state_dict: dict[str, torch.Tensor] | None = None,
    on_state_dict_loaded=None,
) -> tuple[list[str], list[str]]:
    if preloaded_state_dict is None:
        loaded = _load_vae_state_dict_from_safetensors(
            safetensors_list,
            server_args,
            weight_load_profile,
        )
        if on_state_dict_loaded is not None:
            on_state_dict_loaded(loaded)
    else:
        loaded = preloaded_state_dict
    with weight_load_profile.timing_scope(WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS):
        vae.load_state_dict(loaded, strict=False)
    return _get_vae_key_mismatches(vae, set(loaded.keys()))


def _module_has_meta_tensor(module: nn.Module) -> str | None:
    for name, tensor in module.state_dict().items():
        if isinstance(tensor, torch.Tensor) and tensor.is_meta:
            return name
    return None


def _load_vae_weights_rank0_broadcast(
    vae: nn.Module,
    safetensors_list: list[str],
    server_args: ServerArgs,
    component_name: str,
    component_model_path: str,
    component_class: str,
    component_dtype: torch.dtype,
    weight_load_profile: DiffusionWeightLoadProfiler,
    broadcast_decision: Rank0BroadcastDecision,
) -> tuple[list[str], list[str]]:
    confirm_rank0_broadcast_entry(
        broadcast_decision.sp_group,
        component_name=component_name,
        weight_load_profile=weight_load_profile,
    )

    rank0_load_exc: Exception | None = None
    rank0_load_status: dict | None = None
    if broadcast_decision.sp_rank == 0:
        warm_pool_requested = normalize_weight_warm_pool_mode(
            server_args.diffusion_weight_warm_pool
        )
        warm_pool_key = None
        warm_pool_entry = None
        preloaded_state_dict = None

        weight_load_profile.set_warm_pool_requested(warm_pool_requested)
        weight_load_profile.set_warm_pool_effective("disabled")
        weight_load_profile.set_warm_pool_hit(False)

        staging_mode = normalize_weight_staging_mode(
            server_args.diffusion_weight_staging
        )
        warm_pool_enabled = (
            staging_mode == "pageable"
            and is_weight_warm_pool_enabled(
                mode=warm_pool_requested,
                components=server_args.diffusion_weight_warm_pool_components,
                component=component_name,
            )
        )
        if warm_pool_requested != "disabled" and staging_mode != "pageable":
            weight_load_profile.set_warm_pool_error(
                f"requires_pageable_staging:got_{staging_mode}"
            )

        if warm_pool_enabled:
            weight_load_profile.set_warm_pool_effective("pageable")
            try:
                warm_pool_key = build_weight_warm_pool_key(
                    component=component_name,
                    model_path=component_model_path,
                    safetensors_files=safetensors_list,
                    dtype=component_dtype,
                    component_class=component_class,
                )
                warm_pool_entry = get_weight_warm_pool_entry(warm_pool_key)
                if warm_pool_entry is not None:
                    preloaded_state_dict = warm_pool_entry.tensors
                    weight_load_profile.set_warm_pool_hit(True)
            except Exception as exc:
                warm_pool_key = None
                weight_load_profile.set_warm_pool_effective("disabled")
                weight_load_profile.set_warm_pool_error(
                    f"{type(exc).__name__}: {exc}"
                )

        def _store_warm_pool_entry(loaded: dict[str, torch.Tensor]) -> None:
            if warm_pool_key is None or warm_pool_entry is not None:
                return
            try:
                entry = make_weight_warm_pool_entry(loaded)
                stored = put_weight_warm_pool_entry(
                    warm_pool_key,
                    entry,
                    max_gb=server_args.diffusion_weight_warm_pool_max_gb,
                )
                weight_load_profile.set_warm_pool_store_bytes(
                    entry.bytes if stored else 0
                )
            except Exception as exc:
                weight_load_profile.set_warm_pool_error(
                    f"{type(exc).__name__}: {exc}"
                )

        log_broadcast_stage(
            "rank0_load_start",
            broadcast_decision.sp_group,
            component_name=component_name,
            detail=f"file_count={len(safetensors_list)}",
            weight_load_profile=weight_load_profile,
        )
        try:
            missing_keys, unexpected_keys = _load_vae_weights_default(
                vae,
                safetensors_list,
                server_args,
                weight_load_profile,
                preloaded_state_dict=preloaded_state_dict,
                on_state_dict_loaded=_store_warm_pool_entry
                if warm_pool_enabled
                else None,
            )
        except Exception as exc:
            rank0_load_exc = exc
            error = f"rank0_load_error:{type(exc).__name__}: {exc}"
            rank0_load_status = {"ok": False, "error": error}
            weight_load_profile.set_broadcast_error(error)
            log_broadcast_stage(
                "rank0_load_error",
                broadcast_decision.sp_group,
                component_name=component_name,
                detail=error,
                weight_load_profile=weight_load_profile,
            )
        else:
            rank0_load_status = {
                "ok": True,
                "error": None,
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
            }
            log_broadcast_stage(
                "rank0_load_done",
                broadcast_decision.sp_group,
                component_name=component_name,
                weight_load_profile=weight_load_profile,
            )

    received_status = broadcast_rank0_load_status(
        broadcast_decision.sp_group,
        rank0_status=rank0_load_status,
        component_name=component_name,
        weight_load_profile=weight_load_profile,
    )
    if not received_status.get("ok"):
        error = str(received_status.get("error") or "rank0 VAE load failed")
        weight_load_profile.set_broadcast_error(error)
        if rank0_load_exc is not None:
            raise rank0_load_exc
        raise RuntimeError(error)

    meta_tensor_name = _module_has_meta_tensor(vae)
    local_ok = meta_tensor_name is None
    local_error = (
        None
        if local_ok
        else f"vae_meta_tensor_before_broadcast:{meta_tensor_name}"
    )
    if local_error is not None:
        weight_load_profile.set_broadcast_error(local_error)
    confirm_tensor_broadcast_ready(
        broadcast_decision.sp_group,
        local_ok=local_ok,
        component_name=component_name,
        local_error=local_error,
        weight_load_profile=weight_load_profile,
    )

    broadcast_module_tensors(
        vae,
        broadcast_decision.sp_group,
        component_name=component_name,
        weight_load_profile=weight_load_profile,
    )
    return (
        list(received_status.get("missing_keys") or []),
        list(received_status.get("unexpected_keys") or []),
    )


def _convert_conv3d_weights_to_channels_last_3d(module: nn.Module) -> int:
    """
    Convert Conv3d weights to channels_last_3d (NDHWC) memory format.
    Returns the number of Conv3d modules converted.
    """
    if not hasattr(torch, "channels_last_3d"):
        return 0
    num_converted = 0
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            try:
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                num_converted += 1
            except Exception:
                # Best-effort; skip unsupported cases.
                continue
    return num_converted


class VAELoader(ComponentLoader):
    """Shared loader for (video/audio) VAE modules."""

    component_names = ["vae", "audio_vae", "video_vae"]
    expected_library = "diffusers"

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ):
        return server_args.vae_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the VAE based on the model path, and inference args."""
        weight_load_profile = DiffusionWeightLoadProfiler.from_server_args(
            server_args, component_name
        )
        error: str | None = None
        try:
            return self._load_customized_with_profile(
                component_model_path,
                server_args,
                component_name,
                weight_load_profile,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            weight_load_profile.finalize(
                status="error" if error else "success", error=error
            )

    def _load_customized_with_profile(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        weight_load_profile: DiffusionWeightLoadProfiler,
    ):
        config = get_diffusers_component_config(component_path=component_model_path)
        class_name = config.pop("_class_name", None)
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        server_args.model_paths[component_name] = component_model_path

        if component_name in ("vae", "video_vae"):
            pipeline_vae_config_attr = "vae_config"
            pipeline_vae_precision = "vae_precision"
        elif component_name in ("audio_vae",):
            pipeline_vae_config_attr = "audio_vae_config"
            pipeline_vae_precision = "audio_vae_precision"
        else:
            raise ValueError(
                f"Unsupported module name for VAE loader: {component_name}"
            )
        vae_config = getattr(server_args.pipeline_config, pipeline_vae_config_attr)
        vae_precision = getattr(server_args.pipeline_config, pipeline_vae_precision)
        vae_config.update_model_arch(config)
        if hasattr(vae_config, "post_init"):
            # NOTE: some post init logics are only available after updated with config
            vae_config.post_init()

        should_offload = self.should_offload(server_args)
        target_device = self.target_device(should_offload)

        # Check for auto_map first (custom VAE classes)
        auto_map = config.get("auto_map", {})
        auto_model_map = auto_map.get("AutoModel")
        if auto_model_map:
            module_path, cls_name = auto_model_map.rsplit(".", 1)
            custom_module_file = os.path.join(component_model_path, f"{module_path}.py")
            spec = importlib.util.spec_from_file_location("_custom", custom_module_file)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            vae_cls = getattr(custom_module, cls_name)
            vae_dtype = PRECISION_TO_TYPE[vae_precision]
            with set_default_torch_dtype(vae_dtype):
                vae = vae_cls.from_pretrained(
                    component_model_path,
                    revision=server_args.revision,
                    trust_remote_code=server_args.trust_remote_code,
                )
            vae = vae.to(device=target_device, dtype=vae_dtype)
            if (
                component_name in ("vae", "video_vae")
                and torch.cuda.is_available()
                and getattr(envs, "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D", False)
            ):
                n = _convert_conv3d_weights_to_channels_last_3d(vae)
                if n > 0:
                    logger.info(
                        "VAE: converted %d Conv3d weights to channels_last_3d", n
                    )
            vae = current_platform.optimize_vae(vae)
            return vae

        # Load from ModelRegistry (standard VAE classes)
        with (
            set_default_torch_dtype(PRECISION_TO_TYPE[vae_precision]),
            skip_init_modules(),
        ):
            vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vae = vae_cls(vae_config).to(target_device)

        with weight_load_profile.timing_scope(WEIGHT_LOAD_DISCOVER_FILES_MS):
            safetensors_list = _list_safetensors_files(component_model_path)
        assert (
            len(safetensors_list) >= 1
        ), f"Found no safetensors files in {component_model_path}"

        broadcast_decision = resolve_rank0_broadcast_decision(
            load_mode=server_args.diffusion_weight_load_mode,
            broadcast_components=server_args.diffusion_weight_broadcast_components,
            component_name=component_name,
            tp_size=1,
            fsdp_inference=False,
        )
        if broadcast_decision.enabled and target_device.type == "cpu":
            logger.warning(
                "Diffusion rank0-broadcast for VAE requires device-resident "
                "weights; falling back to default loader because vae_cpu_offload "
                "is enabled."
            )
            broadcast_decision = Rank0BroadcastDecision(
                enabled=False,
                requested_mode=broadcast_decision.requested_mode,
                effective_mode="default",
                reason="vae_cpu_offload_enabled",
                sp_rank=broadcast_decision.sp_rank,
                sp_world_size=broadcast_decision.sp_world_size,
                sp_group=broadcast_decision.sp_group,
            )
        set_profile_load_mode(weight_load_profile, broadcast_decision)
        if broadcast_decision.requested_mode == "rank0-broadcast":
            logger.info(
                "Diffusion weight load mode requested=%s effective=%s "
                "component=%s sp_rank=%s sp_world_size=%s reason=%s",
                broadcast_decision.requested_mode,
                broadcast_decision.effective_mode,
                component_name,
                broadcast_decision.sp_rank,
                broadcast_decision.sp_world_size,
                broadcast_decision.reason,
                main_process_only=False,
                local_main_process_only=False,
            )

        if broadcast_decision.enabled:
            missing_keys, unexpected_keys = _load_vae_weights_rank0_broadcast(
                vae,
                safetensors_list,
                server_args,
                component_name,
                component_model_path,
                class_name,
                PRECISION_TO_TYPE[vae_precision],
                weight_load_profile,
                broadcast_decision,
            )
        else:
            missing_keys, unexpected_keys = _load_vae_weights_default(
                vae,
                safetensors_list,
                server_args,
                weight_load_profile,
            )
        _warn_vae_key_mismatches(
            missing_keys,
            unexpected_keys,
            should_log=(
                not broadcast_decision.enabled or broadcast_decision.sp_rank == 0
            ),
        )

        if (
            component_name in ("vae", "video_vae")
            and torch.cuda.is_available()
            and getattr(envs, "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D", False)
        ):
            n = _convert_conv3d_weights_to_channels_last_3d(vae)
            if n > 0:
                logger.info("VAE: converted %d Conv3d weights to channels_last_3d", n)

        vae = current_platform.optimize_vae(vae)
        return vae
