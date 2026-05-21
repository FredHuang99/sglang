# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
import os
import pkgutil
import time
from abc import ABC
from typing import Any, Type

import torch
from diffusers import AutoModel
from torch import nn
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.utils import (
    _normalize_component_type,
    component_name_to_loader_cls,
    get_memory_usage_of_component,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import get_hf_config
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.launch_task_logger import record_task
from sglang.multimodal_gen.runtime.utils.module_load_profiler import (
    DiffusionModuleLoadProfiler,
)
from sglang.multimodal_gen.runtime.utils.profile_log_utils import (
    get_profile_log_context,
)

logger = init_logger(__name__)


class ComponentLoader(ABC):
    """Base class for loading a specific type of model component."""

    # the list of possible name of the component in model_index.json, e.g., scheduler
    component_names: list[str] = []

    # diffusers or transformers
    expected_library: str = ""

    _loaders_registered = False

    def __init_subclass__(cls, **kwargs):
        """
        register loaders, called when subclass is imported
        """
        super().__init_subclass__(**kwargs)
        for component_name in cls.component_names:
            component_name_to_loader_cls[component_name] = cls

    def __init__(self, device=None) -> None:
        self.device = device

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ):
        # not offload by default
        return False

    def target_device(self, should_offload):
        if should_offload:
            return (
                torch.device("mps")
                if current_platform.is_mps()
                else torch.device("cpu")
            )
        else:
            return get_local_torch_device()

    def load(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
    ) -> tuple[AutoModel, float]:
        """
        Template method that standardizes logging around the core load implementation.
        The priority of loading method is:
            1. load customized component
            2. load native diffusers/transformers component
        If all of the above methods failed, an error will be thrown

        """
        launch_task_start = time.perf_counter()
        launch_task_extra = {
            "component_model_path": component_model_path,
            "loader": self.__class__.__name__,
            "library": transformers_or_diffusers,
        }
        profile_logs_enabled = getattr(
            server_args, "profile_enabled", False
        ) or logger.isEnabledFor(logging.DEBUG)
        profile_ctx = (
            get_profile_log_context(server_args) if profile_logs_enabled else None
        )
        mem_before_loading = current_platform.get_available_gpu_memory()
        module_profile = (
            DiffusionModuleLoadProfiler.from_server_args(
                server_args,
                component=component_name,
                component_path=component_model_path,
                available_before_gb=mem_before_loading,
            )
            if getattr(server_args, "launch_module_profile_enabled", False)
            else None
        )
        if profile_ctx is not None:
            logger.info(
                "ProfileModuleLoadStart role=%s instance=%s rank=%s physical_rank=%s "
                "world_size=%s device=%s component=%s path=%s mem_kind=%s "
                "available_before_gb=%.2f",
                profile_ctx.role,
                profile_ctx.instance_id,
                profile_ctx.rank,
                profile_ctx.physical_rank,
                profile_ctx.world_size,
                profile_ctx.device,
                component_name,
                component_model_path,
                profile_ctx.mem_kind,
                mem_before_loading,
            )
        else:
            logger.info(
                "Loading %s from %s. avail mem: %.2f GB",
                component_name,
                component_model_path,
                mem_before_loading,
            )
        source: str | None = None
        fallback = False
        fallback_reason: str | None = None
        customized_error_type: str | None = None
        component_class: str | None = None
        model_size: Any | None = None
        current_mem: float | None = None
        consumed: float | None = None
        try:
            try:
                component = self.load_customized(
                    component_model_path, server_args, component_name
                )
                source = "sgl-diffusion"
            except Exception as e:
                fallback = True
                fallback_reason = str(e)
                customized_error_type = type(e).__name__
                if "Unsupported model architecture" in str(e):
                    logger.info(
                        "Component %s does not have a customized version yet; "
                        "using native version.",
                        component_name,
                    )
                else:
                    logger.warning(
                        "Customized component load failed; falling back to native. "
                        "component=%s rank=%s error_type=%s fallback_source=native "
                        "error=%s",
                        component_name,
                        getattr(profile_ctx, "rank", "unknown"),
                        customized_error_type,
                        fallback_reason,
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )
                # fallback to native version
                component = self.load_native(
                    component_model_path, server_args, transformers_or_diffusers
                )
                should_offload = self.should_offload(server_args)
                target_device = self.target_device(should_offload)
                component = component.to(device=target_device)
                source = "native"
                logger.warning(
                    "Native component %s: %s is loaded, performance may be sub-optimal",
                    component_name,
                    component.__class__.__name__,
                )

            if component is None:
                logger.error("Load %s failed", component_name)
                consumed = 0.0
            else:
                if isinstance(component, nn.Module):
                    component = component.eval()
                current_mem = current_platform.get_available_gpu_memory()
                model_size = get_memory_usage_of_component(component) or "NA"
                consumed = mem_before_loading - current_mem
                component_class = component.__class__.__name__
                if profile_ctx is not None:
                    logger.info(
                        "ProfileModuleLoadDone role=%s instance=%s rank=%s "
                        "physical_rank=%s world_size=%s device=%s component=%s "
                        "class=%s source=%s model_size_gb=%s mem_kind=%s "
                        "consumed_gb=%.2f available_after_gb=%.2f",
                        profile_ctx.role,
                        profile_ctx.instance_id,
                        profile_ctx.rank,
                        profile_ctx.physical_rank,
                        profile_ctx.world_size,
                        profile_ctx.device,
                        component_name,
                        component_class,
                        source,
                        model_size,
                        profile_ctx.mem_kind,
                        consumed,
                        current_mem,
                    )
                else:
                    logger.info(
                        "Loaded %s: %s (%s version). model size: %s GB, "
                        "consumed: %.2f GB, avail mem: %.2f GB",
                        component_name,
                        component_class,
                        source,
                        model_size,
                        consumed,
                        current_mem,
                    )
            if module_profile is not None:
                module_profile.finalize(
                    status="success" if component is not None else "error",
                    source=source,
                    component_class=component_class,
                    model_size_gb=model_size,
                    available_after_gb=current_mem,
                    consumed_gb=consumed,
                    error=None if component is not None else "component_none",
                    fallback=fallback,
                    fallback_reason=fallback_reason,
                    customized_error_type=customized_error_type,
                    transformers_or_diffusers=transformers_or_diffusers,
                )
        except Exception as exc:
            current_mem = current_platform.get_available_gpu_memory()
            if module_profile is not None:
                module_profile.finalize(
                    status="error",
                    source=source,
                    component_class=component_class,
                    model_size_gb=model_size,
                    available_after_gb=current_mem,
                    consumed_gb=mem_before_loading - current_mem,
                    error=f"{type(exc).__name__}: {exc}",
                    fallback=fallback,
                    fallback_reason=fallback_reason,
                    customized_error_type=customized_error_type,
                    transformers_or_diffusers=transformers_or_diffusers,
                )
            record_task(
                "component_load",
                start_perf=launch_task_start,
                component=component_name,
                status="error",
                extra=launch_task_extra,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        record_task(
            "component_load",
            start_perf=launch_task_start,
            component=component_name,
            status="ok" if component is not None else "error",
            extra=launch_task_extra,
            error=None if component is not None else "component_none",
        )
        return component, float(consumed or 0.0)

    def load_native(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str,
    ) -> AutoModel:
        """
        Load the component using the native library (transformers/diffusers).
        """
        if transformers_or_diffusers == "transformers":
            from transformers import AutoModel

            config = get_hf_config(
                component_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            return AutoModel.from_pretrained(
                component_model_path,
                config=config,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
        elif transformers_or_diffusers == "diffusers":
            from diffusers import AutoModel

            return AutoModel.from_pretrained(
                component_model_path,
                revision=server_args.revision,
                trust_remote_code=server_args.trust_remote_code,
            )
        else:
            raise ValueError(f"Unsupported library: {transformers_or_diffusers}")

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """
        Load the customized version component, implemented and optimized in SGL-diffusion
        """
        raise NotImplementedError(
            f"load_customized not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def _ensure_loaders_registered(cls):
        """
        avoid multiple registration
        """
        if cls._loaders_registered:
            return

        package_dir = os.path.dirname(__file__)
        package_name = (
            __package__
            or "sglang.multimodal_gen.runtime.loader.component_loaders.component_loaders"
        )

        for _, name, _ in pkgutil.iter_modules([package_dir]):
            # skip importing self to avoid circular dependency issues
            if name == "component_loader":
                continue
            try:
                importlib.import_module(f".{name}", package=package_name)
            except ImportError as e:
                logger.warning(f"Failed to import loader component {name}: {e}")

        cls._loaders_registered = True

    @classmethod
    def for_component_type(
        cls, component_name: str, transformers_or_diffusers: str
    ) -> "ComponentLoader":
        """
        Factory method to create a component loader for a specific component type.

        Args:
            component_name: Type of component (e.g., "vae", "text_encoder", "transformer", "scheduler")
            transformers_or_diffusers: Whether the component is from transformers or diffusers
        """
        cls._ensure_loaders_registered()

        # Map of component types to their loader classes and expected library
        component_name = _normalize_component_type(component_name)

        # NOTE(FlamingoPg): special for LTX-2 models
        if component_name == "vocoder" or component_name == "connectors":
            transformers_or_diffusers = "diffusers"

        # NOTE(CloudRipple): special for MOVA models
        # TODO(CloudRipple): remove most of these special cases after unifying the loading logic
        if component_name in [
            "audio_vae",
            "audio_dit",
            "dual_tower_bridge",
            "video_dit",
        ]:
            transformers_or_diffusers = "diffusers"

        if (
            component_name == "scheduler"
            and transformers_or_diffusers == "mova.diffusion.schedulers.flow_match_pair"
        ):
            transformers_or_diffusers = "diffusers"

        if component_name in component_name_to_loader_cls:
            loader_cls: Type[ComponentLoader] = component_name_to_loader_cls[
                component_name
            ]
            expected_library = loader_cls.expected_library
            # Assert that the library matches what's expected for this component type
            assert (
                transformers_or_diffusers == expected_library
            ), f"{component_name} must be loaded from {expected_library}, got {transformers_or_diffusers}"
            return loader_cls()

        # For unknown component types, use a generic loader
        logger.warning(
            "No specific loader found for component type: %s. Using generic loader.",
            component_name,
        )
        return GenericComponentLoader(transformers_or_diffusers)


class ImageProcessorLoader(ComponentLoader):
    """Loader for image processor."""

    component_names = ["image_processor"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> Any:
        return AutoImageProcessor.from_pretrained(component_model_path, use_fast=True)


class AutoProcessorLoader(ComponentLoader):
    """Loader for auto processor."""

    component_names = ["processor"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> Any:
        return AutoProcessor.from_pretrained(component_model_path)


class TokenizerLoader(ComponentLoader):
    """Loader for tokenizers."""

    component_names = ["tokenizer"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> Any:
        return AutoTokenizer.from_pretrained(
            component_model_path,
            padding_size="right",
        )


class GenericComponentLoader(ComponentLoader):
    """Generic loader for components that don't have a specific loader."""

    def __init__(self, library="transformers") -> None:
        super().__init__()
        self.library = library


class PipelineComponentLoader:
    """
    Utility class for loading the components in a pipeline.
    """

    @staticmethod
    def load_component(
        component_name: str,
        component_model_path: str,
        transformers_or_diffusers: str,
        server_args: ServerArgs,
    ):
        """
        Load a pipeline component.

        Args:
            component_name: Name of the component (e.g., "vae", "text_encoder", "transformer", "scheduler")
            component_model_path: Path to the component model
            transformers_or_diffusers: Whether the component is from transformers or diffusers

        """

        # Get the appropriate loader for this component type
        loader = ComponentLoader.for_component_type(
            component_name, transformers_or_diffusers
        )

        try:
            # Load the component
            return loader.load(
                component_model_path,
                server_args,
                component_name,
                transformers_or_diffusers,
            )
        except Exception as e:
            logger.error(
                f"Error while loading component: {component_name}, {component_model_path=}"
            )
            raise e
