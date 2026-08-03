"""Role-specific SFWan2.1 model loading and direct forward execution.

CUDA-heavy SGLang imports are intentionally local to constructors and methods so
the protocol, client, server wiring, and CPU tests remain importable without
initializing a device or downloading weights.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import msgspec

from .protocol import (
    DIT_KV_WINDOW_LATENT_FRAMES,
    LATENT_CHANNELS,
    LATENT_FRAMES_PER_CHUNK,
    GenerationRequest,
    decoded_frames_for_chunk,
)

VaePrecision = Literal["fp32", "fp16", "fp16_trt", "int8_trt"]
VaeTrtVariant = Literal["baseline", "fusion_v1"]
TRT_VAE_PRECISIONS = frozenset({"fp16_trt", "int8_trt"})


class ModelLoadConfig(msgspec.Struct, frozen=True, kw_only=True):
    model_path: str
    device_index: int = 0
    vae_precision: VaePrecision = "fp32"
    vae_engine_dir: str | None = None
    vae_trt_variant: VaeTrtVariant = "baseline"
    text_encoder_cpu_offload: bool = True
    dit_cpu_offload: bool = False
    vae_cpu_offload: bool = False
    enable_profile: bool = False
    enable_trt_layer_profile: bool = False
    enable_nvtx: bool = False


class DecodedChunk(msgspec.Struct, frozen=True, kw_only=True):
    chunk_index: int
    frames: Any
    frame_count: int
    metrics: dict[str, Any]


class MonolithicOutput(msgspec.Struct, frozen=True, kw_only=True):
    chunks: list[DecodedChunk]
    metrics: dict[str, Any]


class LatentChunk(msgspec.Struct, frozen=True, kw_only=True):
    """One normalized BCTHW block published only after clean-KV finishes."""

    chunk_index: int
    tensor: Any
    metrics: dict[str, Any]


DitChunkCallback = Callable[[LatentChunk], None]
DecodedChunkCallback = Callable[[DecodedChunk], None]
DIT_COMPONENT_NAMES = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "scheduler",
)
VAE_COMPONENT_NAMES = ("vae",)


def _uses_trt_vae(vae_precision: str) -> bool:
    return vae_precision in TRT_VAE_PRECISIONS


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _initialize_single_gpu_runtime(device_index: int) -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SFWan model execution requires a CUDA device")

    os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "cuda"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    os.environ["LOCAL_RANK"] = str(device_index)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.cuda.set_device(device_index)

    from sglang.multimodal_gen.runtime.distributed import (
        get_world_size,
        maybe_init_distributed_environment_and_model_parallel,
    )

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
        cfg_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        dp_size=1,
        distributed_init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
    )
    if get_world_size() != 1:
        raise RuntimeError("the minimal SFWan service supports world size one only")
    return torch.device(f"cuda:{device_index}")


def _distributed_runtime_contract() -> tuple[str, bool]:
    from sglang.multimodal_gen.runtime.distributed import (
        get_distributed_backend_name,
        is_local_single_process_mode,
    )

    return get_distributed_backend_name(), is_local_single_process_mode()


def _resolve_offload_settings(
    *,
    load_config: ModelLoadConfig,
    component_names: tuple[str, ...],
    local_single_process: bool,
) -> tuple[list[str] | None, dict[str, str]]:
    effective = {
        "text_encoder": (
            "fsdp_cpu_offload" if load_config.text_encoder_cpu_offload else "resident"
        ),
        "dit": "fsdp" if load_config.dit_cpu_offload else "resident",
        "vae": ("per_chunk_module" if load_config.vae_cpu_offload else "resident"),
    }
    if not local_single_process:
        return None, effective

    layerwise_components: list[str] = []
    if load_config.text_encoder_cpu_offload and "text_encoder" in component_names:
        layerwise_components.append("text_encoder")
        effective["text_encoder"] = "layerwise"
    if load_config.dit_cpu_offload and "transformer" in component_names:
        layerwise_components.append("dit")
        effective["dit"] = "layerwise"
    return layerwise_components or None, effective


def _component_runtime_contract(
    components: "_ComponentSet",
    component_keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "distributed_backend": components.distributed_backend,
        "cpu_offload_requested": {
            key: components.cpu_offload_requested[key] for key in component_keys
        },
        "cpu_offload_effective": {
            key: components.cpu_offload_effective[key] for key in component_keys
        },
    }


class _ComponentSet:
    """Load exact model-index components without constructing a pipeline."""

    def __init__(
        self,
        *,
        load_config: ModelLoadConfig,
        component_names: tuple[str, ...],
    ) -> None:
        self.device = _initialize_single_gpu_runtime(load_config.device_index)
        (
            self.distributed_backend,
            self.local_single_process,
        ) = _distributed_runtime_contract()
        self.cpu_offload_requested = {
            "text_encoder": load_config.text_encoder_cpu_offload,
            "dit": load_config.dit_cpu_offload,
            "vae": load_config.vae_cpu_offload,
        }
        (
            layerwise_offload_components,
            self.cpu_offload_effective,
        ) = _resolve_offload_settings(
            load_config=load_config,
            component_names=component_names,
            local_single_process=self.local_single_process,
        )

        from sglang.multimodal_gen.configs.pipeline_configs import (
            SelfForcingWanT2V480PConfig,
        )
        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs,
            set_global_server_args,
        )
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model,
        )

        self.model_path = maybe_download_model(
            load_config.model_path,
            force_diffusers_model=True,
        )
        self.pipeline_config = SelfForcingWanT2V480PConfig()
        # Generic component loaders know only native Torch precision names.
        # A TensorRT monolithic process loads DiT components here and creates
        # its VAE directly from plans, so keep this otherwise-unused field valid.
        self.pipeline_config.vae_precision = (
            "fp16"
            if _uses_trt_vae(load_config.vae_precision)
            else load_config.vae_precision
        )
        layerwise_selection = set(layerwise_offload_components or ())
        self.server_args = ServerArgs(
            model_path=self.model_path,
            pipeline_config=self.pipeline_config,
            num_gpus=1,
            tp_size=1,
            sp_degree=1,
            performance_mode="manual",
            text_encoder_cpu_offload=(
                load_config.text_encoder_cpu_offload
                and "text_encoder" not in layerwise_selection
            ),
            dit_cpu_offload=(
                load_config.dit_cpu_offload and "dit" not in layerwise_selection
            ),
            vae_cpu_offload=load_config.vae_cpu_offload,
            use_fsdp_inference=(
                load_config.dit_cpu_offload and not self.local_single_process
            ),
            layerwise_offload_components=layerwise_offload_components,
        )
        set_global_server_args(self.server_args)
        model_index_path = Path(self.model_path) / "model_index.json"
        with model_index_path.open(encoding="utf-8") as model_index_file:
            self.model_index = json.load(model_index_file)

        self.modules: dict[str, Any] = {}
        for name in component_names:
            self.modules[name] = self._load_component(name)
        if layerwise_offload_components:
            from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
                configure_layerwise_offload_modules,
            )

            configured = configure_layerwise_offload_modules(
                self.modules,
                self.server_args,
                component_names=layerwise_offload_components,
            )
            expected = set()
            if "text_encoder" in layerwise_offload_components:
                expected.add("text_encoder")
            if "dit" in layerwise_offload_components:
                expected.add("transformer")
            missing = expected.difference(configured)
            if missing:
                raise RuntimeError(
                    "Jetson local CPU offload requires layerwise-capable "
                    f"components; configuration failed for {sorted(missing)}."
                )

    def _load_component(self, name: str) -> Any:
        from sglang.multimodal_gen.runtime.loader.component_loaders import (
            component_loader,
        )

        descriptor = self.model_index.get(name)
        if not isinstance(descriptor, list) or len(descriptor) != 2:
            raise ValueError(
                f"model_index.json has no valid [{name}] component descriptor"
            )
        library, architecture = descriptor
        component_path = str(Path(self.model_path) / name)
        module, _memory_usage = component_loader.PipelineComponentLoader.load_component(
            component_name=name,
            component_model_path=component_path,
            transformers_or_diffusers=library,
            server_args=self.server_args,
            component_architecture=architecture,
        )
        if hasattr(module, "eval"):
            module.eval()
        return module


def _fastvideo_t5_postprocess(
    hidden_state: Any,
    attention_mask: Any,
    *,
    text_len: int = 512,
) -> Any:
    """Reproduce FastVideo's Wan T5 postprocess without a pipeline stage."""

    import torch

    if hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("T5 hidden state and attention mask must be rank 3 and 2")
    if hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("T5 hidden state and attention mask shapes do not match")
    if hidden_state.shape[0] != 1:
        raise ValueError("the minimal SFWan service supports batch size one")
    if hidden_state.shape[1] > text_len:
        raise ValueError(f"T5 sequence length exceeds {text_len}")
    if torch.isnan(hidden_state).any():
        raise ValueError("T5 produced NaN prompt embeddings")

    sequence_lengths = attention_mask.gt(0).sum(dim=1).long()
    prompt_embeds = []
    for row, sequence_length in zip(hidden_state, sequence_lengths, strict=True):
        valid = row[: int(sequence_length.item())]
        prompt_embeds.append(
            torch.cat(
                (
                    valid,
                    valid.new_zeros(text_len - valid.shape[0], valid.shape[1]),
                )
            )
        )
    return torch.stack(prompt_embeds, dim=0)


def _map_fastvideo_dmd_timesteps(
    *,
    scheduler_timesteps: Any,
    raw_dmd_steps: list[int] | tuple[int, ...],
    num_train_timesteps: int,
) -> Any:
    """Map raw DMD indices exactly as FastVideo's causal denoising stage."""

    import torch

    timesteps_cpu = scheduler_timesteps.detach().to(device="cpu")
    if timesteps_cpu.ndim != 1 or len(timesteps_cpu) != num_train_timesteps:
        raise ValueError(
            "SFWan scheduler must expose exactly "
            f"{num_train_timesteps} one-dimensional timesteps"
        )
    raw = torch.tensor(raw_dmd_steps, dtype=torch.long)
    if torch.any(raw < 0) or torch.any(raw > num_train_timesteps):
        raise ValueError("DMD timestep indices are outside the scheduler range")
    with_zero = torch.cat((timesteps_cpu, torch.tensor([0], dtype=torch.float32)))
    return with_zero[num_train_timesteps - raw].to(torch.float32)


def _pred_noise_to_pred_video_fastvideo(
    *,
    pred_noise: Any,
    noise_input_latent: Any,
    timestep: Any,
    scheduler: Any,
) -> Any:
    """FastVideo's FP64 flow-matching conversion, returning prediction dtype."""

    import torch

    if timestep.ndim == 2:
        timestep = timestep.flatten(0, 1)
        if timestep.numel() != noise_input_latent.shape[0]:
            raise ValueError("timestep count does not match flattened latent batch")
    elif timestep.ndim == 1:
        if timestep.shape[0] == 1:
            timestep = timestep.expand(noise_input_latent.shape[0])
        elif timestep.numel() != noise_input_latent.shape[0]:
            raise ValueError("timestep count does not match flattened latent batch")
    else:
        raise ValueError(f"invalid timestep shape: {tuple(timestep.shape)}")

    output_dtype = pred_noise.dtype
    device = pred_noise.device
    pred_noise_fp64 = pred_noise.double().to(device)
    noise_input_fp64 = noise_input_latent.double().to(device)
    sigmas = scheduler.sigmas.double().to(device)
    scheduler_timesteps = scheduler.timesteps.double().to(device)
    timestep_ids = torch.argmin(
        (scheduler_timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
        dim=1,
    )
    sigma_t = sigmas[timestep_ids].reshape(-1, 1, 1, 1)
    return (noise_input_fp64 - sigma_t * pred_noise_fp64).to(output_dtype)


def _denormalize_vae_latents_fastvideo(
    normalized_latents: Any,
    *,
    latents_mean: Any,
    latents_std: Any,
) -> Any:
    """Widen clean BF16 values, then apply FastVideo's FP32 Wan denorm."""

    import torch

    normalized_fp32 = normalized_latents.to(torch.float32)
    mean_fp32 = latents_mean.to(
        device=normalized_fp32.device,
        dtype=torch.float32,
    )
    std_fp32 = latents_std.to(
        device=normalized_fp32.device,
        dtype=torch.float32,
    )
    return normalized_fp32 * std_fp32 + mean_fp32


def _timed_cuda_call(
    *,
    label: str,
    enabled_profile: bool,
    enabled_nvtx: bool,
    function: Callable[[], Any],
) -> tuple[Any, float | None]:
    import torch

    if not enabled_profile:
        if enabled_nvtx and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(label)
            try:
                return function(), None
            finally:
                torch.cuda.nvtx.range_pop()
        return function(), None

    if not torch.cuda.is_available():
        start_time = time.perf_counter()
        result = function()
        return result, (time.perf_counter() - start_time) * 1000

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if enabled_nvtx:
        torch.cuda.nvtx.range_push(label)
    try:
        start.record()
        result = function()
        end.record()
        end.synchronize()
    finally:
        if enabled_nvtx:
            torch.cuda.nvtx.range_pop()
    return result, float(start.elapsed_time(end))


class SfWanDitModel:
    """SFWan prompt encoding, four-step DMD, and clean-KV forward."""

    _DIT_COMPONENTS = DIT_COMPONENT_NAMES

    def __init__(
        self,
        *,
        load_config: ModelLoadConfig,
        components: _ComponentSet | None = None,
    ) -> None:
        import torch

        self._load_config = load_config
        self._components = components or _ComponentSet(
            load_config=load_config,
            component_names=self._DIT_COMPONENTS,
        )
        missing = set(self._DIT_COMPONENTS).difference(self._components.modules)
        if missing:
            raise ValueError(f"DiT component set is missing {sorted(missing)}")
        self.device = self._components.device
        self.pipeline_config = self._components.pipeline_config
        self.server_args = self._components.server_args
        self.transformer = self._components.modules["transformer"]
        self.text_encoder = self._components.modules["text_encoder"]
        self.tokenizer = self._components.modules["tokenizer"]
        self.scheduler = self._components.modules["scheduler"]
        self.target_dtype = torch.bfloat16
        arch_config = self.transformer.config.arch_config
        self.num_layers = arch_config.num_layers
        self.num_frames_per_block = arch_config.num_frames_per_block
        self.sliding_window_num_frames = arch_config.sliding_window_num_frames
        if tuple(arch_config.patch_size) != (1, 2, 2):
            raise ValueError(
                f"SFWan transformer patch size must be (1, 2, 2), got "
                f"{tuple(arch_config.patch_size)}"
            )
        if int(arch_config.in_channels) != LATENT_CHANNELS:
            raise ValueError("SFWan transformer must consume 16 latent channels")
        if int(arch_config.out_channels) != LATENT_CHANNELS:
            raise ValueError("SFWan transformer must produce 16 latent channels")
        if int(arch_config.text_dim) != 4096:
            raise ValueError("SFWan transformer text dimension must be 4096")
        if self.num_frames_per_block != LATENT_FRAMES_PER_CHUNK:
            raise ValueError(
                f"checkpoint chunk size is {self.num_frames_per_block}, expected 3"
            )
        if self.sliding_window_num_frames != DIT_KV_WINDOW_LATENT_FRAMES:
            raise ValueError(
                "checkpoint causal KV window does not match the SFWan reference"
            )
        if bool(self.transformer.independent_first_frame):
            raise ValueError(
                "this minimal T2V service requires independent_first_frame=False"
            )
        if self.num_layers != 30:
            raise ValueError(
                f"SFWan transformer must have 30 layers, got {self.num_layers}"
            )
        if self.transformer.num_attention_heads != 12:
            raise ValueError(
                "SFWan transformer must have 12 attention heads, got "
                f"{self.transformer.num_attention_heads}"
            )
        if self.transformer.attention_head_dim != 128:
            raise ValueError(
                "SFWan transformer head dimension must be 128, got "
                f"{self.transformer.attention_head_dim}"
            )

        num_train_timesteps_value = getattr(
            self.scheduler,
            "num_train_timesteps",
            None,
        )
        if num_train_timesteps_value is None:
            num_train_timesteps_value = self.scheduler.config.num_train_timesteps
        num_train_timesteps = int(num_train_timesteps_value)
        if num_train_timesteps != 1000:
            raise ValueError(
                "SFWan scheduler must use 1000 training timesteps, got "
                f"{num_train_timesteps}"
            )
        scheduler_config = getattr(self.scheduler, "config", None)
        shift = float(
            getattr(self.scheduler, "shift", getattr(scheduler_config, "shift", -1))
        )
        if shift != 5.0:
            raise ValueError(f"SFWan scheduler shift must be 5, got {shift}")
        raw_dmd_steps = self.pipeline_config.dmd_denoising_steps
        if raw_dmd_steps != [1000, 750, 500, 250]:
            raise ValueError(
                f"SFWan DMD steps must be [1000, 750, 500, 250], got {raw_dmd_steps}"
            )
        if not bool(self.pipeline_config.warp_denoising_step):
            raise ValueError("SFWan requires warped DMD timestep indices")
        if int(getattr(self.pipeline_config, "context_noise", 0)) != 0:
            raise ValueError("SFWan clean-KV context_noise must be zero")
        if self.scheduler.sigmas.ndim != 1 or len(self.scheduler.sigmas) != 1000:
            raise ValueError("SFWan scheduler must expose exactly 1000 sigmas")
        self._dmd_timesteps_cpu = _map_fastvideo_dmd_timesteps(
            scheduler_timesteps=self.scheduler.timesteps,
            raw_dmd_steps=raw_dmd_steps,
            num_train_timesteps=num_train_timesteps,
        )

        self._kv_cache: list[Any] | None = None
        self._crossattn_cache: list[Any] | None = None
        self._cache_tokens = 0
        self.contract = {
            "batch_size": 1,
            "latent_channels": LATENT_CHANNELS,
            "latent_frames_per_chunk": LATENT_FRAMES_PER_CHUNK,
            "kv_window_latent_frames": self.sliding_window_num_frames,
            "dit_dtype": "bfloat16",
            "text_embedding_dtype": "float32",
            "text_length": 512,
            "cfg": False,
            "profile_enabled": load_config.enable_profile,
            **_component_runtime_contract(
                self._components,
                ("text_encoder", "dit"),
            ),
            "cpu_offload": {
                "text_encoder": load_config.text_encoder_cpu_offload,
                "dit": load_config.dit_cpu_offload,
            },
            "raw_dmd_steps": list(raw_dmd_steps),
            "mapped_dmd_timesteps": [
                float(value) for value in self._dmd_timesteps_cpu.tolist()
            ],
        }

    def generate(
        self,
        *,
        request: GenerationRequest,
        chunk_callback: DitChunkCallback,
    ) -> dict[str, Any]:
        import torch

        generation_start = time.perf_counter()
        profile_enabled = self._load_config.enable_profile
        cache_reset_ms = 0.0
        with torch.no_grad():
            try:
                prompt_embeds, text_metrics = self._encode_prompt(request.prompt)
                generator = torch.Generator(device="cpu").manual_seed(request.seed)
                latent_init_start = time.perf_counter() if profile_enabled else None
                latents, latent_init_cuda_ms = _timed_cuda_call(
                    label="sfwan.latent_init",
                    enabled_profile=profile_enabled,
                    enabled_nvtx=self._load_config.enable_nvtx,
                    function=lambda: self._prepare_initial_latents(
                        request,
                        generator=generator,
                        dtype=prompt_embeds.dtype,
                    ),
                )
                latent_init_ms = (
                    (time.perf_counter() - latent_init_start) * 1000
                    if latent_init_start is not None
                    else None
                )
                latent_height, latent_width = latents.shape[-2:]
                cache_start = time.perf_counter() if profile_enabled else None
                tokens_per_frame, cache_prepare_cuda_ms = _timed_cuda_call(
                    label="sfwan.cache_prepare",
                    enabled_profile=profile_enabled,
                    enabled_nvtx=self._load_config.enable_nvtx,
                    function=lambda: self._prepare_caches(
                        latent_height=latent_height,
                        latent_width=latent_width,
                    ),
                )
                cache_prepare_ms = (
                    (time.perf_counter() - cache_start) * 1000
                    if cache_start is not None
                    else None
                )
                timestep_start = time.perf_counter() if profile_enabled else None
                timesteps, timestep_to_device_cuda_ms = _timed_cuda_call(
                    label="sfwan.timesteps_to_device",
                    enabled_profile=profile_enabled,
                    enabled_nvtx=self._load_config.enable_nvtx,
                    function=lambda: self._dmd_timesteps_cpu.to(self.device),
                )
                timestep_to_device_wall_ms = (
                    (time.perf_counter() - timestep_start) * 1000
                    if timestep_start is not None
                    else None
                )
                chunk_metrics = []

                for chunk_index in range(request.total_chunks):
                    chunk_start = time.perf_counter() if profile_enabled else None
                    start_frame = chunk_index * LATENT_FRAMES_PER_CHUNK
                    chunk = latents[
                        :,
                        :,
                        start_frame : start_frame + LATENT_FRAMES_PER_CHUNK,
                        :,
                        :,
                    ]
                    clean_chunk, metrics = self._denoise_chunk(
                        chunk=chunk,
                        prompt_embeds=prompt_embeds,
                        timesteps=timesteps,
                        generator=generator,
                        current_start_tokens=start_frame * tokens_per_frame,
                        start_frame=start_frame,
                    )
                    if profile_enabled:
                        metrics["chunk_index"] = chunk_index
                        metrics["chunk_execution_wall_ms"] = (
                            time.perf_counter() - chunk_start
                        ) * 1000
                        chunk_metrics.append(metrics)
                    # This callback is intentionally after the clean-KV forward.
                    chunk_callback(
                        LatentChunk(
                            chunk_index=chunk_index,
                            tensor=clean_chunk,
                            metrics=metrics,
                        )
                    )
            finally:
                reset_start = time.perf_counter() if profile_enabled else None
                self._reset_caches()
                if reset_start is not None:
                    cache_reset_ms = (time.perf_counter() - reset_start) * 1000

        result: dict[str, Any] = {
            "num_chunks": request.total_chunks,
            "dit_total_ms": (time.perf_counter() - generation_start) * 1000,
        }
        if profile_enabled:
            assert latent_init_ms is not None
            assert latent_init_cuda_ms is not None
            assert cache_prepare_ms is not None
            assert cache_prepare_cuda_ms is not None
            assert timestep_to_device_cuda_ms is not None
            assert timestep_to_device_wall_ms is not None
            dit_execution_wall_ms = (
                float(text_metrics["text_encode_ms"])
                + latent_init_ms
                + cache_prepare_ms
                + timestep_to_device_wall_ms
                + sum(
                    float(metrics["chunk_execution_wall_ms"])
                    for metrics in chunk_metrics
                )
            )
            profile_execution = {
                "component": "dit",
                "num_chunks": request.total_chunks,
                **text_metrics,
                "latent_init_cuda_ms": latent_init_cuda_ms,
                "latent_init_ms": latent_init_ms,
                "cache_prepare_cuda_ms": cache_prepare_cuda_ms,
                "cache_prepare_ms": cache_prepare_ms,
                "timestep_to_device_cuda_ms": timestep_to_device_cuda_ms,
                "timestep_to_device_wall_ms": timestep_to_device_wall_ms,
                "chunks": chunk_metrics,
                "dit_execution_wall_ms": dit_execution_wall_ms,
            }
            result["cache_reset_ms"] = cache_reset_ms
            result["profile_execution"] = profile_execution
        return result

    def _encode_prompt(self, prompt: str) -> tuple[Any, dict[str, float]]:
        import torch

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        profile_enabled = self._load_config.enable_profile
        text_start = time.perf_counter() if profile_enabled else None
        tokenizer_start = time.perf_counter() if profile_enabled else None
        tokenized = self.tokenizer(
            [prompt],
            truncation=True,
            padding=True,
            max_length=512,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        tokenizer_ms = (
            (time.perf_counter() - tokenizer_start) * 1000
            if tokenizer_start is not None
            else None
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        if input_ids.dtype != torch.int64 or attention_mask.dtype != torch.int64:
            raise ValueError("SFWan tokenizer must return int64 IDs and attention mask")

        def _forward() -> Any:
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
            ):
                return self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )

        t5_start = time.perf_counter() if profile_enabled else None
        outputs, t5_cuda_ms = _timed_cuda_call(
            label="sfwan.text_encode",
            enabled_profile=self._load_config.enable_profile,
            enabled_nvtx=self._load_config.enable_nvtx,
            function=_forward,
        )
        t5_wall_ms = (
            (time.perf_counter() - t5_start) * 1000 if t5_start is not None else None
        )
        postprocess_start = time.perf_counter() if profile_enabled else None
        prompt_embeds, postprocess_cuda_ms = _timed_cuda_call(
            label="sfwan.text_postprocess",
            enabled_profile=profile_enabled,
            enabled_nvtx=self._load_config.enable_nvtx,
            function=lambda: _fastvideo_t5_postprocess(
                outputs.last_hidden_state,
                attention_mask,
            ),
        )
        postprocess_ms = (
            (time.perf_counter() - postprocess_start) * 1000
            if postprocess_start is not None
            else None
        )
        if tuple(prompt_embeds.shape[:2]) != (1, 512):
            raise ValueError(
                "SFWan prompt embedding must have shape [1, 512, hidden_size]"
            )
        if prompt_embeds.shape[2] != 4096:
            raise ValueError(
                f"SFWan prompt hidden size must be 4096, got {prompt_embeds.shape[2]}"
            )
        if prompt_embeds.dtype != torch.float32:
            raise ValueError(
                "FastVideo-compatible SFWan text embeddings must be float32, got "
                f"{prompt_embeds.dtype}"
            )
        prompt_to_device_start = time.perf_counter() if profile_enabled else None
        prompt_embeds, prompt_to_device_cuda_ms = _timed_cuda_call(
            label="sfwan.text_to_device",
            enabled_profile=profile_enabled,
            enabled_nvtx=self._load_config.enable_nvtx,
            function=lambda: prompt_embeds.to(self.device),
        )
        prompt_to_device_wall_ms = (
            (time.perf_counter() - prompt_to_device_start) * 1000
            if prompt_to_device_start is not None
            else None
        )
        if not profile_enabled:
            return prompt_embeds, {}
        assert text_start is not None
        assert tokenizer_ms is not None
        assert t5_wall_ms is not None
        assert t5_cuda_ms is not None
        assert postprocess_ms is not None
        assert postprocess_cuda_ms is not None
        assert prompt_to_device_cuda_ms is not None
        assert prompt_to_device_wall_ms is not None
        return prompt_embeds, {
            "tokenizer_wall_ms": tokenizer_ms,
            "t5_wall_ms": t5_wall_ms,
            "t5_cuda_ms": t5_cuda_ms,
            "t5_postprocess_cuda_ms": postprocess_cuda_ms,
            "t5_postprocess_wall_ms": postprocess_ms,
            "prompt_to_device_cuda_ms": prompt_to_device_cuda_ms,
            "prompt_to_device_wall_ms": prompt_to_device_wall_ms,
            "text_encode_ms": (time.perf_counter() - text_start) * 1000,
        }

    def _prepare_initial_latents(
        self,
        request: GenerationRequest,
        *,
        generator: Any,
        dtype: Any,
    ) -> Any:
        from diffusers.utils.torch_utils import randn_tensor

        shape = (
            1,
            LATENT_CHANNELS,
            (request.resolved_num_frames - 1) // 4 + 1,
            request.height // 8,
            request.width // 8,
        )
        return randn_tensor(
            shape,
            generator=generator,
            dtype=dtype,
            device=self.device,
        )

    def _prepare_caches(
        self,
        *,
        latent_height: int,
        latent_width: int,
    ) -> int:
        import torch

        from sglang.multimodal_gen.runtime.layers.kvcache import (
            causal_attention_cache,
        )

        patch_size = self.transformer.config.arch_config.patch_size
        patch_ratio = int(patch_size[1]) * int(patch_size[2])
        tokens_per_frame = (latent_height * latent_width) // patch_ratio
        cache_tokens = tokens_per_frame * self.sliding_window_num_frames
        num_heads = self.transformer.num_attention_heads
        head_dim = self.transformer.attention_head_dim
        max_text_len = self.pipeline_config.text_encoder_configs[0].arch_config.text_len

        if self._kv_cache is None or self._cache_tokens != cache_tokens:
            self._kv_cache = []
            self._crossattn_cache = []
            for _ in range(self.num_layers):
                self._kv_cache.append(
                    causal_attention_cache.CausalSelfAttentionKVCache(
                        k=torch.zeros(
                            (1, cache_tokens, num_heads, head_dim),
                            device=self.device,
                            dtype=self.target_dtype,
                        ),
                        v=torch.zeros(
                            (1, cache_tokens, num_heads, head_dim),
                            device=self.device,
                            dtype=self.target_dtype,
                        ),
                        global_end_index=torch.zeros(
                            1,
                            device=self.device,
                            dtype=torch.long,
                        ),
                        local_end_index=torch.zeros(
                            1,
                            device=self.device,
                            dtype=torch.long,
                        ),
                        cache_size=cache_tokens,
                        attention_window_size=cache_tokens,
                    )
                )
                self._crossattn_cache.append(
                    causal_attention_cache.CrossAttentionKVCache(
                        k=torch.zeros(
                            (1, max_text_len, num_heads, head_dim),
                            device=self.device,
                            dtype=self.target_dtype,
                        ),
                        v=torch.zeros(
                            (1, max_text_len, num_heads, head_dim),
                            device=self.device,
                            dtype=self.target_dtype,
                        ),
                    )
                )
            self._cache_tokens = cache_tokens
        else:
            for cache in self._kv_cache:
                cache.reset_indices()
            for cache in self._crossattn_cache:
                cache.reset()
        return tokens_per_frame

    def _reset_caches(self) -> None:
        if self._kv_cache is not None:
            for cache in self._kv_cache:
                cache.reset_indices()
        if self._crossattn_cache is not None:
            for cache in self._crossattn_cache:
                cache.reset()

    def _forward_transformer(
        self,
        *,
        latent: Any,
        prompt_embeds: Any,
        timestep: Any,
        current_timestep: int,
        current_start_tokens: int,
        start_frame: int,
        label: str,
    ) -> tuple[Any, float | None]:
        import torch

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        if self._kv_cache is None or self._crossattn_cache is None:
            raise RuntimeError("causal caches have not been initialized")

        def _forward() -> Any:
            with (
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.target_dtype,
                    enabled=self.device.type == "cuda",
                ),
                set_forward_context(
                    current_timestep=current_timestep,
                    attn_metadata=None,
                ),
            ):
                return self.transformer(
                    latent.to(self.target_dtype),
                    prompt_embeds,
                    timestep,
                    kv_cache=self._kv_cache,
                    crossattn_cache=self._crossattn_cache,
                    current_start=current_start_tokens,
                    start_frame=start_frame,
                )

        return _timed_cuda_call(
            label=label,
            enabled_profile=self._load_config.enable_profile,
            enabled_nvtx=self._load_config.enable_nvtx,
            function=_forward,
        )

    def _denoise_chunk(
        self,
        *,
        chunk: Any,
        prompt_embeds: Any,
        timesteps: Any,
        generator: Any,
        current_start_tokens: int,
        start_frame: int,
    ) -> tuple[Any, dict[str, Any]]:
        import torch

        current_latents = chunk
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        raw_shape = noise_latents_btchw.shape
        profile_enabled = self._load_config.enable_profile
        step_metrics = []

        for step_index, timestep in enumerate(timesteps):
            noise_latents = noise_latents_btchw.clone()
            transformer_timestep = timestep.reshape(1, 1).expand(
                current_latents.shape[0],
                1,
            )
            conversion_timestep = timestep.reshape(1, 1).expand(
                current_latents.shape[0],
                noise_latents.shape[1],
            )
            pred_noise, elapsed_ms = self._forward_transformer(
                latent=current_latents,
                prompt_embeds=prompt_embeds,
                timestep=transformer_timestep,
                current_timestep=step_index,
                current_start_tokens=current_start_tokens,
                start_frame=start_frame,
                label=f"sfwan.dit.chunk.{start_frame // 3}.step.{step_index}",
            )
            if pred_noise.dtype != self.target_dtype:
                raise ValueError(
                    f"SFWan DiT output must be bfloat16, got {pred_noise.dtype}"
                )
            if tuple(pred_noise.shape) != tuple(current_latents.shape):
                raise ValueError(
                    "SFWan DiT output shape does not match its current chunk"
                )
            pred_noise_btchw = pred_noise.permute(0, 2, 1, 3, 4)
            conversion_wall_start = time.perf_counter() if profile_enabled else None

            def _convert_prediction() -> Any:
                return _pred_noise_to_pred_video_fastvideo(
                    pred_noise=pred_noise_btchw.flatten(0, 1),
                    noise_input_latent=noise_latents.flatten(0, 1),
                    timestep=conversion_timestep,
                    scheduler=self.scheduler,
                ).unflatten(0, pred_noise_btchw.shape[:2])

            x0_btchw, conversion_cuda_ms = _timed_cuda_call(
                label=(
                    f"sfwan.dit.chunk.{start_frame // 3}.pred_to_video.{step_index}"
                ),
                enabled_profile=profile_enabled,
                enabled_nvtx=self._load_config.enable_nvtx,
                function=_convert_prediction,
            )
            conversion_wall_ms = (
                (time.perf_counter() - conversion_wall_start) * 1000
                if conversion_wall_start is not None
                else None
            )
            if profile_enabled:
                assert elapsed_ms is not None
                assert conversion_cuda_ms is not None
                assert conversion_wall_ms is not None
                step_metrics.append(
                    {
                        "step_index": step_index,
                        "timestep": float(timestep.item()),
                        "cuda_ms": elapsed_ms,
                        "pred_to_video_cuda_ms": conversion_cuda_ms,
                        "pred_to_video_wall_ms": conversion_wall_ms,
                    }
                )

            renoise_cuda_ms = None
            renoise_wall_ms = None
            if step_index < len(timesteps) - 1:
                next_timestep = timesteps[step_index + 1 : step_index + 2]
                renoise_wall_start = time.perf_counter() if profile_enabled else None

                def _renoise() -> Any:
                    noise = torch.randn(
                        raw_shape,
                        dtype=x0_btchw.dtype,
                        generator=generator,
                        device="cpu",
                    ).to(self.device)
                    return self.scheduler.add_noise(
                        x0_btchw.flatten(0, 1),
                        noise.flatten(0, 1),
                        next_timestep,
                    ).unflatten(0, x0_btchw.shape[:2])

                noise_latents_btchw, renoise_cuda_ms = _timed_cuda_call(
                    label=(f"sfwan.dit.chunk.{start_frame // 3}.renoise.{step_index}"),
                    enabled_profile=profile_enabled,
                    enabled_nvtx=self._load_config.enable_nvtx,
                    function=_renoise,
                )
                if noise_latents_btchw.dtype != self.target_dtype:
                    raise ValueError("SFWan re-noised model input must remain bfloat16")
                current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
                if renoise_wall_start is not None:
                    renoise_wall_ms = (time.perf_counter() - renoise_wall_start) * 1000
            else:
                current_latents = x0_btchw.permute(0, 2, 1, 3, 4)
            if profile_enabled:
                step_metrics[-1]["renoise_cuda_ms"] = renoise_cuda_ms
                step_metrics[-1]["renoise_wall_ms"] = renoise_wall_ms

        # FastVideo stores BF16 clean values in its FP32 parent before clean-KV.
        chunk.copy_(current_latents)

        context_noise = int(getattr(self.pipeline_config, "context_noise", 0))
        context_timestep = torch.full(
            (current_latents.shape[0], 1),
            context_noise,
            device=self.device,
            dtype=torch.long,
        )
        _unused, clean_kv_ms = self._forward_transformer(
            latent=current_latents,
            prompt_embeds=prompt_embeds,
            timestep=context_timestep,
            current_timestep=0,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            label=f"sfwan.dit.chunk.{start_frame // 3}.clean_kv",
        )
        metrics: dict[str, Any] = {}
        if profile_enabled:
            assert clean_kv_ms is not None
            metrics = {
                "denoise_steps": step_metrics,
                "clean_kv_cuda_ms": clean_kv_ms,
            }
        return current_latents.contiguous(), metrics


class SfWanVaeModel:
    """Stateful per-latent-frame Wan VAE decoder."""

    _VAE_COMPONENTS = VAE_COMPONENT_NAMES

    def __init__(
        self,
        *,
        load_config: ModelLoadConfig,
        components: _ComponentSet | None = None,
    ) -> None:
        import torch

        self._load_config = load_config
        self._trt_runtime: Any = None
        if _uses_trt_vae(load_config.vae_precision):
            self._initialize_trt(
                load_config=load_config,
                components=components,
                torch=torch,
            )
            return

        self._components = components or _ComponentSet(
            load_config=load_config,
            component_names=self._VAE_COMPONENTS,
        )
        if "vae" not in self._components.modules:
            raise ValueError("VAE component set is missing the decoder")
        self.device = self._components.device
        self.pipeline_config = self._components.pipeline_config
        self.server_args = self._components.server_args
        self.vae = self._components.modules["vae"]
        self.vae_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
        }[load_config.vae_precision]
        self._vae_weights_on_device = not load_config.vae_cpu_offload
        self.vae.to(
            device=("cpu" if load_config.vae_cpu_offload else self.device),
            dtype=self.vae_dtype,
        )
        if not bool(self.vae.use_feature_cache):
            raise ValueError("SFWan realtime decoding requires VAE feature cache")
        if int(self.vae.config.z_dim) != LATENT_CHANNELS:
            raise ValueError("SFWan VAE must use 16 latent channels")
        temporal_factor = 2 ** sum(
            bool(value) for value in self.vae.config.temperal_downsample
        )
        if temporal_factor != 4:
            raise ValueError(
                f"SFWan VAE temporal compression must be 4, got {temporal_factor}"
            )
        spatial_factor = 2 ** (len(self.vae.config.dim_mult) - 1)
        if spatial_factor != 8:
            raise ValueError(
                f"SFWan VAE spatial compression must be 8, got {spatial_factor}"
            )
        if int(self.vae.config.out_channels) != 3:
            raise ValueError("SFWan VAE decoder must produce three RGB channels")
        latents_mean = getattr(self.vae.config, "latents_mean", None)
        latents_std = getattr(self.vae.config, "latents_std", None)
        if latents_mean is None or latents_std is None:
            raise ValueError("SFWan VAE config must provide latents_mean/std")
        if len(latents_mean) != LATENT_CHANNELS or len(latents_std) != LATENT_CHANNELS:
            raise ValueError("SFWan VAE mean/std must contain 16 channels")
        self._latents_mean = torch.tensor(
            latents_mean,
            device=self.device,
            dtype=torch.float32,
        ).view(1, LATENT_CHANNELS, 1, 1, 1)
        self._latents_std = torch.tensor(
            latents_std,
            device=self.device,
            dtype=torch.float32,
        ).view(1, LATENT_CHANNELS, 1, 1, 1)
        self._request_active = False
        self.contract = {
            "batch_size": 1,
            "latent_layout": "BCTHW",
            "latent_dtype": "bfloat16",
            "latent_frames_per_chunk": LATENT_FRAMES_PER_CHUNK,
            "denormalize_dtype": "float32",
            "vae_dtype": load_config.vae_precision,
            "feature_cache_scope": "request",
            "profile_enabled": load_config.enable_profile,
            "vae_backend": "pytorch",
            "vae_engine_dir": None,
            "vae_engine_precision": load_config.vae_precision,
            "vae_engine_schema_version": None,
            "vae_engine_sm": None,
            "vae_engine_tensorrt_version": None,
            "vae_engine_cuda_version": None,
            "vae_runtime_gpu_name": str(torch.cuda.get_device_name(self.device)),
            "vae_runtime_compute_capability": list(
                torch.cuda.get_device_capability(self.device)
            ),
            "vae_runtime_cuda_version": str(torch.version.cuda),
            "vae_runtime_torch_version": str(torch.__version__),
            "vae_trt_variant": "baseline",
            "vae_engine_plan_sha256": None,
            "vae_trt_plugin_sha256": None,
            "vae_int8_audit_passed": False,
            "vae_cache_tensor_count": 32,
            "vae_cache_bank_bytes": None,
            **_component_runtime_contract(
                self._components,
                ("vae",),
            ),
            "cpu_offload": {
                "vae": load_config.vae_cpu_offload,
            },
        }

    def _initialize_trt(
        self,
        *,
        load_config: ModelLoadConfig,
        components: _ComponentSet | None,
        torch: Any,
    ) -> None:
        if load_config.vae_engine_dir is None:
            raise ValueError(
                f"vae_precision={load_config.vae_precision} requires --vae-engine-dir"
            )
        if load_config.vae_cpu_offload:
            raise ValueError("TensorRT VAE engines do not support --vae-cpu-offload")

        self._components = components
        if components is None:
            self.device = _initialize_single_gpu_runtime(load_config.device_index)
            distributed_backend, _local_mode = _distributed_runtime_contract()
            self.pipeline_config = None
            self.server_args = None
        else:
            self.device = components.device
            distributed_backend = components.distributed_backend
            self.pipeline_config = components.pipeline_config
            self.server_args = components.server_args

        from .vae_trt_runtime import TensorRTVaeRuntime

        trt_precision = {
            "fp16_trt": "fp16",
            "int8_trt": "int8",
        }[load_config.vae_precision]
        self._trt_runtime = TensorRTVaeRuntime(
            engine_dir=load_config.vae_engine_dir,
            precision=trt_precision,
            model_path=load_config.model_path,
            device=self.device,
            enable_profile=load_config.enable_profile,
            enable_trt_layer_profile=load_config.enable_trt_layer_profile,
            enable_nvtx=load_config.enable_nvtx,
            variant=load_config.vae_trt_variant,
        )
        manifest = self._trt_runtime.manifest
        latents_mean = manifest.get("latents_mean")
        latents_std = manifest.get("latents_std")
        if (
            not isinstance(latents_mean, list)
            or not isinstance(latents_std, list)
            or len(latents_mean) != LATENT_CHANNELS
            or len(latents_std) != LATENT_CHANNELS
        ):
            raise ValueError(
                "TensorRT VAE manifest must provide 16-channel latents_mean/std"
            )
        self._latents_mean = torch.tensor(
            latents_mean,
            device=self.device,
            dtype=torch.float32,
        ).view(1, LATENT_CHANNELS, 1, 1, 1)
        self._latents_std = torch.tensor(
            latents_std,
            device=self.device,
            dtype=torch.float32,
        ).view(1, LATENT_CHANNELS, 1, 1, 1)
        self.vae = None
        self.vae_dtype = torch.float16
        self._vae_weights_on_device = True
        self._request_active = False
        self.contract = {
            "batch_size": 1,
            "latent_layout": "BCTHW",
            "latent_dtype": "bfloat16",
            "latent_frames_per_chunk": LATENT_FRAMES_PER_CHUNK,
            "denormalize_dtype": "float32",
            "vae_dtype": load_config.vae_precision,
            "feature_cache_scope": "request",
            "profile_enabled": load_config.enable_profile,
            "distributed_backend": distributed_backend,
            "cpu_offload_requested": {"vae": False},
            "cpu_offload_effective": {"vae": "resident"},
            "cpu_offload": {"vae": False},
            **self._trt_runtime.contract,
        }

    def reset_request(self) -> None:
        self._request_active = False
        if self._trt_runtime is not None:
            self._trt_runtime.reset_request()
        else:
            self.vae.reset_causal_decode_state()
        self._request_active = True

    @property
    def trt_layer_profile_metadata(self) -> dict[str, Any] | None:
        if self._trt_runtime is None:
            return None
        return self._trt_runtime.layer_profile_metadata

    def finish_request(self) -> None:
        try:
            if self._trt_runtime is not None:
                self._trt_runtime.finish_request()
            else:
                self.vae.reset_causal_decode_state()
        finally:
            try:
                self._offload_vae_weights()
            finally:
                self._request_active = False

    def close(self) -> None:
        if self._trt_runtime is not None:
            self._trt_runtime.close()
            self._trt_runtime = None

    def _activate_vae_for_chunk(self) -> None:
        if not getattr(
            getattr(self, "_load_config", None),
            "vae_cpu_offload",
            False,
        ):
            return
        if self._vae_weights_on_device:
            return
        self.vae.to(device=self.device, dtype=self.vae_dtype)
        self._vae_weights_on_device = True

    def _offload_vae_weights(self) -> None:
        if not getattr(
            getattr(self, "_load_config", None),
            "vae_cpu_offload",
            False,
        ):
            return
        if not self._vae_weights_on_device:
            return
        self.vae.to(device="cpu", dtype=self.vae_dtype)
        self._vae_weights_on_device = False

    def decode_chunk(
        self,
        *,
        chunk_index: int,
        latents: Any,
        return_frames: bool = True,
    ) -> DecodedChunk:
        import torch

        if not self._request_active:
            raise RuntimeError("reset_request() must be called before decode_chunk()")
        if tuple(latents.shape[0:3]) != (1, LATENT_CHANNELS, 3):
            raise ValueError(
                f"expected latent prefix (1, 16, 3), got {tuple(latents.shape[0:3])}"
            )
        if latents.dtype != torch.bfloat16:
            raise ValueError(
                f"normalized latent chunks must be bfloat16, got {latents.dtype}"
            )
        if not latents.is_contiguous():
            raise ValueError("normalized latent chunks must be contiguous BCTHW")

        # Weight migration is intentionally outside profile_execution. The
        # request-level decoder feature cache is not a registered module
        # parameter/buffer, so it remains on the GPU between chunks.
        self._activate_vae_for_chunk()
        profile_enabled = self._load_config.enable_profile
        chunk_start_event = None
        chunk_end_event = None
        chunk_wall_start = None
        if profile_enabled:
            if getattr(self.device, "type", None) == "cuda":
                chunk_start_event = torch.cuda.Event(enable_timing=True)
                chunk_end_event = torch.cuda.Event(enable_timing=True)
                chunk_start_event.record()
            else:
                chunk_wall_start = time.perf_counter()

        ingress_start = time.perf_counter() if profile_enabled else None
        normalized, bf16_ingress_cuda_ms = _timed_cuda_call(
            label=f"sfwan.vae.chunk.{chunk_index}.bf16_ingress",
            enabled_profile=profile_enabled,
            enabled_nvtx=self._load_config.enable_nvtx,
            function=lambda: latents.to(self.device, dtype=torch.bfloat16),
        )
        bf16_ingress_ms = (
            (time.perf_counter() - ingress_start) * 1000
            if ingress_start is not None
            else None
        )
        denorm_start = time.perf_counter() if profile_enabled else None
        z, denorm_cuda_ms = _timed_cuda_call(
            label=f"sfwan.vae.chunk.{chunk_index}.denorm",
            enabled_profile=profile_enabled,
            enabled_nvtx=self._load_config.enable_nvtx,
            function=lambda: _denormalize_vae_latents_fastvideo(
                normalized,
                latents_mean=self._latents_mean,
                latents_std=self._latents_std,
            ),
        )
        denorm_ms = (
            (time.perf_counter() - denorm_start) * 1000
            if denorm_start is not None
            else None
        )
        trt_metrics: dict[str, Any] = {}
        if self._trt_runtime is not None:
            decoded, frame_metrics, trt_metrics = self._decode_trt_chunk(
                chunk_index=chunk_index,
                z=z,
            )
            post_quant_ms = None
        else:
            decoded, frame_metrics, post_quant_ms = self._decode_per_latent(
                chunk_index=chunk_index,
                z=z,
            )
        chunk_execution_ms = None
        if chunk_end_event is not None:
            chunk_end_event.record()
            chunk_end_event.synchronize()
            chunk_execution_ms = float(chunk_start_event.elapsed_time(chunk_end_event))
        elif chunk_wall_start is not None:
            chunk_execution_ms = (time.perf_counter() - chunk_wall_start) * 1000

        frame_count = int(decoded.shape[2])
        frames = None
        rgb_d2h_ms = None
        if return_frames:
            rgb_start = time.perf_counter() if profile_enabled else None
            image = (decoded / 2 + 0.5).clamp(0, 1)
            frames = image[0].permute(1, 2, 3, 0).mul(255).to(torch.uint8).cpu().numpy()
            if rgb_start is not None:
                rgb_d2h_ms = (time.perf_counter() - rgb_start) * 1000

        metrics: dict[str, Any] = {
            "chunk_index": chunk_index,
            "decoded_rgb_frames": frame_count,
            **{
                key: value
                for key, value in trt_metrics.items()
                if key in {"trt_engine_kind", "trt_precision"}
            },
        }
        if profile_enabled:
            assert bf16_ingress_ms is not None
            assert bf16_ingress_cuda_ms is not None
            assert denorm_ms is not None
            assert denorm_cuda_ms is not None
            assert chunk_execution_ms is not None
            profile_execution = {
                "chunk_index": chunk_index,
                "chunk_execution_cuda_ms": chunk_execution_ms,
                "bf16_ingress_cuda_ms": bf16_ingress_cuda_ms,
                "bf16_ingress_wall_ms": bf16_ingress_ms,
                "denorm_cuda_ms": denorm_cuda_ms,
                "denorm_wall_ms": denorm_ms,
                "post_quant_cuda_ms": post_quant_ms,
                "latent_frames": frame_metrics,
                "decoded_rgb_frames": frame_count,
                **trt_metrics,
            }
            metrics["profile_execution"] = profile_execution
            if rgb_d2h_ms is not None:
                metrics["rgb_d2h_wall_ms"] = rgb_d2h_ms
        result = DecodedChunk(
            chunk_index=chunk_index,
            frames=frames,
            frame_count=frame_count,
            metrics=metrics,
        )
        self._offload_vae_weights()
        return result

    def _decode_trt_chunk(
        self,
        *,
        chunk_index: int,
        z: Any,
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        if self._trt_runtime is None:
            raise RuntimeError("TensorRT VAE runtime is not initialized")
        raw_output, trt_metrics = self._trt_runtime.decode_chunk(
            chunk_index=chunk_index,
            denormalized_fp32=z,
        )
        output, finalize_ms = _timed_cuda_call(
            label=f"sfwan.vae.trt.chunk.{chunk_index}.output_finalize",
            enabled_profile=self._load_config.enable_profile,
            enabled_nvtx=self._load_config.enable_nvtx,
            function=lambda: raw_output.float().clamp(-1.0, 1.0),
        )
        frame_counts = (1, 4, 4) if chunk_index == 0 else (4, 4, 4)
        frame_metrics = [
            {
                "latent_index": latent_index,
                "cuda_ms": None,
                "decoded_rgb_frames": decoded_frames,
                "fused_in_trt": True,
            }
            for latent_index, decoded_frames in enumerate(frame_counts)
        ]
        if self._load_config.enable_profile:
            trt_metrics["trt_output_finalize_cuda_ms"] = finalize_ms
        return output, frame_metrics, trt_metrics

    def _decode_per_latent(
        self,
        *,
        chunk_index: int,
        z: Any,
    ) -> tuple[Any, list[dict[str, Any]], float | None]:
        import torch

        from sglang.multimodal_gen.runtime.models.vaes import wanvae as wanvae_module

        autocast_enabled = self.vae_dtype != torch.float32
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=self.vae_dtype,
                enabled=autocast_enabled,
            ),
        ):
            if not autocast_enabled:
                z = z.to(self.vae_dtype)
            x, post_quant_ms = _timed_cuda_call(
                label=f"sfwan.vae.chunk.{chunk_index}.post_quant",
                enabled_profile=self._load_config.enable_profile,
                enabled_nvtx=self._load_config.enable_nvtx,
                function=lambda: self.vae.post_quant_conv(z),
            )
            first_request_chunk = not self.vae._causal_decode_initialized
            outputs = []
            metrics = []
            spatial_context = (
                nullcontext()
                if self.vae._should_use_spatial_parallel_decode(z)
                else wanvae_module.disable_spatial_parallel_decode()
            )
            with (
                spatial_context,
                wanvae_module.forward_context(
                    feat_cache_arg=self.vae._feat_map,
                    feat_idx_arg=self.vae._conv_idx,
                ),
            ):
                for latent_index in range(x.shape[2]):
                    wanvae_module.feat_idx.set(0)
                    wanvae_module.first_chunk.set(
                        first_request_chunk and latent_index == 0
                    )
                    decoded, elapsed_ms = _timed_cuda_call(
                        label=(f"sfwan.vae.chunk.{chunk_index}.latent.{latent_index}"),
                        enabled_profile=self._load_config.enable_profile,
                        enabled_nvtx=self._load_config.enable_nvtx,
                        function=lambda index=latent_index: self.vae.decoder(
                            x[:, :, index : index + 1]
                        ),
                    )
                    outputs.append(decoded)
                    if self._load_config.enable_profile:
                        assert elapsed_ms is not None
                        metrics.append(
                            {
                                "latent_index": latent_index,
                                "cuda_ms": elapsed_ms,
                                "decoded_rgb_frames": int(decoded.shape[2]),
                            }
                        )

            output = torch.cat(outputs, dim=2)
            if self.vae.config.patch_size is not None:
                output = wanvae_module.unpatchify(
                    output,
                    patch_size=self.vae.config.patch_size,
                )
            output = output.float().clamp(-1.0, 1.0)
            self.vae._causal_decode_initialized = True
        return output, metrics, post_quant_ms


class SfWanMonolithicModel:
    """One-process model that interleaves each clean DiT chunk with VAE decode."""

    def __init__(self, *, load_config: ModelLoadConfig) -> None:
        self._load_config = load_config
        component_names = (
            DIT_COMPONENT_NAMES
            if _uses_trt_vae(load_config.vae_precision)
            else (*DIT_COMPONENT_NAMES, *VAE_COMPONENT_NAMES)
        )
        components = _ComponentSet(
            load_config=load_config,
            component_names=component_names,
        )
        self.dit = SfWanDitModel(
            load_config=load_config,
            components=components,
        )
        self.vae = SfWanVaeModel(
            load_config=load_config,
            components=components,
        )
        self.device = components.device
        self.contract = {
            "mode": "monolithic",
            "dit": self.dit.contract,
            "vae": self.vae.contract,
            "latent_transport": "in_memory_gpu",
            "profile_enabled": load_config.enable_profile,
            **_component_runtime_contract(
                components,
                ("text_encoder", "dit", "vae"),
            ),
        }

    def close(self) -> None:
        self.vae.close()

    def generate(
        self,
        *,
        request: GenerationRequest,
        chunk_callback: DecodedChunkCallback | None = None,
    ) -> MonolithicOutput:
        generation_start = time.perf_counter()
        decoded_chunks: list[DecodedChunk] = []

        def _decode_after_clean_kv(
            latent_chunk: LatentChunk,
        ) -> None:
            decoded = self.vae.decode_chunk(
                chunk_index=latent_chunk.chunk_index,
                latents=latent_chunk.tensor,
            )
            expected_frames = decoded_frames_for_chunk(latent_chunk.chunk_index)
            if decoded.frame_count != expected_frames:
                raise RuntimeError(
                    f"monolithic VAE chunk {latent_chunk.chunk_index} decoded "
                    f"{decoded.frame_count} frames; expected {expected_frames}"
                )
            metrics = dict(decoded.metrics)
            metrics["dit"] = latent_chunk.metrics
            completed_chunk = DecodedChunk(
                chunk_index=latent_chunk.chunk_index,
                frames=decoded.frames,
                frame_count=decoded.frame_count,
                metrics=metrics,
            )
            decoded_chunks.append(completed_chunk)
            if chunk_callback is not None:
                chunk_callback(completed_chunk)

        reset_attempted = False
        try:
            reset_attempted = True
            self.vae.reset_request()
            dit_metrics = self.dit.generate(
                request=request,
                chunk_callback=_decode_after_clean_kv,
            )
        finally:
            if reset_attempted:
                self.vae.finish_request()
        metrics: dict[str, Any] = {
            "dit": dit_metrics,
            "vae_chunks": [chunk.metrics for chunk in decoded_chunks],
            "monolithic_total_ms": (time.perf_counter() - generation_start) * 1000,
        }
        if self._load_config.enable_profile:
            vae_chunks = [
                chunk.metrics["profile_execution"]
                for chunk in decoded_chunks
                if "profile_execution" in chunk.metrics
            ]
            metrics["profile_execution"] = {
                "component": "monolithic",
                "dit": dit_metrics.get("profile_execution"),
                "vae": {
                    "num_chunks": len(vae_chunks),
                    "chunks": vae_chunks,
                    "vae_execution_cuda_ms": sum(
                        float(chunk["chunk_execution_cuda_ms"]) for chunk in vae_chunks
                    ),
                },
            }
        return MonolithicOutput(chunks=decoded_chunks, metrics=metrics)
