# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig
    from sglang.multimodal_gen.configs.sample import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

__all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]


def __getattr__(name: str) -> Any:
    """Load public diffusion APIs only when they are explicitly requested.

    Importing a narrow runtime submodule must not construct the generic
    pipeline/engine dependency graph. This is also required by Jetson PyTorch
    builds that intentionally omit C10d.
    """

    if name == "DiffGenerator":
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        value = DiffGenerator
    elif name == "PipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig

        value = PipelineConfig
    elif name == "SamplingParams":
        from sglang.multimodal_gen.configs.sample import SamplingParams

        value = SamplingParams
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


# Trigger multimodal CI tests
