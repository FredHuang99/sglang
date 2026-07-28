"""Minimal SFWan2.1 realtime serving experiment.

The package intentionally keeps the single-model request protocol, FCFS engines,
and role-specific model execution separate from SGLang's generic diffusion
pipeline runtime.
"""

from .protocol import (
    DEFAULT_MODEL_PATH,
    GenerationRequest,
    JobState,
    LatentJobSpec,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "GenerationRequest",
    "JobState",
    "LatentJobSpec",
]
