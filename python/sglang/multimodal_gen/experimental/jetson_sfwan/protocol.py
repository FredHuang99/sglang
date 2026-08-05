"""Wire protocol and model-shape invariants for the SFWan2.1 experiment."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_MODEL_PATH = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 81
DEFAULT_FPS = 16
DEFAULT_SEED = 1024
DEFAULT_MAX_PIXELS = DEFAULT_HEIGHT * DEFAULT_WIDTH

LATENT_CHANNELS = 16
LATENT_TEMPORAL_DOWNSAMPLE = 4
LATENT_SPATIAL_DOWNSAMPLE = 8
LATENT_FRAMES_PER_CHUNK = 3
DIT_KV_WINDOW_LATENT_FRAMES = 21
WIRE_DTYPE = "bfloat16"
WIRE_LAYOUT = "BCTHW"
SAFETENSORS_WIRE_DTYPE = "BF16"
MAX_SAFETENSORS_HEADER_BYTES = 65_536
LatentTransport = Literal["http", "shm"]
OFFICIAL_RESOLUTIONS = {
    (DEFAULT_HEIGHT, DEFAULT_WIDTH),
    (DEFAULT_WIDTH, DEFAULT_HEIGHT),
}


def resolve_video_num_frames(
    *,
    num_frames: int | None,
    duration_seconds: float | None,
    fps: int,
) -> int:
    """Resolve and strictly validate a physical-frame count.

    Wan's temporal VAE maps F physical frames to (F - 1) / 4 + 1 latent
    frames. SFWan denoises three latent frames at a time, so valid physical
    frame counts satisfy F % 12 == 9.
    """

    if num_frames is not None and duration_seconds is not None:
        raise ValueError("num_frames and duration_seconds are mutually exclusive")

    if duration_seconds is not None:
        raw_frames = duration_seconds * fps
        resolved = round(raw_frames)
        if not math.isclose(raw_frames, resolved, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "duration_seconds * fps must be an integer number of frames"
            )
        num_frames = resolved
    elif num_frames is None:
        num_frames = DEFAULT_NUM_FRAMES

    if num_frames < 9 or num_frames % 12 != 9:
        raise ValueError(
            "num_frames must satisfy num_frames % 12 == 9 (for example 81, 93, or 105)"
        )
    return num_frames


def latent_frame_count(num_frames: int) -> int:
    return (num_frames - 1) // LATENT_TEMPORAL_DOWNSAMPLE + 1


def latent_chunk_count(num_frames: int) -> int:
    latent_frames = latent_frame_count(num_frames)
    if latent_frames % LATENT_FRAMES_PER_CHUNK != 0:
        raise ValueError("latent frame count must be divisible by three")
    return latent_frames // LATENT_FRAMES_PER_CHUNK


def expected_latent_chunk_shape(height: int, width: int) -> tuple[int, ...]:
    return (
        1,
        LATENT_CHANNELS,
        LATENT_FRAMES_PER_CHUNK,
        height // LATENT_SPATIAL_DOWNSAMPLE,
        width // LATENT_SPATIAL_DOWNSAMPLE,
    )


def decoded_frames_for_chunk(chunk_index: int) -> int:
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return 9 if chunk_index == 0 else 12


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_frames: int | None = None
    duration_seconds: float | None = Field(default=None, gt=0)
    fps: int = Field(default=DEFAULT_FPS, gt=0)
    seed: int = DEFAULT_SEED

    @model_validator(mode="after")
    def validate_shape_and_duration(self) -> "GenerationRequest":
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError("height and width must both be divisible by 16")
        resolve_video_num_frames(
            num_frames=self.num_frames,
            duration_seconds=self.duration_seconds,
            fps=self.fps,
        )
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        return self

    @property
    def resolved_num_frames(self) -> int:
        return resolve_video_num_frames(
            num_frames=self.num_frames,
            duration_seconds=self.duration_seconds,
            fps=self.fps,
        )

    @property
    def total_chunks(self) -> int:
        return latent_chunk_count(self.resolved_num_frames)

    @property
    def latent_chunk_shape(self) -> tuple[int, ...]:
        return expected_latent_chunk_shape(self.height, self.width)

    def validate_server_limits(self, max_pixels: int) -> None:
        if max_pixels > 0 and self.height * self.width > max_pixels:
            raise ValueError(
                f"requested area {self.height * self.width} exceeds "
                f"server max_pixels={max_pixels}"
            )

    def warnings(self) -> list[str]:
        warnings = []
        if self.resolved_num_frames > DEFAULT_NUM_FRAMES:
            warnings.append(
                "SFWan2.1 code supports this frame count, but checkpoint quality "
                "beyond 81 frames is not validated"
            )
        if (self.height, self.width) not in OFFICIAL_RESOLUTIONS:
            warnings.append(
                "resolution is shape-compatible but outside the checkpoint's "
                "official 480x832/832x480 resolutions"
            )
        return warnings


class DitProfileRequest(GenerationRequest):
    """One independently queued full-DiT profile iteration."""

    profile_warmup: bool = False
    iteration: int = Field(default=0, ge=0)


class LatentJobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_frames: int = DEFAULT_NUM_FRAMES
    fps: int = Field(default=DEFAULT_FPS, gt=0)
    total_chunks: int | None = Field(default=None, gt=0)
    source: Literal["dit", "profile"] = "dit"
    transport: LatentTransport = "http"
    discard_output: bool = False
    profile_warmup: bool = False

    @model_validator(mode="after")
    def validate_shape(self) -> "LatentJobSpec":
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError("height and width must both be divisible by 16")
        frames = resolve_video_num_frames(
            num_frames=self.num_frames,
            duration_seconds=None,
            fps=self.fps,
        )
        expected_chunks = latent_chunk_count(frames)
        if self.total_chunks is not None and self.total_chunks != expected_chunks:
            raise ValueError(
                f"total_chunks={self.total_chunks} does not match "
                f"num_frames={frames} (expected {expected_chunks})"
            )
        self.total_chunks = expected_chunks
        return self

    @property
    def latent_chunk_shape(self) -> tuple[int, ...]:
        return expected_latent_chunk_shape(self.height, self.width)

    @classmethod
    def from_generation(
        cls,
        request_id: str,
        request: GenerationRequest,
        *,
        transport: LatentTransport = "http",
    ) -> "LatentJobSpec":
        return cls(
            request_id=request_id,
            height=request.height,
            width=request.width,
            num_frames=request.resolved_num_frames,
            fps=request.fps,
            total_chunks=request.total_chunks,
            source="dit",
            transport=transport,
        )


class SharedMemoryDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    lease_token: str = Field(min_length=16)
    total_bytes: int = Field(gt=0)
    data_offset: int = Field(ge=0)
    chunk_stride: int = Field(gt=0)
    tensor_nbytes: int = Field(gt=0)
    shape: tuple[int, int, int, int, int]
    dtype: Literal["bfloat16"] = "bfloat16"
    total_chunks: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_layout(self) -> "SharedMemoryDescriptor":
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("shared-memory tensor dimensions must be positive")
        if self.shape[:3] != (
            1,
            LATENT_CHANNELS,
            LATENT_FRAMES_PER_CHUNK,
        ):
            raise ValueError("shared-memory tensor must use SFWan BF16 BCTHW chunks")
        expected_nbytes = math.prod(self.shape) * 2
        if self.tensor_nbytes != expected_nbytes:
            raise ValueError(
                "shared-memory tensor_nbytes does not match shape and BF16 dtype"
            )
        if self.chunk_stride < self.tensor_nbytes:
            raise ValueError("shared-memory chunk stride is smaller than tensor bytes")
        required = self.data_offset + self.chunk_stride * self.total_chunks
        if required > self.total_bytes:
            raise ValueError("shared-memory descriptor exceeds its allocation")
        return self


class SharedMemoryChunkReady(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_token: str = Field(min_length=16)
    digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]+$")


class JobState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LatentJobRegistrationResponse(BaseModel):
    request_id: str
    created: bool
    status: JobState
    transport: LatentTransport
    shm: SharedMemoryDescriptor | None = None


TERMINAL_JOB_STATES = {
    JobState.COMPLETED,
    JobState.FAILED,
    JobState.CANCELLED,
}


class JobEvent(BaseModel):
    sequence: int
    kind: str
    unix_time_ns: int = Field(default_factory=time.time_ns)
    details: dict[str, Any] = Field(default_factory=dict)


class JobStatus(BaseModel):
    request_id: str
    role: Literal["monolithic", "dit", "vae"]
    state: JobState
    queue_sequence: int
    created_unix_time_ns: int
    started_unix_time_ns: int | None = None
    completed_unix_time_ns: int | None = None
    error: str | None = None
    output_available: bool = False
    events: list[JobEvent] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class SubmissionResponse(BaseModel):
    request_id: str
    status: JobState
    status_url: str
    result_url: str
    vae_status_url: str | None = None
    vae_result_url: str | None = None
    warnings: list[str] = Field(default_factory=list)


class EngineStatus(BaseModel):
    role: Literal["monolithic", "dit", "vae"]
    model_loaded: bool = False
    waiting_count: int
    waiting_ids: list[str]
    running_ids: list[str]
    running_capacity: int = 1
    contract: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    role: Literal["monolithic", "dit", "vae"]
    model_loaded: bool
    model_path: str
    model_load_ms: float | None = None


@dataclass(frozen=True, slots=True)
class SafetensorsLatentPayload:
    """Validated safetensors body with a zero-copy view of its tensor data."""

    payload: bytes
    shape: tuple[int, ...]
    dtype: str
    data_offset: int
    data_nbytes: int

    def data_view(self) -> memoryview:
        """Return a view, never a bytes slice, so latent data is not copied."""

        return memoryview(self.payload)[
            self.data_offset : self.data_offset + self.data_nbytes
        ]


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"safetensors header contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_latent_safetensors_payload(
    payload: bytes,
    *,
    expected_shape: tuple[int, ...],
) -> SafetensorsLatentPayload:
    """Validate one BF16/BCTHW safetensors body without loading tensor data."""

    if not isinstance(payload, bytes):
        raise TypeError("safetensors payload must be bytes")
    if sys.byteorder != "little":
        raise ValueError("safetensors latent ingress requires a little-endian host")
    if len(expected_shape) != 5 or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in expected_shape
    ):
        raise ValueError(
            "expected latent shape must be positive five-dimensional BCTHW"
        )
    if len(payload) < 8:
        raise ValueError("payload is too short to contain a safetensors header")

    header_nbytes = int.from_bytes(memoryview(payload)[:8], byteorder="little")
    if header_nbytes <= 0:
        raise ValueError("safetensors header length must be positive")
    if header_nbytes > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(
            f"safetensors header exceeds {MAX_SAFETENSORS_HEADER_BYTES} bytes"
        )
    data_offset = 8 + header_nbytes
    if data_offset > len(payload):
        raise ValueError("safetensors header extends beyond the payload")

    try:
        header_text = memoryview(payload)[8:data_offset].tobytes().decode("utf-8")
        header = json.loads(
            header_text,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("payload has an invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")

    if "__metadata__" in header:
        metadata = header["__metadata__"]
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise ValueError("safetensors __metadata__ must map strings to strings")
    tensor_names = set(header) - {"__metadata__"}
    if tensor_names != {"latents"}:
        raise ValueError("payload must contain exactly one tensor named 'latents'")

    tensor_header = header["latents"]
    if not isinstance(tensor_header, dict) or set(tensor_header) != {
        "dtype",
        "shape",
        "data_offsets",
    }:
        raise ValueError(
            "latents header must contain dtype, shape, and data_offsets only"
        )
    dtype = tensor_header["dtype"]
    if dtype != SAFETENSORS_WIRE_DTYPE:
        raise ValueError(f"latent dtype must be BF16, got {dtype!r}")

    shape = tensor_header["shape"]
    if not isinstance(shape, list) or any(
        isinstance(dimension, bool) or not isinstance(dimension, int)
        for dimension in shape
    ):
        raise ValueError("latent shape must be a list of integers")
    parsed_shape = tuple(shape)
    if parsed_shape != tuple(expected_shape):
        raise ValueError(
            f"latent shape {parsed_shape} does not match expected {expected_shape}"
        )

    data_offsets = tensor_header["data_offsets"]
    if (
        not isinstance(data_offsets, list)
        or len(data_offsets) != 2
        or any(
            isinstance(offset, bool) or not isinstance(offset, int)
            for offset in data_offsets
        )
    ):
        raise ValueError("latent data_offsets must contain two integers")
    expected_data_nbytes = math.prod(expected_shape) * 2
    if data_offsets != [0, expected_data_nbytes]:
        raise ValueError(
            "latent data_offsets must cover exactly one contiguous BF16 tensor"
        )
    if data_offset + expected_data_nbytes != len(payload):
        raise ValueError(
            "safetensors tensor data length does not match the payload length"
        )

    return SafetensorsLatentPayload(
        payload=payload,
        shape=parsed_shape,
        dtype=dtype,
        data_offset=data_offset,
        data_nbytes=expected_data_nbytes,
    )


def materialize_latent_tensor_for_cpu(
    parsed: SafetensorsLatentPayload,
) -> Any:
    """Materialize a CPU tensor for fake/no-CUDA execution only."""

    import torch

    storage = bytearray(parsed.data_view())
    return torch.frombuffer(storage, dtype=torch.bfloat16).view(parsed.shape)


def serialize_latent_tensor(tensor: Any) -> bytes:
    """Serialize one normalized BF16 BCTHW latent without pickle."""

    import torch
    from safetensors.torch import save

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("latent must be a torch.Tensor")
    if tensor.ndim != 5:
        raise ValueError("latent must use five-dimensional BCTHW layout")
    tensor = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return save({"latents": tensor})


def deserialize_latent_tensor(
    payload: bytes,
    *,
    expected_shape: tuple[int, ...],
) -> Any:
    """Load and validate one safetensors latent payload."""

    import torch
    from safetensors.torch import load

    try:
        tensors = load(payload)
    except Exception as exc:
        raise ValueError("payload is not a valid safetensors document") from exc
    if set(tensors) != {"latents"}:
        raise ValueError("payload must contain exactly one tensor named 'latents'")
    tensor = tensors["latents"]
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(
            f"latent shape {tuple(tensor.shape)} does not match "
            f"expected {tuple(expected_shape)}"
        )
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"latent dtype must be bfloat16, got {tensor.dtype}")
    return tensor.contiguous()


def payload_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
