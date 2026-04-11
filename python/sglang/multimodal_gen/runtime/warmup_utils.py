# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for constructing warmup requests for diffusion schedulers."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


MINIMUM_PICTURE_BASE64_FOR_WARMUP = (
    "data:image/jpg;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)


def _warmup_image_path() -> str:
    uploads_dir = os.path.join("outputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    return asyncio.run(
        save_image_to_path(
            MINIMUM_PICTURE_BASE64_FOR_WARMUP,
            os.path.join(uploads_dir, "warmup_image.jpg"),
        )
    )


def build_server_warmup_reqs(server_args: "ServerArgs") -> list[Req]:
    if not server_args.warmup:
        return []

    resolutions = server_args.warmup_resolutions or [None]
    task_type = server_args.pipeline_config.task_type
    warmup_reqs: list[Req] = []

    needs_image = task_type in (
        ModelTaskType.I2I,
        ModelTaskType.TI2I,
        ModelTaskType.I2V,
        ModelTaskType.TI2V,
        ModelTaskType.I2M,
    )
    input_path = _warmup_image_path() if needs_image else None

    for idx, resolution in enumerate(resolutions, start=1):
        width = height = None
        if resolution is not None:
            width, height = _parse_size(resolution)

        sampling_kwargs = {
            "request_id": f"warmup-{idx}",
            "data_type": task_type.data_type(),
            "prompt": "warmup",
        }
        if width is not None:
            sampling_kwargs["width"] = width
        if height is not None:
            sampling_kwargs["height"] = height
        if input_path is not None:
            sampling_kwargs["negative_prompt"] = ""
            sampling_kwargs["image_path"] = [input_path]

        sampling_params = SamplingParams.from_user_sampling_params_args(
            model_path=server_args.model_path,
            server_args=server_args,
            **sampling_kwargs,
        )
        req = Req(sampling_params=sampling_params)
        req.set_as_warmup(server_args.warmup_steps)
        warmup_reqs.append(req)

    return warmup_reqs
