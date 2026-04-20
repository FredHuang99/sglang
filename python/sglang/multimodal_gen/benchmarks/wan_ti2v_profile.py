# SPDX-License-Identifier: Apache-2.0
"""HTTP benchmark and summary tooling for diffusion profiling presets."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal

from sglang.multimodal_gen.configs.sample.wan import (
    Wan2_2_TI2V_5B_SamplingParam,
    WanT2V_1_3B_SamplingParams,
)
from sglang.multimodal_gen.configs.sample.zimage import (
    ZImageSamplingParams,
    ZImageTurboSamplingParams,
)
from sglang.multimodal_gen.runtime.utils.request_profiling import (
    CsvProfileWriter,
    build_summary_lines,
    resolve_profile_dir,
    resolve_run_id,
    summarize_throughput,
)

PROFILE_PRESET_CHOICES = [
    "auto",
    "wan2_2_ti2v_5b",
    "wan2_1_t2v_1_3b",
    "z_image",
]
_TERMINAL_STATUSES = {"completed", "failed", "deleted"}


@dataclass(frozen=True)
class RequestSpec:
    endpoint_kind: Literal["video", "image"]
    payload: dict[str, Any]


@dataclass
class BenchmarkRequest:
    index: int
    scheduled_submit_time_s: float
    payload: dict[str, Any]


def _parse_json_like(text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        curl_match = re.search(r"-d\s+'(\{.*\})'", text, re.DOTALL)
        if curl_match:
            return json.loads(curl_match.group(1))
        raise ValueError(f"Failed to parse request config: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Request config must decode to a JSON object.")
    return loaded


def load_request_config(request_config: str | None) -> dict[str, Any] | None:
    if request_config is None:
        return None
    if os.path.exists(request_config):
        with open(request_config, "r", encoding="utf-8") as fp:
            return _parse_json_like(fp.read())
    return _parse_json_like(request_config)


def detect_profile_preset(
    profile_preset: str,
    *,
    model_override: str | None = None,
    request_model: str | None = None,
) -> str:
    if profile_preset != "auto":
        return profile_preset

    candidate = (model_override or request_model or "").lower()
    if "z-image" in candidate:
        return "z_image"
    if "wan2.1-t2v-1.3b" in candidate or "wan2_1_t2v_1_3b" in candidate:
        return "wan2_1_t2v_1_3b"
    return "wan2_2_ti2v_5b"


def _sample_size(sample: Any) -> str | None:
    width = getattr(sample, "width", None)
    height = getattr(sample, "height", None)
    if width is None or height is None:
        return None
    return f"{width}x{height}"


def _default_video_payload(
    *,
    model: str,
    prompt: str,
    sample: Any,
    input_reference: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "num_frames": sample.num_frames,
        "fps": sample.fps,
        "guidance_scale": sample.guidance_scale,
        "num_inference_steps": sample.num_inference_steps,
        "seed": 1024,
        "generator_device": "cuda",
    }
    size = _sample_size(sample)
    if size is not None:
        payload["size"] = size
    negative_prompt = getattr(sample, "negative_prompt", None)
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt
    if input_reference is not None:
        payload["input_reference"] = input_reference
    return payload


def _default_zimage_payload(model: str) -> dict[str, Any]:
    sample = (
        ZImageTurboSamplingParams()
        if "turbo" in model.lower()
        else ZImageSamplingParams()
    )
    payload = {
        "model": model,
        "prompt": "A cozy bookstore interior, detailed illustration, warm cinematic lighting",
        "response_format": "url",
        "num_inference_steps": sample.num_inference_steps,
        "guidance_scale": sample.guidance_scale,
        "seed": 1024,
        "generator_device": "cuda",
    }
    negative_prompt = getattr(sample, "negative_prompt", None)
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt
    size = _sample_size(sample)
    if size is not None:
        payload["size"] = size
    return payload


def build_default_request_spec(
    profile_preset: str,
    *,
    model_override: str | None = None,
) -> RequestSpec:
    if profile_preset == "wan2_1_t2v_1_3b":
        sample = WanT2V_1_3B_SamplingParams()
        model = model_override or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        return RequestSpec(
            endpoint_kind="video",
            payload=_default_video_payload(
                model=model,
                prompt="A curious raccoon exploring a garden, cinematic lighting, natural motion",
                sample=sample,
            ),
        )

    if profile_preset == "z_image":
        model = model_override or "Tongyi-MAI/Z-Image-Turbo"
        return RequestSpec(endpoint_kind="image", payload=_default_zimage_payload(model))

    sample = Wan2_2_TI2V_5B_SamplingParam()
    model = model_override or "Wan2.2-TI2V-5B-Diffusers"
    return RequestSpec(
        endpoint_kind="video",
        payload=_default_video_payload(
            model=model,
            prompt="A person turns around slowly, natural motion, cinematic lighting, realistic details",
            sample=sample,
            input_reference="/sgl-workspace/sglang/examples/assets/example_image.png",
        ),
    )


def apply_request_overrides(
    base_config: dict[str, Any],
    *,
    endpoint_kind: Literal["video", "image"],
    model: str | None = None,
    prompt: str | None = None,
    input_reference: str | None = None,
    size: str | None = None,
    num_frames: int | None = None,
    fps: int | None = None,
    num_inference_steps: int | None = None,
    seed: int | None = None,
    guidance_scale: float | None = None,
) -> dict[str, Any]:
    payload = dict(base_config)

    common_overrides = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
    }
    for key, value in common_overrides.items():
        if value is not None:
            payload[key] = value

    if endpoint_kind == "video":
        video_overrides = {
            "input_reference": input_reference,
            "num_frames": num_frames,
            "fps": fps,
        }
        for key, value in video_overrides.items():
            if value is not None:
                payload[key] = value

    return payload


def _clamp_cfg_scale(value: Any, *, fallback: float = 1.0) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return float(value) if float(value) <= 1.0 else fallback
    return fallback


def disable_cfg_for_request(
    payload: dict[str, Any],
    *,
    endpoint_kind: Literal["video", "image"],
) -> dict[str, Any]:
    disabled_payload = dict(payload)
    disabled_payload["guidance_scale"] = _clamp_cfg_scale(
        disabled_payload.get("guidance_scale")
    )
    disabled_payload["negative_prompt"] = None

    if endpoint_kind == "video" and "guidance_scale_2" in disabled_payload:
        disabled_payload["guidance_scale_2"] = _clamp_cfg_scale(
            disabled_payload.get("guidance_scale_2")
        )

    if "true_cfg_scale" in disabled_payload:
        disabled_payload["true_cfg_scale"] = _clamp_cfg_scale(
            disabled_payload.get("true_cfg_scale")
        )

    if "cfg_normalization" in disabled_payload:
        disabled_payload["cfg_normalization"] = 0.0

    return disabled_payload


def build_submission_schedule(
    *,
    num_requests: int,
    traffic_mode: str,
    requests_per_minute: float,
    start_time_s: float | None = None,
) -> list[float]:
    start = time.time() if start_time_s is None else start_time_s
    if num_requests <= 0:
        return []
    if traffic_mode == "burst":
        return [start] * num_requests
    if requests_per_minute <= 0:
        raise ValueError("requests_per_minute must be positive for normal traffic.")
    interval_s = 60.0 / requests_per_minute
    return [start + (index * interval_s) for index in range(num_requests)]


def build_benchmark_requests(
    *,
    payload: dict[str, Any],
    num_requests: int,
    traffic_mode: str,
    requests_per_minute: float,
    start_time_s: float | None = None,
) -> list[BenchmarkRequest]:
    schedule = build_submission_schedule(
        num_requests=num_requests,
        traffic_mode=traffic_mode,
        requests_per_minute=requests_per_minute,
        start_time_s=start_time_s,
    )
    return [
        BenchmarkRequest(
            index=index,
            scheduled_submit_time_s=scheduled_time_s,
            payload=dict(payload),
        )
        for index, scheduled_time_s in enumerate(schedule)
    ]


def build_warmup_requests(
    *,
    payload: dict[str, Any],
    num_warmup_requests: int,
    start_time_s: float | None = None,
) -> list[BenchmarkRequest]:
    if num_warmup_requests <= 0:
        return []
    start = time.time() if start_time_s is None else start_time_s
    return [
        BenchmarkRequest(
            index=-(index + 1),
            scheduled_submit_time_s=start,
            payload=dict(payload),
        )
        for index in range(num_warmup_requests)
    ]


async def _submit_and_poll_video_request(
    *,
    session,
    base_url: str,
    request: BenchmarkRequest,
    poll_interval_s: float,
) -> dict[str, Any]:
    now_s = time.time()
    sleep_s = request.scheduled_submit_time_s - now_s
    if sleep_s > 0:
        await asyncio.sleep(sleep_s)

    submit_time_s = time.time()
    async with session.post(f"{base_url}/v1/videos", json=request.payload) as response:
        response.raise_for_status()
        created = await response.json()
    http_ack_time_s = time.time()

    request_id = created.get("id")
    if not request_id:
        raise RuntimeError(f"Video creation response missing id: {created}")

    terminal_payload = created
    while terminal_payload.get("status") not in _TERMINAL_STATUSES:
        await asyncio.sleep(poll_interval_s)
        async with session.get(f"{base_url}/v1/videos/{request_id}") as response:
            response.raise_for_status()
            terminal_payload = await response.json()
    terminal_time_s = time.time()

    error_value = terminal_payload.get("error")
    if isinstance(error_value, dict):
        error_value = error_value.get("message") or json.dumps(error_value)

    return {
        "client_request_index": request.index,
        "request_id": request_id,
        "scheduled_submit_time_s": request.scheduled_submit_time_s,
        "submit_time_s": submit_time_s,
        "http_ack_time_s": http_ack_time_s,
        "terminal_time_s": terminal_time_s,
        "status": terminal_payload.get("status"),
        "error": error_value,
        "url": terminal_payload.get("url"),
        "file_path": terminal_payload.get("file_path"),
    }


async def _submit_image_request(
    *,
    session,
    base_url: str,
    request: BenchmarkRequest,
) -> dict[str, Any]:
    now_s = time.time()
    sleep_s = request.scheduled_submit_time_s - now_s
    if sleep_s > 0:
        await asyncio.sleep(sleep_s)

    submit_time_s = time.time()
    async with session.post(
        f"{base_url}/v1/images/generations", json=request.payload
    ) as response:
        response.raise_for_status()
        result = await response.json()
    terminal_time_s = time.time()

    request_id = result.get("id")
    if not request_id:
        raise RuntimeError(f"Image generation response missing id: {result}")

    data = result.get("data") or []
    first_item = data[0] if data else {}
    return {
        "client_request_index": request.index,
        "request_id": request_id,
        "scheduled_submit_time_s": request.scheduled_submit_time_s,
        "submit_time_s": submit_time_s,
        "http_ack_time_s": terminal_time_s,
        "terminal_time_s": terminal_time_s,
        "status": "completed",
        "error": None,
        "url": first_item.get("url"),
        "file_path": first_item.get("file_path"),
    }


async def run_benchmark_async(
    *,
    base_url: str,
    endpoint_kind: Literal["video", "image"],
    payload: dict[str, Any],
    num_requests: int,
    traffic_mode: str,
    requests_per_minute: float,
    poll_interval_s: float,
) -> list[dict[str, Any]]:
    import aiohttp

    benchmark_requests = build_benchmark_requests(
        payload=payload,
        num_requests=num_requests,
        traffic_mode=traffic_mode,
        requests_per_minute=requests_per_minute,
    )
    connector = aiohttp.TCPConnector(limit=max(1, num_requests))
    async with aiohttp.ClientSession(connector=connector) as session:
        if endpoint_kind == "image":
            tasks = [
                asyncio.create_task(
                    _submit_image_request(
                        session=session,
                        base_url=base_url,
                        request=request,
                    )
                )
                for request in benchmark_requests
            ]
        else:
            tasks = [
                asyncio.create_task(
                    _submit_and_poll_video_request(
                        session=session,
                        base_url=base_url,
                        request=request,
                        poll_interval_s=poll_interval_s,
                    )
                )
                for request in benchmark_requests
            ]
        return await asyncio.gather(*tasks)


async def run_warmup_async(
    *,
    base_url: str,
    endpoint_kind: Literal["video", "image"],
    payload: dict[str, Any],
    num_warmup_requests: int,
    poll_interval_s: float,
) -> list[dict[str, Any]]:
    if num_warmup_requests <= 0:
        return []

    import aiohttp

    rows: list[dict[str, Any]] = []
    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        warmup_requests = build_warmup_requests(
            payload=payload,
            num_warmup_requests=num_warmup_requests,
        )
        for index, request in enumerate(warmup_requests):
            if endpoint_kind == "image":
                row = await _submit_image_request(
                    session=session,
                    base_url=base_url,
                    request=request,
                )
            else:
                row = await _submit_and_poll_video_request(
                    session=session,
                    base_url=base_url,
                    request=request,
                    poll_interval_s=poll_interval_s,
                )
            if row.get("status") != "completed":
                error = row.get("error") or row.get("status")
                raise RuntimeError(
                    f"Warmup request {index + 1}/{num_warmup_requests} failed: {error}"
                )
            rows.append(row)
    return rows


def read_csv_rows(file_path: str) -> list[dict[str, str]]:
    with open(file_path, "r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def maybe_copy_file(source_path: str, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, os.path.basename(source_path))
    if os.path.abspath(source_path) != os.path.abspath(target_path):
        shutil.copy2(source_path, target_path)
    return target_path


def _candidate_profile_dirs(
    output_dir: str,
    run_id: str,
    deployment_mode: str,
    traffic_mode: str,
) -> list[str]:
    output_dir = os.path.abspath(output_dir)
    parent_dir = os.path.dirname(output_dir)
    candidates = [output_dir]
    for candidate in [
        os.path.join(parent_dir, f"{run_id}_{deployment_mode}"),
        os.path.join(parent_dir, f"{run_id}_{deployment_mode}_{traffic_mode}"),
    ]:
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def find_existing_profile_file(
    *,
    output_dir: str,
    run_id: str,
    deployment_mode: str,
    traffic_mode: str,
    filename: str,
) -> str | None:
    for candidate_dir in _candidate_profile_dirs(
        output_dir=output_dir,
        run_id=run_id,
        deployment_mode=deployment_mode,
        traffic_mode=traffic_mode,
    ):
        candidate_path = os.path.join(candidate_dir, filename)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def summarize_monolithic_rows(rows: list[dict[str, str]]) -> tuple[list[float], dict[str, list[float]], float | None]:
    e2e_ms: list[float] = []
    stages = {"encoder": [], "denoiser": [], "decoder": []}
    arrivals: list[float] = []
    finishes: list[float] = []
    for row in rows:
        arrival = _parse_float(row.get("arrival_time_s"))
        finish = _parse_float(row.get("finish_time_s"))
        if arrival is not None and finish is not None and finish >= arrival:
            e2e_ms.append((finish - arrival) * 1000.0)
            arrivals.append(arrival)
            finishes.append(finish)
        for stage_name in stages:
            value = _parse_float(row.get(f"logical_{stage_name}_duration_ms"))
            if value is not None:
                stages[stage_name].append(value)
    throughput = summarize_throughput(
        len(e2e_ms),
        first_arrival_time_s=min(arrivals) if arrivals else None,
        last_finish_time_s=max(finishes) if finishes else None,
    )
    return e2e_ms, stages, throughput


def summarize_disagg_rows(
    server_rows: list[dict[str, str]],
    encoder_rows: list[dict[str, str]],
    denoiser_rows: list[dict[str, str]],
    decoder_rows: list[dict[str, str]],
) -> tuple[list[float], dict[str, list[float]], float | None]:
    e2e_ms: list[float] = []
    arrivals: list[float] = []
    finishes: list[float] = []
    for row in server_rows:
        arrival = _parse_float(row.get("request_arrival_time_s"))
        finish = _parse_float(row.get("finish_time_s"))
        if arrival is not None and finish is not None and finish >= arrival:
            e2e_ms.append((finish - arrival) * 1000.0)
            arrivals.append(arrival)
            finishes.append(finish)

    def collect_role_duration(role_rows: list[dict[str, str]]) -> list[float]:
        values: list[float] = []
        for row in role_rows:
            duration = _parse_float(row.get("compute_duration_ms"))
            if duration is not None:
                values.append(duration)
        return values

    stages = {
        "encoder": collect_role_duration(encoder_rows),
        "denoiser": collect_role_duration(denoiser_rows),
        "decoder": collect_role_duration(decoder_rows),
    }
    throughput = summarize_throughput(
        len(e2e_ms),
        first_arrival_time_s=min(arrivals) if arrivals else None,
        last_finish_time_s=max(finishes) if finishes else None,
    )
    return e2e_ms, stages, throughput


def summarize_profile_run(
    *,
    output_dir: str,
    run_id: str,
    deployment_mode: str,
    traffic_mode: str,
    client_rows: list[dict[str, str]],
) -> list[str]:
    request_ids = {
        row["request_id"]
        for row in client_rows
        if row.get("request_id") and row.get("status") in _TERMINAL_STATUSES
    }
    if not request_ids:
        return build_summary_lines(
            e2e_latency_ms=[],
            logical_stage_duration_ms={"encoder": [], "denoiser": [], "decoder": []},
            throughput_rps=None,
        )

    if deployment_mode == "monolithic":
        monolithic_file = find_existing_profile_file(
            output_dir=output_dir,
            run_id=run_id,
            deployment_mode=deployment_mode,
            traffic_mode=traffic_mode,
            filename="monolithic_server.csv",
        )
        if monolithic_file is None:
            raise FileNotFoundError("Failed to locate monolithic_server.csv for summary.")
        maybe_copy_file(monolithic_file, output_dir)
        rows = [row for row in read_csv_rows(monolithic_file) if row.get("request_id") in request_ids]
        e2e_ms, stages, throughput = summarize_monolithic_rows(rows)
    else:
        filenames = ["server.csv", "encoder.csv", "denoiser.csv", "decoder.csv"]
        located = {}
        for filename in filenames:
            file_path = find_existing_profile_file(
                output_dir=output_dir,
                run_id=run_id,
                deployment_mode=deployment_mode,
                traffic_mode=traffic_mode,
                filename=filename,
            )
            if file_path is None:
                raise FileNotFoundError(f"Failed to locate {filename} for summary.")
            located[filename] = file_path
            maybe_copy_file(file_path, output_dir)
        e2e_ms, stages, throughput = summarize_disagg_rows(
            [row for row in read_csv_rows(located["server.csv"]) if row.get("request_id") in request_ids],
            [row for row in read_csv_rows(located["encoder.csv"]) if row.get("request_id") in request_ids],
            [row for row in read_csv_rows(located["denoiser.csv"]) if row.get("request_id") in request_ids],
            [row for row in read_csv_rows(located["decoder.csv"]) if row.get("request_id") in request_ids],
        )

    return build_summary_lines(
        e2e_latency_ms=e2e_ms,
        logical_stage_duration_ms=stages,
        throughput_rps=throughput,
    )


def write_summary(output_dir: str, lines: list[str]) -> str:
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-mode", choices=["monolithic", "disaggregation"], required=True)
    parser.add_argument("--traffic-mode", choices=["burst", "normal"], required=True)
    parser.add_argument("--profile-preset", choices=PROFILE_PRESET_CHOICES, default="wan2_2_ti2v_5b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--num-warmup-requests", type=int, default=3)
    parser.add_argument("--requests-per-minute", type=float, default=24.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--request-config", default=None)
    parser.add_argument("--output-dir", default="/data/profile")
    parser.add_argument("--profile-run-id", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--input-reference", default=None)
    parser.add_argument("--size", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--disable-cfg", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_id = resolve_run_id(args.profile_run_id)
    output_dir = resolve_profile_dir(
        args.output_dir,
        run_id,
        args.deployment_mode,
        args.traffic_mode,
    )
    base_url = args.base_url or f"http://{args.host}:{args.port}"

    request_config = load_request_config(args.request_config)
    resolved_preset = detect_profile_preset(
        args.profile_preset,
        model_override=args.model,
        request_model=(request_config or {}).get("model"),
    )
    request_spec = build_default_request_spec(
        resolved_preset,
        model_override=args.model,
    )
    payload = apply_request_overrides(
        request_config or request_spec.payload,
        endpoint_kind=request_spec.endpoint_kind,
        model=args.model,
        prompt=args.prompt,
        input_reference=args.input_reference,
        size=args.size,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
    )
    if args.disable_cfg:
        payload = disable_cfg_for_request(
            payload,
            endpoint_kind=request_spec.endpoint_kind,
        )

    warmup_rows = asyncio.run(
        run_warmup_async(
            base_url=base_url,
            endpoint_kind=request_spec.endpoint_kind,
            payload=payload,
            num_warmup_requests=args.num_warmup_requests,
            poll_interval_s=args.poll_interval_s,
        )
    )

    rows = asyncio.run(
        run_benchmark_async(
            base_url=base_url,
            endpoint_kind=request_spec.endpoint_kind,
            payload=payload,
            num_requests=args.num_requests,
            traffic_mode=args.traffic_mode,
            requests_per_minute=args.requests_per_minute,
            poll_interval_s=args.poll_interval_s,
        )
    )

    client_csv_path = os.path.join(output_dir, "client.csv")
    writer = CsvProfileWriter(client_csv_path)
    for row in rows:
        writer.write_row(row)

    summary_lines = summarize_profile_run(
        output_dir=output_dir,
        run_id=run_id,
        deployment_mode=args.deployment_mode,
        traffic_mode=args.traffic_mode,
        client_rows=rows,
    )
    write_summary(output_dir, summary_lines)

    print(f"run_id={run_id}")
    print(f"output_dir={output_dir}")
    print(f"profile_preset={resolved_preset}")
    print(f"endpoint_kind={request_spec.endpoint_kind}")
    print(f"disable_cfg={args.disable_cfg}")
    print(f"num_warmup_requests={len(warmup_rows)}")
    for line in summary_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
