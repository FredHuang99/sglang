"""Async load generator and VAE-only profiler for the minimal SFWan service."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from .protocol import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
    LATENT_CHANNELS,
    DitProfileRequest,
    GenerationRequest,
    JobState,
    LatentJobSpec,
    SubmissionResponse,
    TERMINAL_JOB_STATES,
    latent_frame_count,
    serialize_latent_tensor,
)

ArrivalMode = Literal["burst", "fixed", "poisson"]
DEFAULT_PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, "
    "its eyes wide with interest. The playful yet serene atmosphere is "
    "complemented by soft natural light filtering through the petals. "
    "Mid-shot, warm and cheerful tones."
)


def build_interarrival_delays(
    *,
    count: int,
    mode: ArrivalMode,
    seed: int,
    fixed_interval_seconds: float = 0.0,
    poisson_lambda: float = 1.0,
) -> list[float]:
    """Return the delay before each request; the first request is immediate."""

    if count <= 0:
        raise ValueError("count must be positive")
    if fixed_interval_seconds < 0:
        raise ValueError("fixed interval must be non-negative")
    if poisson_lambda <= 0:
        raise ValueError("poisson lambda must be positive")

    rng = random.Random(seed)
    delays = [0.0]
    for _ in range(count - 1):
        if mode == "burst":
            delay = 0.0
        elif mode == "fixed":
            delay = fixed_interval_seconds
        elif mode == "poisson":
            delay = rng.expovariate(poisson_lambda)
        else:
            raise ValueError(f"unsupported arrival mode: {mode}")
        delays.append(delay)
    return delays


def _load_generation_requests(args: argparse.Namespace) -> list[GenerationRequest]:
    base = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "duration_seconds": args.duration_seconds,
        "fps": args.fps,
        "seed": args.seed,
    }
    if args.workload_jsonl is None:
        return [
            GenerationRequest.model_validate(base) for _ in range(args.num_requests)
        ]

    overrides = []
    path = Path(args.workload_jsonl)
    with path.open(encoding="utf-8") as workload_file:
        for line_number, line in enumerate(workload_file, start=1):
            if not line.strip():
                continue
            try:
                override = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(override, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            overrides.append(override)
    if not overrides:
        raise ValueError("workload JSONL did not contain any requests")
    if len(overrides) != args.num_requests:
        raise ValueError(
            f"--num-requests={args.num_requests} but workload JSONL contains "
            f"{len(overrides)} request overrides"
        )
    return [GenerationRequest.model_validate(base | override) for override in overrides]


async def _post_at_offset(
    *,
    client: Any,
    url: str,
    generation: GenerationRequest,
    offset_seconds: float,
    start_time: float,
) -> tuple[SubmissionResponse, dict[str, Any]]:
    wait_seconds = start_time + offset_seconds - time.perf_counter()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
    send_start = time.perf_counter()
    response = await client.post(
        f"{url.rstrip('/')}/v1/generations",
        json=generation.model_dump(mode="json"),
    )
    response.raise_for_status()
    submission = SubmissionResponse.model_validate(response.json())
    return submission, {
        "request_id": submission.request_id,
        "scheduled_offset_seconds": offset_seconds,
        "actual_send_offset_seconds": send_start - start_time,
        "submit_rtt_ms": (time.perf_counter() - send_start) * 1000,
    }


async def _wait_for_status(
    *,
    client: Any,
    status_url: str,
    poll_interval_seconds: float,
    allow_initial_not_found: bool,
) -> dict[str, Any]:
    while True:
        response = await client.get(status_url)
        if response.status_code == 404 and allow_initial_not_found:
            await asyncio.sleep(poll_interval_seconds)
            continue
        response.raise_for_status()
        body = response.json()
        state = JobState(body["state"])
        if state in TERMINAL_JOB_STATES:
            return body
        await asyncio.sleep(poll_interval_seconds)


async def _download_result(
    *,
    client: Any,
    result_url: str,
    output_dir: Path,
    request_id: str,
) -> str:
    response = await client.get(result_url)
    response.raise_for_status()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{request_id}.mp4"
    path.write_bytes(response.content)
    return str(path.resolve())


async def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    generations = _load_generation_requests(args)
    delays = build_interarrival_delays(
        count=len(generations),
        mode=args.arrival_mode,
        seed=args.arrival_seed,
        fixed_interval_seconds=args.fixed_interval_seconds,
        poisson_lambda=args.poisson_lambda,
    )
    offsets = []
    cumulative = 0.0
    for delay in delays:
        cumulative += delay
        offsets.append(cumulative)

    timeout = httpx.Timeout(args.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        start_time = time.perf_counter()
        tasks = [
            asyncio.create_task(
                _post_at_offset(
                    client=client,
                    url=args.server_url,
                    generation=generation,
                    offset_seconds=offset,
                    start_time=start_time,
                )
            )
            for generation, offset in zip(generations, offsets, strict=True)
        ]
        submissions = await asyncio.gather(*tasks)

        async def _finish_one(
            submission: SubmissionResponse,
            submit_metrics: dict[str, Any],
        ) -> dict[str, Any]:
            vae_task = None
            if submission.vae_status_url is not None:
                vae_task = asyncio.create_task(
                    _wait_for_status(
                        client=client,
                        status_url=submission.vae_status_url,
                        poll_interval_seconds=args.poll_interval_seconds,
                        allow_initial_not_found=True,
                    )
                )
            try:
                coordinator_status = await _wait_for_status(
                    client=client,
                    status_url=submission.status_url,
                    poll_interval_seconds=args.poll_interval_seconds,
                    allow_initial_not_found=False,
                )
            except BaseException:
                if vae_task is not None:
                    vae_task.cancel()
                    try:
                        await vae_task
                    except asyncio.CancelledError:
                        pass
                raise
            vae_status = None
            if vae_task is not None:
                if (
                    JobState(coordinator_status["state"]) == JobState.COMPLETED
                    or vae_task.done()
                ):
                    vae_status = await vae_task
                else:
                    vae_task.cancel()
                    try:
                        await vae_task
                    except asyncio.CancelledError:
                        pass
                    assert submission.vae_status_url is not None
                    response = await client.get(submission.vae_status_url)
                    if response.status_code == 200:
                        vae_status = response.json()

            final_status = vae_status or coordinator_status
            output_path = None
            if (
                JobState(final_status["state"]) == JobState.COMPLETED
                and not args.skip_download
            ):
                result_url = submission.vae_result_url or submission.result_url
                output_path = await _download_result(
                    client=client,
                    result_url=result_url,
                    output_dir=Path(args.output_dir),
                    request_id=submission.request_id,
                )
            return {
                **submit_metrics,
                "warnings": submission.warnings,
                "coordinator": coordinator_status,
                "vae": vae_status,
                "output_path": output_path,
            }

        completed = await asyncio.gather(
            *[
                _finish_one(submission, submit_metrics)
                for submission, submit_metrics in submissions
            ]
        )
    return {
        "mode": "generate",
        "arrival_mode": args.arrival_mode,
        "request_count": len(completed),
        "requests": completed,
    }


def _make_dummy_latents(*, spec: LatentJobSpec, seed: int) -> list[Any]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    full = torch.randn(
        (
            1,
            LATENT_CHANNELS,
            latent_frame_count(spec.num_frames),
            spec.height // 8,
            spec.width // 8,
        ),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(torch.bfloat16)
    return [
        full[:, :, index * 3 : (index + 1) * 3].contiguous()
        for index in range(int(spec.total_chunks))
    ]


async def _run_profile_iteration(
    *,
    client: Any,
    server_url: str,
    args: argparse.Namespace,
    warmup: bool,
    iteration: int,
) -> dict[str, Any]:
    request_id = f"profile-{uuid.uuid4()}"
    spec = LatentJobSpec(
        request_id=request_id,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        source="profile",
        discard_output=not args.save_output,
        profile_warmup=warmup,
    )
    response = await client.post(
        f"{server_url.rstrip('/')}/v1/latent-jobs",
        json=spec.model_dump(mode="json"),
    )
    response.raise_for_status()

    for chunk_index, tensor in enumerate(
        _make_dummy_latents(spec=spec, seed=args.seed)
    ):
        payload = await asyncio.to_thread(serialize_latent_tensor, tensor)
        upload = await client.put(
            (
                f"{server_url.rstrip('/')}/v1/latent-jobs/"
                f"{request_id}/chunks/{chunk_index}"
            ),
            content=payload,
            headers={"content-type": "application/x-safetensors"},
        )
        upload.raise_for_status()

    status_url = f"{server_url.rstrip('/')}/v1/jobs/{request_id}"
    final_status = await _wait_for_status(
        client=client,
        status_url=status_url,
        poll_interval_seconds=args.poll_interval_seconds,
        allow_initial_not_found=False,
    )
    output_path = None
    if args.save_output and JobState(final_status["state"]) == JobState.COMPLETED:
        output_path = await _download_result(
            client=client,
            result_url=f"{status_url}/result",
            output_dir=Path(args.output_dir),
            request_id=request_id,
        )
    profile_execution = final_status.get("metrics", {}).get("profile_execution")
    if (
        JobState(final_status["state"]) == JobState.COMPLETED
        and profile_execution is None
    ):
        raise RuntimeError(
            "VAE profile completed without profile_execution metrics; "
            "start the VAE server with --enable-profile"
        )
    result = {
        "iteration": iteration,
        "warmup": warmup,
        "request_id": request_id,
        "state": final_status["state"],
        "error": final_status.get("error"),
        "profile_execution": profile_execution,
    }
    if output_path is not None:
        # Kept private to this module so measured/all_iterations remain a
        # model-execution-only schema. run_profile_vae exposes saved artifacts
        # separately from measurements.
        result["_saved_output_path"] = output_path
    return result


def _profile_measurement_view(iteration: dict[str, Any]) -> dict[str, Any]:
    """Keep profile measurements independent of transfer/output diagnostics."""

    keys = (
        "iteration",
        "warmup",
        "request_id",
        "state",
        "error",
        "profile_execution",
    )
    return {key: iteration.get(key) for key in keys}


async def run_profile_vae(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    timeout = httpx.Timeout(args.request_timeout_seconds)
    iterations = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        engine_response = await client.get(f"{args.server_url.rstrip('/')}/v1/engine")
        engine_response.raise_for_status()
        engine_status = engine_response.json()
        engine_contract = engine_status.get("contract", {})
        if not isinstance(engine_contract, dict):
            raise RuntimeError("VAE server returned an invalid engine contract")
        layer_profile_enabled = bool(
            engine_contract.get("trt_layer_profile_enabled", False)
        )
        layer_profile_path = getattr(args, "trt_layer_profile_json", None)
        summary_path = getattr(args, "summary_json", None)
        if (
            layer_profile_path is not None
            and summary_path is not None
            and Path(layer_profile_path).expanduser().resolve()
            == Path(summary_path).expanduser().resolve()
        ):
            raise ValueError(
                "--trt-layer-profile-json and --summary-json must use different files"
            )
        if layer_profile_enabled and layer_profile_path is None:
            raise ValueError(
                "the VAE server has TensorRT layer profiling enabled; "
                "provide --trt-layer-profile-json"
            )
        if layer_profile_path is not None and not layer_profile_enabled:
            raise ValueError(
                "--trt-layer-profile-json requires a VAE server started with "
                "--enable-trt-layer-profile"
            )
        for index in range(args.warmup + args.repeat):
            iterations.append(
                await _run_profile_iteration(
                    client=client,
                    server_url=args.server_url,
                    args=args,
                    warmup=index < args.warmup,
                    iteration=index,
                )
            )
    measurements = [_profile_measurement_view(item) for item in iterations]
    summary = {
        "mode": "profile-vae",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "measured": [
            iteration for iteration in measurements if not iteration["warmup"]
        ],
        "all_iterations": measurements,
        "saved_outputs": [
            {
                "iteration": item["iteration"],
                "request_id": item["request_id"],
                "path": item["_saved_output_path"],
            }
            for item in iterations
            if "_saved_output_path" in item
        ],
    }
    if layer_profile_enabled:
        from .vae_trt_profile import (
            TRT_LAYER_PROFILE_SCHEMA_VERSION,
            aggregate_trt_layer_profile_iterations,
            write_trt_layer_profile_artifact,
        )

        metadata_values = []
        for measurement in measurements:
            execution = measurement.get("profile_execution")
            metadata = (
                execution.pop("trt_layer_profile_metadata", None)
                if isinstance(execution, dict)
                else None
            )
            if not isinstance(metadata, dict):
                raise RuntimeError(
                    "TensorRT layer-profile result has no catalog metadata"
                )
            metadata_values.append(metadata)
        metadata = metadata_values[0]
        if any(value != metadata for value in metadata_values[1:]):
            diagnostic = {
                "schema_version": TRT_LAYER_PROFILE_SCHEMA_VERSION,
                "validation": {
                    "valid_for_optimization_decision": False,
                    "catalog_stable": False,
                    "warnings": [
                        "TensorRT layer-profile metadata drifted across iterations"
                    ],
                },
                "iterations": measurements,
                "metadata_by_iteration": metadata_values,
            }
            reference = write_trt_layer_profile_artifact(
                path=layer_profile_path,
                detailed=diagnostic,
            )
            raise RuntimeError(
                "TensorRT layer-profile metadata drifted across iterations; "
                f"diagnostic evidence was written to {reference['path']}"
            )
        try:
            aggregated = aggregate_trt_layer_profile_iterations(
                measurements,
                catalogs=metadata["catalogs"],
                environment=metadata.get("environment"),
                precision=metadata["precision"],
                plan_sha256=metadata["plan_sha256"],
                qdq_schema_version=metadata.get("qdq_schema_version"),
                weight_encoding=metadata.get("weight_encoding"),
            )
        except BaseException as exc:
            diagnostic = {
                "schema_version": TRT_LAYER_PROFILE_SCHEMA_VERSION,
                "validation": {
                    "valid_for_optimization_decision": False,
                    "warnings": [f"{type(exc).__name__}: {exc}"],
                },
                "environment": metadata.get("environment", {}),
                "metadata": metadata,
                "iterations": measurements,
            }
            reference = write_trt_layer_profile_artifact(
                path=layer_profile_path,
                detailed=diagnostic,
            )
            raise RuntimeError(
                "TensorRT layer-profile aggregation failed; diagnostic evidence "
                f"was written to {reference['path']}"
            ) from exc

        reference = write_trt_layer_profile_artifact(
            path=layer_profile_path,
            detailed=aggregated["detailed"],
        )
        layer_summary = dict(aggregated["summary"])
        layer_summary["detailed_artifact"] = reference
        summary["trt_layer_profile_summary"] = layer_summary

        # The normal summary retains model-execution metrics and the aggregate,
        # but the per-physical-layer vectors live only in the detailed artifact.
        for measurement in measurements:
            execution = measurement.get("profile_execution")
            chunks = execution.get("chunks", []) if isinstance(execution, dict) else []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    chunk.pop("trt_layer_profile", None)

        if not layer_summary["valid_for_optimization_decision"]:
            raise RuntimeError(
                "TensorRT layer-profile evidence is not valid for an optimization "
                f"decision; inspect {reference['path']}"
            )
    return summary


async def _run_dit_profile_iteration(
    *,
    client: Any,
    server_url: str,
    args: argparse.Namespace,
    warmup: bool,
    iteration: int,
) -> dict[str, Any]:
    request = DitProfileRequest(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        seed=args.seed,
        profile_warmup=warmup,
        iteration=iteration,
    )
    response = await client.post(
        f"{server_url.rstrip('/')}/v1/dit-profiles",
        json=request.model_dump(mode="json"),
    )
    response.raise_for_status()
    submission = SubmissionResponse.model_validate(response.json())
    final_status = await _wait_for_status(
        client=client,
        status_url=submission.status_url,
        poll_interval_seconds=args.poll_interval_seconds,
        allow_initial_not_found=False,
    )
    profile_execution = final_status.get("metrics", {}).get("profile_execution")
    if (
        JobState(final_status["state"]) == JobState.COMPLETED
        and profile_execution is None
    ):
        raise RuntimeError(
            "DiT profile completed without profile_execution metrics; "
            "start the DiT server with --enable-profile"
        )
    return {
        "iteration": iteration,
        "warmup": warmup,
        "request_id": submission.request_id,
        "state": final_status["state"],
        "error": final_status.get("error"),
        "profile_execution": profile_execution,
    }


async def run_profile_dit(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    timeout = httpx.Timeout(args.request_timeout_seconds)
    iterations = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index in range(args.warmup + args.repeat):
            iterations.append(
                await _run_dit_profile_iteration(
                    client=client,
                    server_url=args.server_url,
                    args=args,
                    warmup=index < args.warmup,
                    iteration=index,
                )
            )
    return {
        "mode": "profile-dit",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "measured": [iteration for iteration in iterations if not iteration["warmup"]],
        "all_iterations": iterations,
    }


def _write_summary(summary: dict[str, Any], path: str | None) -> None:
    serialized = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def _add_common_transport_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.5)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--output-dir", default="sfwan_client_outputs")
    parser.add_argument("--summary-json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="send text generation load")
    _add_common_transport_args(generate)
    generate.add_argument("--num-requests", type=int, default=1)
    generate.add_argument("--workload-jsonl")
    generate.add_argument("--prompt", default=DEFAULT_PROMPT)
    generate.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    generate.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    generate.add_argument("--num-frames", type=int)
    generate.add_argument("--duration-seconds", type=float)
    generate.add_argument("--fps", type=int, default=DEFAULT_FPS)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument(
        "--arrival-mode",
        choices=("burst", "fixed", "poisson"),
        default="burst",
    )
    generate.add_argument("--arrival-seed", type=int, default=0)
    generate.add_argument("--fixed-interval-seconds", type=float, default=1.0)
    generate.add_argument("--poisson-lambda", type=float, default=1.0)
    generate.add_argument("--skip-download", action="store_true")

    profile = subparsers.add_parser(
        "profile-vae",
        help="send deterministic dummy latents directly to a VAE server",
    )
    _add_common_transport_args(profile)
    profile.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    profile.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    profile.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    profile.add_argument("--fps", type=int, default=DEFAULT_FPS)
    profile.add_argument("--seed", type=int, default=DEFAULT_SEED)
    profile.add_argument("--warmup", type=int, default=0)
    profile.add_argument("--repeat", type=int, default=1)
    profile.add_argument(
        "--trt-layer-profile-json",
        help=(
            "separate detailed TensorRT physical-layer artifact; required "
            "when the VAE server enables fine-grained layer profiling"
        ),
    )
    profile.add_argument(
        "--save-video",
        "--save-output",
        dest="save_output",
        action="store_true",
    )

    dit_profile = subparsers.add_parser(
        "profile-dit",
        help="profile prompt encoding, four-step DMD, and clean-KV only",
    )
    _add_common_transport_args(dit_profile)
    dit_profile.add_argument("--prompt", default=DEFAULT_PROMPT)
    dit_profile.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    dit_profile.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    dit_profile.add_argument("--num-frames", type=int)
    dit_profile.add_argument("--duration-seconds", type=float)
    dit_profile.add_argument("--fps", type=int, default=DEFAULT_FPS)
    dit_profile.add_argument("--seed", type=int, default=DEFAULT_SEED)
    dit_profile.add_argument("--warmup", type=int, default=0)
    dit_profile.add_argument("--repeat", type=int, default=1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "generate":
        if args.num_requests <= 0:
            raise ValueError("--num-requests must be positive")
        summary = asyncio.run(run_generate(args))
    elif args.command == "profile-vae":
        summary = asyncio.run(run_profile_vae(args))
    else:
        summary = asyncio.run(run_profile_dit(args))
    _write_summary(summary, args.summary_json)


if __name__ == "__main__":
    main()
