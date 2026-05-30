"""Send mixed-resolution DDiT video workloads to the OpenAI video endpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SIZE_MAP = {
    "144p": "256x144",
    "240p": "432x240",
    "360p": "640x352",
    "480p": "832x480",
    "720p": "1280x720",
    "1k": "1024x576",
    "2k": "2048x1152"
}
DEFAULT_IMAGE_RELATIVE_PATH = os.path.join("examples", "assets", "example_image.png")
ALLOWED_DDIT_VAE_K = (1, 2, 4, 8)
INPUT_REFERENCE_MODES = ("auto", "none", "image")


def detect_project_root(project_root: str | None = None) -> Path:
    if project_root:
        return Path(project_root).expanduser().resolve()

    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        has_default_image = (candidate / DEFAULT_IMAGE_RELATIVE_PATH).exists()
        has_repo_marker = (candidate / "python" / "sglang").exists() or (
            candidate / ".git"
        ).exists()
        if has_default_image and has_repo_marker:
            return candidate
    raise FileNotFoundError(
        "Could not detect project root containing "
        f"{DEFAULT_IMAGE_RELATIVE_PATH!r}. Pass --project-root explicitly."
    )


def resolve_project_image_path(
    image_path: str | None = None,
    *,
    project_root: str | None = None,
) -> str | None:
    if image_path == "":
        return None

    root = detect_project_root(project_root)
    path = (
        root / DEFAULT_IMAGE_RELATIVE_PATH
        if image_path is None
        else Path(image_path).expanduser()
    )
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input image path does not exist: {path}")
    return str(path)


def default_project_image_path() -> str:
    return str(detect_project_root() / DEFAULT_IMAGE_RELATIVE_PATH)


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    resolution: str
    payload: dict[str, Any]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_ratios(value: str) -> list[float]:
    ratios = [float(part) for part in parse_csv(value)]
    if not ratios:
        raise ValueError("--ratios cannot be empty")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("--ratios cannot contain negative values")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"--ratios must sum to 1.0, got {sum(ratios)}")
    return ratios


def counts_from_ratios(num_requests: int, ratios: list[float]) -> list[int]:
    if num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    counts = [round(num_requests * ratio) for ratio in ratios[:-1]]
    last = num_requests - sum(counts)
    if last < 0:
        # Rounding can overshoot for small request counts; trim from largest bucket.
        counts.append(0)
        over = -last
        for idx in sorted(range(len(counts) - 1), key=lambda i: counts[i], reverse=True):
            take = min(over, counts[idx])
            counts[idx] -= take
            over -= take
            if over == 0:
                break
        counts[-1] = num_requests - sum(counts[:-1])
    else:
        counts.append(last)
    return counts


def load_size_map(raw: str | None) -> dict[str, str]:
    if not raw:
        return dict(DEFAULT_SIZE_MAP)
    custom = json.loads(raw)
    merged = dict(DEFAULT_SIZE_MAP)
    merged.update({str(k): str(v) for k, v in custom.items()})
    return merged


def normalize_profile_model_id(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if "wan" in normalized and "2.1" in normalized and "1.3" in normalized:
        return "wan2.1-t2v-1.3b"
    if "z-image" in normalized or "zimage" in normalized:
        return "z-image"
    return normalized


def infer_input_reference_required(model_id: str | None) -> bool | None:
    """Infer whether a workload should send input_reference from model naming."""
    if not model_id:
        return None
    normalized = normalize_profile_model_id(model_id)
    raw = str(model_id or "").strip().lower().replace("_", "-")
    joined = f"{raw} {normalized}"
    if "ti2v" in joined or "i2v" in joined or "i2i" in joined:
        return True
    if normalized == "z-image" or "t2i" in joined or "t2v" in joined:
        return False
    return None


def resolve_input_reference_path(
    *,
    image_path: str | None,
    mode: str,
    model_id: str | None,
    project_root: str | None = None,
) -> str | None:
    mode = str(mode or "auto").lower()
    if mode not in INPUT_REFERENCE_MODES:
        raise ValueError(
            f"--input-reference-mode must be one of {INPUT_REFERENCE_MODES}, "
            f"got {mode!r}"
        )
    if mode == "none":
        return None
    if mode == "image":
        resolved = resolve_project_image_path(image_path, project_root=project_root)
        if resolved is None:
            raise ValueError(
                "--input-reference-mode image requires a non-empty --image-path"
            )
        return resolved

    inferred = infer_input_reference_required(model_id)
    if inferred is False:
        return None
    if inferred is True:
        resolved = resolve_project_image_path(image_path, project_root=project_root)
        if resolved is None:
            raise ValueError(
                f"Model {model_id!r} appears to require input_reference, "
                "but --image-path disabled it."
            )
        return resolved
    if image_path in (None, ""):
        return None
    return resolve_project_image_path(image_path, project_root=project_root)


def _validate_vae_k(value: Any) -> int:
    vae_k = int(value)
    if vae_k not in ALLOWED_DDIT_VAE_K:
        raise ValueError(
            f"DDiT VAE k must be one of {ALLOWED_DDIT_VAE_K}, got {vae_k}"
        )
    return vae_k


def load_profile_model_payload(
    profile_path: str,
    profile_model_id: str | None,
    *,
    project_root: str | None = None,
) -> dict[str, Any]:
    path = Path(profile_path).expanduser()
    if not path.is_absolute():
        path = detect_project_root(project_root) / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"DDiT profile path does not exist: {path}")
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if "models" not in payload:
        return payload
    if not profile_model_id:
        raise ValueError(
            "--ddit-profile-model-id is required when --ddit-vae-k profile "
            "uses a multi-model profile."
        )
    models = payload.get("models") or {}
    normalized = normalize_profile_model_id(profile_model_id)
    if profile_model_id in models:
        return models[profile_model_id]
    if normalized in models:
        return models[normalized]
    raise ValueError(
        f"Model id {profile_model_id!r} was not found in DDiT profile {path}"
    )


def build_vae_k_resolver(
    raw_vae_k: str | None,
    *,
    profile_path: str | None = None,
    profile_model_id: str | None = None,
    project_root: str | None = None,
):
    if raw_vae_k is None or raw_vae_k == "":
        return None
    raw = str(raw_vae_k).strip()
    if raw.lower() in ("profile", "auto"):
        if not profile_path:
            raise ValueError("--ddit-profile-path is required for --ddit-vae-k profile")
        payload = load_profile_model_payload(
            profile_path, profile_model_id, project_root=project_root
        )
        table = {
            str(resolution): _validate_vae_k(value)
            for resolution, value in (payload.get("opt_vae_k") or {}).items()
        }
        return lambda resolution: table.get(str(resolution), 1)
    if raw.startswith("{"):
        table_payload = json.loads(raw)
        if not isinstance(table_payload, dict):
            raise ValueError("--ddit-vae-k JSON value must be an object")
        table = {
            str(resolution): _validate_vae_k(value)
            for resolution, value in table_payload.items()
        }
        return lambda resolution: table.get(str(resolution), 1)
    constant = _validate_vae_k(raw)
    return lambda _resolution: constant


def build_workload(
    *,
    num_requests: int,
    resolutions: list[str],
    ratios: list[float],
    seed: int,
    prompt: str,
    size_map: dict[str, str],
    image_path: str | None = None,
    include_input_reference: bool = False,
    extra_payload: dict[str, Any] | None = None,
    ddit_vae_k_resolver: Any | None = None,
    workload_id: str | None = None,
) -> list[WorkloadRequest]:
    if len(resolutions) != len(ratios):
        raise ValueError(
            f"resolutions and ratios length mismatch: {len(resolutions)} != {len(ratios)}"
        )
    counts = counts_from_ratios(num_requests, ratios)
    requests_to_send: list[WorkloadRequest] = []
    for resolution, count in zip(resolutions, counts):
        if resolution not in size_map:
            raise ValueError(f"No size mapping for resolution {resolution!r}")
        for idx in range(count):
            request_id = f"ddit_{resolution}_{idx:05d}_{seed}"
            payload = {
                "request_id": request_id,
                "prompt": prompt,
                "size": size_map[resolution],
                "resolution_key": resolution,
                "ddit_resolution_key": resolution,
            }
            if workload_id:
                payload["ddit_workload_id"] = workload_id
                payload["ddit_workload_num_requests"] = num_requests
            if include_input_reference and image_path:
                payload["input_reference"] = image_path
            if ddit_vae_k_resolver is not None:
                payload["ddit_vae_k"] = _validate_vae_k(
                    ddit_vae_k_resolver(resolution)
                )
            if extra_payload:
                payload.update(extra_payload)
            requests_to_send.append(
                WorkloadRequest(
                    request_id=request_id,
                    resolution=resolution,
                    payload=payload,
                )
            )
    rng = random.Random(seed)
    rng.shuffle(requests_to_send)
    return requests_to_send


def parse_rate(value: str) -> float | None:
    if value == "burst":
        return None
    rate = float(value)
    if rate <= 0:
        raise ValueError("--rate must be positive or 'burst'")
    return rate


def send_workload(
    *,
    server_url: str,
    workload: list[WorkloadRequest],
    rate: float | None,
    timeout: float,
    max_inflight: int | None = None,
    post_fn: Any | None = None,
) -> list[dict[str, Any]]:
    endpoint = server_url.rstrip("/") + "/v1/videos"
    sleep_s = None if rate is None else 1.0 / rate
    if not workload:
        return []
    inflight_limit = len(workload) if max_inflight is None else int(max_inflight)
    if inflight_limit <= 0:
        raise ValueError("--max-inflight must be positive when set")
    inflight_limit = min(inflight_limit, len(workload))
    if post_fn is None:
        import requests

        post = requests.post
    else:
        post = post_fn
    responses: list[dict[str, Any] | None] = [None] * len(workload)
    futures: dict[Future[dict[str, Any]], int] = {}

    def post_one(item: WorkloadRequest, submit_time: float) -> dict[str, Any]:
        start = time.time()
        record: dict[str, Any] = {
            "client_request_id": item.request_id,
            "resolution": item.resolution,
            "client_submit_time": submit_time,
            "client_request_start_time": start,
        }
        try:
            response = post(endpoint, json=item.payload, timeout=timeout)
            end = time.time()
            record.update(
                {
                    "status_code": response.status_code,
                    "client_response_time": end,
                    "client_elapsed_s": end - submit_time,
                    "client_http_elapsed_s": end - start,
                }
            )
            try:
                record["response"] = response.json()
            except Exception:
                record["response_text"] = response.text
        except Exception as exc:
            end = time.time()
            record.update(
                {
                    "client_response_time": end,
                    "client_elapsed_s": end - submit_time,
                    "client_http_elapsed_s": end - start,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        return record

    def collect_done(done: set[Future[dict[str, Any]]]) -> None:
        for future in done:
            idx = futures.pop(future)
            try:
                responses[idx] = future.result()
            except Exception as exc:
                item = workload[idx]
                now = time.time()
                responses[idx] = {
                    "client_request_id": item.request_id,
                    "resolution": item.resolution,
                    "client_response_time": now,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

    with ThreadPoolExecutor(max_workers=inflight_limit) as executor:
        for idx, item in enumerate(workload):
            while len(futures) >= inflight_limit:
                done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                collect_done(done)

            submit_time = time.time()
            future = executor.submit(post_one, item, submit_time)
            futures[future] = idx
            if sleep_s is not None and idx + 1 < len(workload):
                time.sleep(sleep_s)

        while futures:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            collect_done(done)

    return [record for record in responses if record is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--resolutions", required=True, help="Comma-separated labels.")
    parser.add_argument(
        "--ratios", required=True, help="Comma-separated floats summing to 1."
    )
    parser.add_argument("--rate", default="1", help="Requests per second or 'burst'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="A cinematic video of a small robot walking through a city.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Input image path for image-conditioned models. Relative paths are "
        "resolved against --project-root. In auto mode this defaults to "
        "examples/assets/example_image.png only for I2V/TI2V model ids.",
    )
    parser.add_argument(
        "--input-reference-mode",
        choices=INPUT_REFERENCE_MODES,
        default="auto",
        help=(
            "Controls whether requests include input_reference: auto infers from "
            "--input-model-id/--ddit-profile-model-id, none disables it, image "
            "forces it."
        ),
    )
    parser.add_argument(
        "--input-model-id",
        default=None,
        help=(
            "Optional model id used only for input_reference auto inference. "
            "Defaults to --ddit-profile-model-id when omitted."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root used to resolve default and relative image paths. "
        "Auto-detected from this script when omitted.",
    )
    parser.add_argument("--size-map-json", default=None)
    parser.add_argument(
        "--ddit-vae-k",
        default=None,
        help=(
            "Optional per-request DDiT VAE GPU count. Accepts an integer "
            "1/2/4/8, a JSON map such as '{\"144p\":1,\"720p\":8}', or "
            "'profile'/'auto' to read opt_vae_k from --ddit-profile-path."
        ),
    )
    parser.add_argument(
        "--ddit-profile-path",
        default=None,
        help="Profile JSON path used when --ddit-vae-k is profile/auto.",
    )
    parser.add_argument(
        "--ddit-profile-model-id",
        default=None,
        help="Model id used to select a model from a multi-model DDiT profile.",
    )
    parser.add_argument(
        "--extra-json", default=None, help="Extra JSON payload merged into every request."
    )
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="Maximum client-side concurrent HTTP requests. Defaults to "
        "--num-requests so the arrival stream is not serialized by the client.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    resolutions = parse_csv(args.resolutions)
    ratios = parse_ratios(args.ratios)
    input_model_id = args.input_model_id or args.ddit_profile_model_id
    image_path = resolve_input_reference_path(
        image_path=args.image_path,
        mode=args.input_reference_mode,
        model_id=input_model_id,
        project_root=args.project_root,
    )
    workload = build_workload(
        num_requests=args.num_requests,
        resolutions=resolutions,
        ratios=ratios,
        seed=args.seed,
        prompt=args.prompt,
        size_map=load_size_map(args.size_map_json),
        image_path=image_path,
        include_input_reference=image_path is not None,
        extra_payload=json.loads(args.extra_json) if args.extra_json else None,
        ddit_vae_k_resolver=build_vae_k_resolver(
            args.ddit_vae_k,
            profile_path=args.ddit_profile_path,
            profile_model_id=args.ddit_profile_model_id,
            project_root=args.project_root,
        ),
        workload_id=f"ddit-workload-{int(time.time() * 1000)}-{args.seed}-{args.num_requests}",
    )
    if args.dry_run:
        print(json.dumps([item.payload for item in workload], indent=2))
        return
    responses = send_workload(
        server_url=args.server_url,
        workload=workload,
        rate=parse_rate(args.rate),
        timeout=args.timeout,
        max_inflight=args.max_inflight,
    )
    print(json.dumps(responses, indent=2))


if __name__ == "__main__":
    main()
