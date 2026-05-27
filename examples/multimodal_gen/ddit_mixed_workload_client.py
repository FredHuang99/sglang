"""Send mixed-resolution DDiT video workloads to the OpenAI video endpoint."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_SIZE_MAP = {
    "144p": "256x144",
    "240p": "426x240",
    "360p": "640x360",
    "480p": "854x480",
    "720p": "1280x720",
    "1080p": "1920x1080",
}


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


def build_workload(
    *,
    num_requests: int,
    resolutions: list[str],
    ratios: list[float],
    seed: int,
    prompt: str,
    size_map: dict[str, str],
    extra_payload: dict[str, Any] | None = None,
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
) -> list[dict[str, Any]]:
    import requests

    endpoint = server_url.rstrip("/") + "/v1/videos"
    sleep_s = None if rate is None else 1.0 / rate
    responses = []
    for item in workload:
        start = time.time()
        response = requests.post(endpoint, json=item.payload, timeout=timeout)
        record = {
            "client_request_id": item.request_id,
            "resolution": item.resolution,
            "status_code": response.status_code,
            "elapsed_s": time.time() - start,
        }
        try:
            record["response"] = response.json()
        except Exception:
            record["response_text"] = response.text
        responses.append(record)
        if sleep_s is not None:
            time.sleep(sleep_s)
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--resolutions", required=True, help="Comma-separated labels.")
    parser.add_argument("--ratios", required=True, help="Comma-separated floats summing to 1.")
    parser.add_argument("--rate", default="1", help="Requests per second or 'burst'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A cinematic video of a small robot walking through a city.")
    parser.add_argument("--size-map-json", default=None)
    parser.add_argument("--extra-json", default=None, help="Extra JSON payload merged into every request.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    resolutions = parse_csv(args.resolutions)
    ratios = parse_ratios(args.ratios)
    workload = build_workload(
        num_requests=args.num_requests,
        resolutions=resolutions,
        ratios=ratios,
        seed=args.seed,
        prompt=args.prompt,
        size_map=load_size_map(args.size_map_json),
        extra_payload=json.loads(args.extra_json) if args.extra_json else None,
    )
    if args.dry_run:
        print(json.dumps([item.payload for item in workload], indent=2))
        return
    responses = send_workload(
        server_url=args.server_url,
        workload=workload,
        rate=parse_rate(args.rate),
        timeout=args.timeout,
    )
    print(json.dumps(responses, indent=2))


if __name__ == "__main__":
    main()
