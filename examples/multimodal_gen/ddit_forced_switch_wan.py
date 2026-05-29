"""Submit one Wan2.1 DDiT forced-switch correctness request."""

from __future__ import annotations

import argparse
import json

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompt", default="A cinematic video of a red kite flying over mountains.")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--resolution-key", default="720p")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--initial-ranks", default="0")
    parser.add_argument("--switch-plan", default="15:1->2;30:2->4;45:4->8")
    parser.add_argument("--ddit-vae-k", type=int, default=1)
    parser.add_argument("--ddit-vae-ranks", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    payload = {
        "request_id": args.request_id,
        "prompt": args.prompt,
        "size": args.size,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "resolution_key": args.resolution_key,
        "ddit_resolution_key": args.resolution_key,
        "ddit_initial_ranks": args.initial_ranks,
        "ddit_switch_plan": args.switch_plan,
        "ddit_vae_k": args.ddit_vae_k,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    if args.ddit_vae_ranks:
        payload["ddit_vae_ranks"] = args.ddit_vae_ranks

    endpoint = args.server_url.rstrip("/") + "/v1/videos"
    response = requests.post(endpoint, json=payload, timeout=args.timeout)
    try:
        body = response.json()
    except Exception:
        body = response.text
    print(json.dumps({"status_code": response.status_code, "response": body}, indent=2))


if __name__ == "__main__":
    main()
