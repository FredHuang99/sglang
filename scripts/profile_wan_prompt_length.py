#!/usr/bin/env python3
"""Measure Wan DiT time for exact raw prompt token lengths on one GPU."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from profile_diffusion_common import (
    DEFAULT_REFERENCE_IMAGE,
    NUM_RUNS,
    NUM_WARMUP_RUNS,
    WAN21,
    WAN22,
    WAN_MODEL_SPECS,
    allocate_ports,
    build_server_command,
    build_server_environment,
    denoising_stage_ms,
    launch_server,
    materialize_reference_image,
    measured_mean,
    model_metadata,
    prepare_output_dir,
    prompt_fingerprint,
    read_perf_dump,
    repository_commit,
    response_request_id,
    save_json,
    send_generation_request,
    server_base_url,
    stop_server,
    validate_model_path,
    wait_for_ready,
)


PROMPT_LENGTHS = (32, 64, 256, 512, 2048)
EFFECTIVE_WAN_CONTEXT_TOKENS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Wan DenoisingStage time for raw prompt lengths 32, 64, "
            "256, 512, and 2048 tokens on one GPU."
        )
    )
    parser.add_argument("--wan22-model-path", type=Path, default=WAN22.default_path)
    parser.add_argument("--wan21-model-path", type=Path, default=WAN21.default_path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--wan22-reference-image", default=DEFAULT_REFERENCE_IMAGE
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/outputs/wan_prompt_length"),
    )
    parser.add_argument("--server-timeout-s", type=float, default=3600.0)
    parser.add_argument("--request-timeout-s", type=float, default=3600.0)
    parser.add_argument("--reference-download-timeout-s", type=float, default=300.0)
    parser.add_argument("--perf-timeout-s", type=float, default=30.0)
    parser.add_argument("--ready-poll-interval-s", type=float, default=0.1)
    parser.add_argument("--video-poll-interval-s", type=float, default=1.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=60.0)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    args = parser.parse_args()
    for name in (
        "server_timeout_s",
        "request_timeout_s",
        "reference_download_timeout_s",
        "perf_timeout_s",
        "ready_poll_interval_s",
        "video_poll_interval_s",
        "shutdown_timeout_s",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.cooldown_s < 0:
        parser.error("--cooldown-s must be non-negative")
    return args


def raw_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    token_ids = encoded["input_ids"]
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError("Tokenizer did not return a flat integer input_ids list")
    return token_ids


def repeated_prompt_with_exact_tokens(
    tokenizer: Any, target_tokens: int, fragment: str
) -> str | None:
    def make_prompt(repetitions: int) -> str:
        return " ".join([fragment] * repetitions)

    low = 1
    high = 1
    high_count = len(raw_token_ids(tokenizer, make_prompt(high)))
    while high_count < target_tokens and high < target_tokens * 8:
        low = high + 1
        high *= 2
        high_count = len(raw_token_ids(tokenizer, make_prompt(high)))
    if high_count < target_tokens:
        return None

    low = max(1, low)
    while low <= high:
        middle = (low + high) // 2
        prompt = make_prompt(middle)
        token_count = len(raw_token_ids(tokenizer, prompt))
        if token_count == target_tokens:
            return prompt
        if token_count < target_tokens:
            low = middle + 1
        else:
            high = middle - 1
    return None


def build_exact_token_prompt(tokenizer: Any, target_tokens: int) -> str:
    for fragment in ("a", "cat", "video", "light", "red", "blue", "one", "."):
        prompt = repeated_prompt_with_exact_tokens(tokenizer, target_tokens, fragment)
        if prompt is not None:
            actual = len(raw_token_ids(tokenizer, prompt))
            if actual != target_tokens:
                raise RuntimeError(
                    f"Internal prompt length mismatch: expected {target_tokens}, got {actual}"
                )
            return prompt
    raise RuntimeError(
        f"Could not construct a deterministic prompt of exactly {target_tokens} tokens"
    )


def load_prompts(model_path: Path) -> dict[int, str]:
    tokenizer_path = model_path / "tokenizer"
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_path}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    prompts = {
        target: build_exact_token_prompt(tokenizer, target)
        for target in PROMPT_LENGTHS
    }
    for target, prompt in prompts.items():
        actual = len(raw_token_ids(tokenizer, prompt))
        if actual != target:
            raise ValueError(
                f"Prompt validation failed for {model_path}: expected {target}, got {actual}"
            )
    return prompts


def write_markdown_summary(path: Path, summaries: dict[str, dict[int, float]]) -> str:
    header = "| Model | " + " | ".join(str(length) for length in PROMPT_LENGTHS) + " |"
    separator = "|---|" + "|".join("---:" for _ in PROMPT_LENGTHS) + "|"
    rows = []
    for spec in WAN_MODEL_SPECS:
        values = " | ".join(
            f"{summaries[spec.key][length]:.3f}" for length in PROMPT_LENGTHS
        )
        rows.append(f"| {spec.label} | {values} |")
    text = (
        "# Wan Prompt Length DiT Time (ms)\n\n"
        + header
        + "\n"
        + separator
        + "\n"
        + "\n".join(rows)
        + "\n\n"
        + "Raw prompt lengths include tokenizer special tokens and are measured "
        + "before padding or truncation. Requests do not set `max_sequence_length`; "
        + "the Wan text path pads or truncates every prompt to 512 tokens before DiT.\n"
    )
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    args = parse_args()
    requested_paths = {
        WAN22.key: args.wan22_model_path,
        WAN21.key: args.wan21_model_path,
    }
    model_paths = {
        spec.key: validate_model_path(spec, requested_paths[spec.key])
        for spec in WAN_MODEL_SPECS
    }
    prompts = {
        spec.key: load_prompts(model_paths[spec.key]) for spec in WAN_MODEL_SPECS
    }
    output_dir = prepare_output_dir(args.output_dir)
    profile_commit = repository_commit()
    reference_image = materialize_reference_image(
        args.wan22_reference_image,
        output_dir,
        args.reference_download_timeout_s,
    )

    state_path = output_dir / "results.json"
    state: dict[str, Any] = {
        "status": "running",
        "metric": "perf_dump.steps[DenoisingStage].duration_ms",
        "unit": "ms",
        "commit_hash": profile_commit,
        "gpu_count": 1,
        "runs_per_length": NUM_RUNS,
        "warmup_runs": NUM_WARMUP_RUNS,
        "raw_prompt_lengths": list(PROMPT_LENGTHS),
        "effective_wan_context_tokens": EFFECTIVE_WAN_CONTEXT_TOKENS,
        "reference_image": str(reference_image),
        "models": [
            model_metadata(spec, model_paths[spec.key]) for spec in WAN_MODEL_SPECS
        ],
        "model_runs": [],
    }
    save_json(state_path, state)
    summaries: dict[str, dict[int, float]] = {
        spec.key: {} for spec in WAN_MODEL_SPECS
    }

    for spec in WAN_MODEL_SPECS:
        model_path = model_paths[spec.key]
        model_dir = output_dir / spec.key
        server_dir = model_dir / "server"
        ports = allocate_ports(args.host)
        command = build_server_command(spec, model_path, 1, args.host, ports, server_dir)
        model_record: dict[str, Any] = {
            "model": spec.key,
            "ports": ports,
            "command": command,
            "status": "starting",
            "lengths": [],
        }
        state["model_runs"].append(model_record)
        save_json(state_path, state)
        server = None
        print(f"[server] launching {spec.label} prompt-length profile", flush=True)
        try:
            server = launch_server(
                command,
                model_dir / "server.log",
                build_server_environment(server_dir),
            )
            base_url = server_base_url(args.host, ports["http"])
            card, ready_ns = wait_for_ready(
                server,
                base_url,
                spec,
                model_path,
                1,
                args.server_timeout_s,
                args.ready_poll_interval_s,
            )
            model_record["status"] = "running"
            model_record["ready_model_card"] = card
            model_record["ready_after_launch_ms"] = (
                ready_ns - server.started_ns
            ) / 1_000_000.0
            save_json(state_path, state)

            for prompt_length in PROMPT_LENGTHS:
                prompt = prompts[spec.key][prompt_length]
                length_record: dict[str, Any] = {
                    "raw_prompt_tokens": prompt_length,
                    "effective_wan_context_tokens": EFFECTIVE_WAN_CONTEXT_TOKENS,
                    "prompt": prompt_fingerprint(prompt),
                    "runs": [],
                    "status": "running",
                }
                model_record["lengths"].append(length_record)
                save_json(state_path, state)
                for run_index in range(NUM_RUNS):
                    perf_path = (
                        model_dir
                        / "perf"
                        / f"tokens_{prompt_length}"
                        / f"run_{run_index + 1:02d}.json"
                    )
                    print(
                        f"[request] {spec.label} tokens={prompt_length} "
                        f"run={run_index + 1}/{NUM_RUNS}",
                        flush=True,
                    )
                    response = send_generation_request(
                        base_url,
                        spec,
                        prompt,
                        perf_path,
                        reference_image if spec.reference_image else None,
                        args.request_timeout_s,
                        args.video_poll_interval_s,
                    )
                    perf_dump = read_perf_dump(
                        perf_path,
                        args.perf_timeout_s,
                        expected_request_id=response_request_id(response),
                        expected_commit_hash=profile_commit,
                        expected_model_path=model_path,
                        expected_world_size=1,
                    )
                    length_record["runs"].append(
                        {
                            "run": run_index + 1,
                            "warmup": run_index < NUM_WARMUP_RUNS,
                            "dit_duration_ms": denoising_stage_ms(perf_dump),
                            "perf_dump_path": str(perf_path),
                        }
                    )
                    save_json(state_path, state)

                average_ms = measured_mean(
                    length_record["runs"], "dit_duration_ms"
                )
                length_record["measured_average_ms"] = average_ms
                length_record["status"] = "complete"
                summaries[spec.key][prompt_length] = average_ms
                save_json(state_path, state)
            model_record["status"] = "complete"
            save_json(state_path, state)
        except BaseException as exc:
            model_record["status"] = "failed"
            model_record["error"] = f"{type(exc).__name__}: {exc}"
            state["status"] = "failed"
            save_json(state_path, state)
            raise
        finally:
            stop_server(server, args.shutdown_timeout_s)
            if args.cooldown_s:
                time.sleep(args.cooldown_s)

    markdown = write_markdown_summary(
        output_dir / "prompt_length_dit_ms.md", summaries
    )
    state["status"] = "complete"
    state["summary"] = {
        spec.key: {
            str(prompt_length): summaries[spec.key][prompt_length]
            for prompt_length in PROMPT_LENGTHS
        }
        for spec in WAN_MODEL_SPECS
    }
    save_json(state_path, state)
    print("\n" + markdown, end="", flush=True)


if __name__ == "__main__":
    main()
