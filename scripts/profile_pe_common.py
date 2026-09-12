"""Shared Qwen model checks and runtime metadata for the PE profilers."""

from __future__ import annotations

import argparse
import json
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

MODEL_FAMILIES = ("hunyuan-reprompt", "qwen2.5-7b")
QWEN_MODEL_FAMILY = "qwen2.5-7b"
QWEN_ARCHITECTURE = "Qwen2ForCausalLM"
QWEN_TP_SIZES = (1, 2, 4)
QWEN_IO_MATRIX = {
    16: (128, 768),
    80: (704,),
    144: (640,),
    208: (576,),
    272: (512,),
    336: (448,),
    400: (384,),
    464: (320,),
    528: (256,),
    592: (192,),
    656: (128,),
}


def apply_model_defaults(
    args: argparse.Namespace, parser: argparse.ArgumentParser, study: str
) -> None:
    """Resolve family-specific CLI defaults without changing Hunyuan calls."""
    is_qwen = args.model_family == QWEN_MODEL_FAMILY
    if args.model_path is None:
        args.model_path = Path(
            "/data/models/Qwen2.5-7B-Instruct"
            if is_qwen
            else "/workspace/models/reprompt"
        )
    if args.served_model_name is None:
        args.served_model_name = (
            "Qwen2.5-7B-Instruct" if is_qwen else "HunyuanImage-2.1-reprompt"
        )
    if args.output_dir is None:
        args.output_dir = Path(
            f"/data/outputs/qwen25_7b_{study}"
            if is_qwen
            else f"/workspace/outputs/hunyuan_reprompt_{study}"
        )
    if args.all_reduce_mode is None:
        args.all_reduce_mode = "custom_v2" if is_qwen else "legacy_v1"
    supported_tp = QWEN_TP_SIZES if is_qwen else (1, 2, 4, 8)
    args.tp_sizes = list(dict.fromkeys(args.tp_sizes or supported_tp))
    invalid = [tp for tp in args.tp_sizes if tp not in supported_tp]
    if invalid:
        parser.error(
            f"{args.model_family} supports TP {supported_tp}; got {invalid}. "
            + (
                "Qwen's 28 attention heads cannot be split across TP8."
                if is_qwen
                else ""
            )
        )


def resolve_qwen_path(model_path: Path) -> Path:
    candidate = model_path.resolve()
    if not (candidate / "config.json").is_file():
        raise FileNotFoundError(
            f"Missing {candidate / 'config.json'}; pass the local Qwen model directory."
        )
    return candidate


def validate_qwen_checkpoint(model_path: Path) -> None:
    # Qwen carries its chat template in tokenizer_config.json, not a Hunyuan
    # tokenizer module or a separately required chat_template.jinja file.
    required = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Qwen directory {model_path}: {missing}")
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    if any(not isinstance(name, str) or not name for name in weight_map.values()):
        raise ValueError(f"Invalid shard names in {index_path}")
    for name in sorted(set(weight_map.values())):
        relative = Path(name)
        # HF cache snapshots can legitimately symlink shards into ../blobs.
        # Check the index entry itself, not the symlink's resolved destination.
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid Qwen weight shard name: {name}")
        if not (model_path / relative).is_file():
            raise FileNotFoundError(f"Missing or invalid Qwen weight shard: {name}")
    tokenizer_config = json.loads(
        (model_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    tokenizer = json.loads((model_path / "tokenizer.json").read_text(encoding="utf-8"))
    if tokenizer_config.get("tokenizer_class") not in {
        "Qwen2Tokenizer",
        "Qwen2TokenizerFast",
    }:
        raise ValueError("Expected a Qwen2 tokenizer in tokenizer_config.json")
    if tokenizer.get("model", {}).get("type") != "BPE":
        raise ValueError("Expected the Qwen BPE tokenizer in tokenizer.json")


def load_qwen_config(model_path: Path) -> dict[str, Any]:
    validate_qwen_checkpoint(model_path)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if QWEN_ARCHITECTURE not in (config.get("architectures") or []):
        raise ValueError(f"Expected architecture {QWEN_ARCHITECTURE}")
    expected = {
        "model_type": "qwen2",
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(
                f"Expected Qwen2.5-7B {name}={value!r}, got {config.get(name)!r}"
            )
    for name in ("vocab_size", "max_position_embeddings"):
        if type(config.get(name)) is not int or config[name] <= 1:
            raise ValueError(f"Invalid Qwen {name}: {config.get(name)!r}")
    if config.get("rope_scaling"):
        raise ValueError("This profile uses Qwen's native context without rope_scaling")
    if config.get("quantization_config"):
        raise ValueError("This profile requires the unquantized Qwen BF16 checkpoint")
    return config


def read_runtime_parameters(url: str) -> dict[str, Any]:
    """Read resolved worker settings AFTER the startup readiness timestamp."""
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"{url}/server_info", timeout=30.0)
        response.raise_for_status()
        info = response.json()
    if not isinstance(info, dict):
        raise RuntimeError("Invalid /server_info response")
    states = info.get("internal_states")
    if (
        not isinstance(states, list)
        or not states
        or not all(isinstance(state, dict) for state in states)
    ):
        raise RuntimeError("/server_info omitted resolved scheduler states")
    fields = (
        "chunked_prefill_size",
        "max_running_requests",
        "effective_max_running_requests_per_dp",
        "cuda_graph_max_bs_decode",
        "cuda_graph_bs_decode",
        "cuda_graph_backend_decode",
        "cuda_graph_backend_prefill",
        "context_length",
        "mem_fraction_static",
        "attention_backend",
        "dtype",
        "skip_server_warmup",
    )
    required = (
        "chunked_prefill_size",
        "effective_max_running_requests_per_dp",
        "cuda_graph_max_bs_decode",
    )
    resolved = []
    for state in states:
        values = {key: state.get(key, info.get(key)) for key in fields}
        # The convenience CLI fields can remain None. GPU-dependent defaults
        # are resolved into cuda_graph_config, so prefer that canonical value.
        graphs = state.get("cuda_graph_config") or info.get("cuda_graph_config") or {}
        decode = graphs.get("decode", {})
        prefill = graphs.get("prefill", {})
        for destination, source in (
            ("cuda_graph_max_bs_decode", "max_bs"),
            ("cuda_graph_bs_decode", "bs"),
            ("cuda_graph_backend_decode", "backend"),
        ):
            if source in decode:
                values[destination] = decode[source]
        if "backend" in prefill:
            values["cuda_graph_backend_prefill"] = prefill["backend"]
        if any(values[name] is None for name in required):
            raise RuntimeError(f"Missing resolved startup parameters: {values}")
        resolved.append(values)
    return {"resolved_per_dp": resolved, "server_info": info}


@contextmanager
def cleanup_on_sigterm():
    """Let the existing finally blocks stop only this profiler's servers."""

    def interrupt(signum, frame):
        # Ignore repeated TERM while the finally blocks reap the server group.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt("Profiling stopped by SIGTERM")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
