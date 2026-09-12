#!/usr/bin/env python3
"""Measure ordinary-chat lengths on T2V-CompBench V2 with eight TP1 replicas.

Installation and downloads are deliberately manual. Only ``run`` starts services;
all commands write outside the source checkout. See --help for path overrides.
"""

import argparse
import asyncio
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import signal
import socket
import subprocess
import sys
import time
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path

FILES = [
    "1_consistent_attr.txt",
    "2_dynamic_attr.txt",
    "3_spatial_relationship.txt",
    "4_motion_binding.txt",
    "5_action_binding.txt",
    "6_interaction.txt",
    "7_numeracy.txt",
]
SYSTEM = "You are a helpful assistant."
MODEL_SPECS = {
    "qwen2.5-7b": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
    },
    "qwen2.5-14b": {
        "repo_id": "Qwen/Qwen2.5-14B-Instruct",
        "hidden_size": 5120,
        "num_hidden_layers": 48,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
    },
}
NO_SYSTEM_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SAMPLING = dict(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.05,
    max_new_tokens=None,
    ignore_eos=False,
    skip_special_tokens=True,
)
CONTEXT = 32768
VALID = {"eos", "length_limit"}
REPO = Path(__file__).resolve().parents[1]
CATEGORY_LABELS = {
    "1_consistent_attr": "一致属性绑定",
    "2_dynamic_attr": "动态属性绑定",
    "3_spatial_relationship": "空间关系",
    "4_motion_binding": "运动绑定",
    "5_action_binding": "动作绑定",
    "6_interaction": "对象交互",
    "7_numeracy": "数量关系",
}
STATUS_LABELS = {
    "eos": "自然结束",
    "length_limit": "达到长度限制",
    "failed": "请求失败",
    "interrupted": "已中断",
    "unfinished": "尚无最终结果",
    "pending": "尚未提交",
}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def packed(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(path, obj):
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cmd(*args):
    return subprocess.check_output(args, text=True).strip()


def sampling_parameters(greedy):
    if greedy:
        return dict(
            temperature=0,
            max_new_tokens=None,
            ignore_eos=False,
            skip_special_tokens=True,
        )
    return dict(SAMPLING)


def validate_model_config(cfg, family):
    spec = MODEL_SPECS[family]
    expected = {k: v for k, v in spec.items() if k != "repo_id"}
    expected.update(model_type="qwen2", max_position_embeddings=CONTEXT)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"Expected {spec['repo_id']} {key}={value}, got {cfg.get(key)!r}"
            )
    if "Qwen2ForCausalLM" not in (cfg.get("architectures") or []):
        raise ValueError("Expected the Qwen2ForCausalLM architecture")
    if cfg.get("quantization_config") or cfg.get("rope_scaling"):
        raise ValueError("Use unquantized BF16 weights with the native context")


def encode_input(tok, text, system_mode):
    if system_mode == "none":
        # Qwen's default template inserts a system message for user-only lists.
        # Render just the user turn and assistant prefix, then send these IDs.
        return tok.encode(
            NO_SYSTEM_TEMPLATE.replace("{prompt}", text), add_special_tokens=False
        )
    return tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        return_tensors=None,
    )


def source_identity(src, paths):
    """Fingerprint staged/unstaged changes and untracked source without editing Git."""
    git = ["git", "-C", str(src)]
    diff = subprocess.check_output(
        [
            *git, "diff", "--no-ext-diff", "--no-textconv", "--binary",
            "HEAD", "--", *paths,
        ]
    )
    untracked = (
        subprocess.check_output(
            [*git, "ls-files", "--others", "--exclude-standard", "-z", "--", *paths]
        )
        .decode("utf-8")
        .split("\0")
    )
    identity = {
        "commit": cmd(*git, "rev-parse", "HEAD"),
        "status": cmd(
            *git, "status", "--porcelain=v1", "--untracked-files=all", "--", *paths
        ),
        "tracked_diff_sha256": digest(diff),
        "untracked_sha256": {
            name: digest((src / name).read_bytes())
            for name in sorted(untracked)
            if name
        },
    }
    return identity, diff


def local_model_revision(model, explicit=None, family="qwen2.5-7b"):
    """Read local HF provenance without calling Hub APIs or changing metadata."""
    info = {"repo_id": MODEL_SPECS[family]["repo_id"]}
    if explicit:
        return {**info, "revision": explicit, "source": "argument"}
    weights = sorted(model.glob("*.safetensors")) or sorted(
        model.glob("pytorch_model*.bin")
    )
    files = [
        model / "config.json",
        model / "generation_config.json",
        model / "tokenizer_config.json",
        model / "tokenizer.json",
        *weights,
    ]
    revisions = set()
    complete = bool(weights)
    for path in files:
        metadata = model / ".cache/huggingface/download" / (path.name + ".metadata")
        try:
            lines = metadata.read_text(encoding="utf-8").splitlines()
            revision, timestamp = lines[0].strip(), float(lines[2])
            if (
                not re.fullmatch(r"[0-9a-f]{40}", revision)
                or path.stat().st_mtime > timestamp + 1
            ):
                complete = False
            else:
                revisions.add(revision)
        except (OSError, UnicodeError, ValueError, IndexError):
            complete = False
    if complete and len(revisions) == 1:
        return {**info, "revision": revisions.pop(), "source": "hf_download_metadata"}
    if model.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", model.name):
        return {**info, "revision": model.name, "source": "hf_snapshot_directory"}
    return {**info, "revision": None, "source": "unknown_local_revision"}


@contextmanager
def output_lock(output_dir):
    import fcntl

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                "This output directory is in use; use status/stop and wait before preparing or exporting."
            )
        yield


def prepare(args):
    import sglang
    from transformers import AutoTokenizer

    root = args.output_dir
    src, ds, model = REPO, args.data_dir, args.model_dir
    if (root / "manifest.json").exists():
        if read(root / "manifest.json").get("signature", {}).get("schema") not in {2, 3}:
            raise ValueError(
                "This result directory belongs to the earlier standalone script. Keep it intact and use that script to resume, or choose a new --output-dir."
            )
    assert Path(sglang.__file__).resolve().is_relative_to(src / "python"), (
        "Install SGLang from this checkout into the selected Python environment"
    )
    assert cmd("git", "-C", str(src), "branch", "--show-current") == "pave_hhy"
    assert cmd("git", "-C", str(ds), "branch", "--show-current") == "V2"
    assert not cmd("git", "-C", str(ds), "status", "--porcelain", "--", "prompts"), (
        "Modified prompts"
    )
    source_paths = [
        "python/sglang",
        "python/pyproject.toml",
        "python/setup.py",
        "scripts/qwen_t2v_length_study.py",
        "scripts/run_qwen_t2v_length_study.sh",
    ]
    source_state, source_diff = source_identity(src, source_paths)
    cfg = read(model / "config.json")
    gen = read(model / "generation_config.json")
    validate_model_config(cfg, args.model_family)
    sampling = sampling_parameters(args.greedy)
    if not args.greedy:
        for k in ("temperature", "top_p", "top_k", "repetition_penalty"):
            assert gen[k] == sampling[k], ("Unexpected generation config", k, gen[k])
        assert gen["do_sample"] is True
    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    for token, token_id in (("<|im_start|>", 151644), ("<|im_end|>", 151645)):
        if tok.convert_tokens_to_ids(token) != token_id:
            raise ValueError(f"Unexpected Qwen ChatML token: {token}")
    special = set(tok.all_special_ids)
    eos = gen["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    special.update(eos)
    config_hashes = {
        p.name: digest(p.read_bytes())
        for p in sorted(model.glob("*"))
        if p.is_file() and (p.suffix in {".json", ".txt", ".jinja"})
    }
    rows, file_hashes = [], {}
    actual = sorted(p.name for p in (ds / "prompts").glob("*.txt"))
    assert actual == sorted(FILES), ("Unexpected prompt files", actual)
    for name in FILES:
        p = ds / "prompts" / name
        file_hashes[name] = digest(p.read_bytes())
        lines = [
            (i, x)
            for i, x in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1)
            if x.strip()
        ]
        assert len(lines) == 200, (name, len(lines))
        for lineno, text in lines:
            raw = tok.encode(text, add_special_tokens=False)
            inputs = encode_input(tok, text, args.system_mode)
            assert 0 < len(inputs) < CONTEXT - 4
            rows.append(
                dict(
                    id=f"{Path(name).stem}-{lineno:04d}",
                    index=len(rows),
                    category=Path(name).stem,
                    source_line=lineno,
                    raw_prompt=text,
                    raw_token_ids=raw,
                    input_token_ids=inputs,
                    raw_tokens=len(raw),
                    input_tokens=len(inputs),
                    **({} if args.greedy else {"sampling_seed": 20260912 + len(rows)}),
                )
            )
    assert len(rows) == 1400
    import importlib.metadata as im

    packages = {
        k: im.version(k)
        for k in ("sglang", "torch", "transformers", "huggingface-hub", "aiohttp")
    }
    signature = dict(
        schema=3,
        task="ordinary_chat",
        model_family=args.model_family,
        system_mode=args.system_mode,
        system=None if args.system_mode == "none" else SYSTEM,
        input_template=(
            NO_SYSTEM_TEMPLATE if args.system_mode == "none" else tok.get_chat_template()
        ),
        decoding="greedy" if args.greedy else "sampling",
        sampling=sampling,
        sampling_defaults="openai",
        context_length=CONTEXT,
        eos_ids=eos,
        special_ids=sorted(special),
        model_revision=local_model_revision(model, args.model_revision, args.model_family),
        model_config_hashes=config_hashes,
        source_commit=source_state["commit"],
        source_state=source_state,
        dataset_commit=cmd("git", "-C", str(ds), "rev-parse", "HEAD"),
        prompt_file_hashes=file_hashes,
        input_hash=digest(packed(rows).encode()),
        packages=packages,
        python=platform.python_version(),
        runner_sha256=digest(Path(__file__).read_bytes()),
        launcher_sha256=digest(
            (src / "scripts/run_qwen_t2v_length_study.sh").read_bytes()
        ),
        deterministic_inference=True,
    )
    fingerprint = digest(packed(signature).encode())
    manifest = dict(
        fingerprint=fingerprint, signature=signature, count=len(rows), created=utc()
    )
    dest = root
    mpath = dest / "manifest.json"
    if mpath.exists():
        assert read(mpath)["fingerprint"] == fingerprint, (
            "Inputs/model/code/environment changed; use a new results directory"
        )
    else:
        save(mpath, manifest)
    save(dest / "provenance" / "source_state.json", source_state)
    (dest / "provenance" / "source.diff").write_bytes(source_diff)
    atomic_text(dest / "inputs.jsonl", "".join(packed(x) + "\n" for x in rows))
    print(f"Prepared {len(rows)} prompts; fingerprint={fingerprint}", flush=True)
    print(
        f"Model={MODEL_SPECS[args.model_family]['repo_id']}; "
        f"system={args.system_mode}; decoding={signature['decoding']}",
        flush=True,
    )
    return rows, tok, special, eos, fingerprint


def checked_records(root, rows, fingerprint):
    expected = {x["id"]: x for x in rows}
    signature = read(root / "manifest.json")["signature"]
    special, eos = set(signature["special_ids"]), set(signature["eos_ids"])
    found = {}
    for path in sorted((root / "records").glob("*.json")):
        r = read(path)
        assert r["id"] in expected and r["id"] not in found
        assert r["fingerprint"] == fingerprint and r["status"] in VALID, str(path)
        for k in (
            "raw_token_ids",
            "input_token_ids",
            "raw_prompt",
            "raw_tokens",
            "input_tokens",
            "category",
            "source_line",
        ):
            assert r[k] == expected[r["id"]][k], (str(path), k)
        original = expected[r["id"]]
        assert ("sampling_seed" in r) == ("sampling_seed" in original), str(path)
        assert r.get("sampling_seed") == original.get("sampling_seed"), str(path)
        assert all(type(t) is int for t in r["output_token_ids"])
        assert r["generated_tokens"] == len(r["output_token_ids"])
        assert r["response_tokens"] == sum(
            t not in special for t in r["output_token_ids"]
        )
        if r["status"] == "eos":
            assert r["finish_reason"]["type"] == "stop"
            assert r["output_token_ids"][-1] == r["finish_reason"]["matched"] in eos
        else:
            assert r["finish_reason"]["type"] == "length"
        found[r["id"]] = r
    return found


def statistics(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def q(p):
        position = (len(values) - 1) * p
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return dict(
        count=len(values),
        mean=sum(values) / len(values),
        minimum=values[0],
        maximum=values[-1],
        **{f"p{p}": q(p / 100) for p in (50, 90, 95, 99)},
        above={
            str(n): {
                "count": sum(x > n for x in values),
                "fraction": sum(x > n for x in values) / len(values),
            }
            for n in (512, 1024, 2048, 4096, 8192)
        },
    )


def markdown_table(headers, rows):
    """Markdown with display-width padding for readable Chinese terminal output."""
    cells = [[str(c).replace("|", r"\|") for c in row] for row in [headers, *rows]]

    def width(text):
        return sum(
            0
            if unicodedata.combining(c)
            else 2
            if unicodedata.east_asian_width(c) in "WF"
            else 1
            for c in text
        )

    widths = [max(3, *(width(row[i]) for row in cells)) for i in range(len(headers))]

    def line(row):
        return (
            "| "
            + " | ".join(c + " " * (w - width(c)) for c, w in zip(row, widths))
            + " |"
        )

    return "\n".join(
        [
            line(cells[0]),
            "| " + " | ".join("-" * w for w in widths) + " |",
            *(line(row) for row in cells[1:]),
        ]
    )


def experiment_metadata(signature):
    # Schema 2 results predate the explicit family/system/decoding fields.
    return {
        "model": signature.get("model_revision", {}).get("repo_id", "未记录型号"),
        "system_mode": signature.get(
            "system_mode", "helpful" if signature.get("system") else "none"
        ),
        "decoding": signature.get(
            "decoding",
            "greedy"
            if signature.get("sampling", {}).get("temperature") == 0
            else "sampling",
        ),
    }


def experiment_description(experiment):
    if not experiment:
        return "实验设置见本目录 manifest.json。"
    system = "无 system" if experiment["system_mode"] == "none" else "helpful system"
    decoding = "greedy" if experiment["decoding"] == "greedy" else "随机采样"
    return f"模型：**{experiment['model']}**；输入：**{system}**；解码：**{decoding}**。"


def summary_markdown(summary):
    def stat_row(label, stats):
        return [label, str(stats["count"])] + [
            (f"{stats[k]:,.0f}" if k == "maximum" else f"{stats[k]:,.2f}")
            if k in stats
            else "—"
            for k in ("mean", "p50", "p90", "p95", "p99", "maximum")
        ]

    headers = ["口径", "条数", "均值", "p50", "p90", "p95", "p99", "最大值"]
    groups = summary["categories"]
    overall = summary["all"]
    parts = [
        "# Qwen 普通对话长度统计",
        experiment_description(summary.get("experiment")),
        f"已保存 **{summary['recorded']}/{summary['expected']}** 条回复：自然结束 **{summary['naturally_finished']}** 条，达到长度限制 **{summary['length_limited']}** 条；未完成 **{summary['missing']}** 条。",
        f"统计更新时间：{summary['generated_at']}。",
        "raw 是原始 prompt 长度；input 是实际送入模型的长度，包含对话标记及所选模式的 system（无 system 模式不包含）；response 按实际生成 token IDs 计数，扣除终止及特殊 token。以下长度单位均为 tokens。",
        "## 总体长度",
        "只统计已保存的有效记录；未完成请求不进入长度统计。",
        markdown_table(
            headers,
            [
                stat_row("raw", overall["raw_tokens"]),
                stat_row("input", overall["input_tokens"]),
                stat_row("response（全部已保存）", overall["observed_response_tokens"]),
                stat_row("response（仅自然结束）", overall["eos_only_response_tokens"]),
            ],
        ),
        "## 各类别进度与输入长度",
        markdown_table(
            ["类别", "自然结束", "长度限制", "raw 均值", "input 均值"],
            [
                [
                    CATEGORY_LABELS.get(c, c),
                    g["eos_only_response_tokens"]["count"],
                    g["length_limited"],
                    *[
                        f"{g[k]['mean']:.2f}" if g[k]["count"] else "—"
                        for k in ("raw_tokens", "input_tokens")
                    ],
                ]
                for c, g in groups.items()
            ],
        ),
        "## 各类别回复长度（全部已保存）",
        markdown_table(
            ["类别", *headers[1:]],
            [
                stat_row(CATEGORY_LABELS.get(c, c), g["observed_response_tokens"])
                for c, g in groups.items()
            ],
        ),
    ]
    if summary["length_limited"]:
        parts.extend(
            [
                "## 各类别回复长度（仅自然结束）",
                markdown_table(
                    ["类别", *headers[1:]],
                    [
                        stat_row(
                            CATEGORY_LABELS.get(c, c), g["eos_only_response_tokens"]
                        )
                        for c, g in groups.items()
                    ],
                ),
            ]
        )

    def threshold(stats, n):
        entry = stats.get("above", {}).get(str(n))
        return [entry["count"], f"{entry['fraction']:.2%}"] if entry else [0, "—"]

    parts.extend(
        [
            "## 长回复数量与比例",
            "“超过”为严格大于阈值；两组比例分别以全部已保存回复、自然结束回复为分母。",
            markdown_table(
                ["超过 tokens", "全部数量", "全部占比", "自然结束数量", "自然结束占比"],
                [
                    [
                        f"{n:,}",
                        *threshold(overall["observed_response_tokens"], n),
                        *threshold(overall["eos_only_response_tokens"], n),
                    ]
                    for n in (512, 1024, 2048, 4096, 8192)
                ],
            ),
            "达到长度限制的回复只记录已观测长度，其自然结束长度尚未完整观测；自然结束统计不包含这些回复。",
        ]
    )
    if summary["missing"]:
        counts = {}
        for status in summary["missing_states"].values():
            counts[status] = counts.get(status, 0) + 1
        parts.extend(
            [
                "## 未完成请求",
                markdown_table(
                    ["状态", "条数"],
                    [[STATUS_LABELS.get(s, s), n] for s, n in sorted(counts.items())],
                ),
                "重新执行同一启动命令可续跑；逐条状态见 lengths.csv。",
            ]
        )
    parts.append(
        "完整 prompt 和回复见 responses.md；简洁数据见 lengths.csv；实际 token IDs、完整精度和诊断信息保留在 JSON/JSONL 中。"
    )
    return "\n\n".join(parts) + "\n"


def write_readable_responses(root, records, expected, experiment=None):
    def fenced(text):
        longest = max((len(m.group()) for m in re.finditer(r"`+", text)), default=0)
        fence = "`" * max(3, longest + 1)
        return fence + "text\n" + text + ("" if text.endswith("\n") else "\n") + fence

    parts = [
        "# 原始 prompt 与完整回复",
        experiment_description(experiment),
        f"已保存 {len(records)}/{expected} 条，按 response tokens 从多到少排列；内容不截短。",
        "raw 不含对话模板；input 为实际发送的完整输入长度；response 按生成 token IDs 扣除终止及特殊 token 后计数。",
    ]
    for i, r in enumerate(
        sorted(records, key=lambda r: (-r["response_tokens"], r["id"])), 1
    ):
        parts.extend(
            [
                f"## {i}. {CATEGORY_LABELS.get(r['category'], r['category'])} · {r['id']}",
                f"raw：**{r['raw_tokens']}** · input：**{r['input_tokens']}** · response：**{r['response_tokens']}** tokens · 耗时：{r['elapsed_s']:.3f} s · 状态：{STATUS_LABELS[r['status']]}（{r['status']}）。",
            ]
        )
        if r["status"] == "length_limit":
            parts.append(
                "此回复触及长度限制，以下为已生成全文；自然结束长度尚未完整观测。"
            )
        parts.extend(
            [
                "### 原始 prompt",
                fenced(r["raw_prompt"]),
                "### 模型回复",
                fenced(r["response_text"]),
            ]
        )
    if not records:
        parts.append("暂无有效回复；请求进度见 lengths.csv。")
    atomic_text(root / "responses.md", "\n\n".join(parts) + "\n")


def export_results(root):
    dest = root
    manifest = read(dest / "manifest.json")
    rows = [
        json.loads(s)
        for s in (dest / "inputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    found = checked_records(root, rows, manifest["fingerprint"])
    records = [found[x["id"]] for x in rows if x["id"] in found]
    atomic_text(dest / "responses.jsonl", "".join(packed(x) + "\n" for x in records))
    fields = [
        "id",
        "category",
        "source_line",
        "raw_tokens",
        "input_tokens",
        "response_tokens",
        "elapsed_s",
        "status",
    ]
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    missing_states = {}
    for row in rows:
        if row["id"] in found:
            record = found[row["id"]]
            w.writerow({**record, "elapsed_s": f"{record['elapsed_s']:.3f}"})
        else:
            attempt = dest / "attempts" / (row["id"] + ".json")
            status = read(attempt)["status"] if attempt.exists() else "pending"
            if status == "running":
                status = "unfinished"
            missing_states[row["id"]] = status
            w.writerow({**row, "status": status})
    atomic_text(dest / "lengths.csv", buf.getvalue())
    summary = dict(
        experiment=experiment_metadata(manifest["signature"]),
        expected=len(rows),
        recorded=len(records),
        missing=len(rows) - len(records),
        naturally_finished=sum(x["status"] == "eos" for x in records),
        length_limited=sum(x["status"] == "length_limit" for x in records),
        missing_states=missing_states,
        explanation="All-record reply lengths are observed lengths. Length-limited outputs are right-censored lower bounds; EOS-only statistics condition on natural completion.",
        all={},
        categories={},
        generated_at=utc(),
    )

    def group(rr):
        return dict(
            raw_tokens=statistics([r["raw_tokens"] for r in rr]),
            input_tokens=statistics([r["input_tokens"] for r in rr]),
            observed_response_tokens=statistics([r["response_tokens"] for r in rr]),
            eos_only_response_tokens=statistics(
                [r["response_tokens"] for r in rr if r["status"] == "eos"]
            ),
            length_limited=sum(r["status"] == "length_limit" for r in rr),
        )

    summary["all"] = group(records)
    summary["categories"] = {
        c: group([r for r in records if r["category"] == c])
        for c in sorted({r["category"] for r in rows})
    }
    save(dest / "summary.json", summary)
    atomic_text(dest / "summary.md", summary_markdown(summary))
    write_readable_responses(dest, records, len(rows), summary["experiment"])
    atomic_text(
        dest / "longest_responses.jsonl",
        "".join(
            packed(r) + "\n"
            for r in sorted(records, key=lambda r: (-r["response_tokens"], r["id"]))
        ),
    )
    save(dest / "missing_ids.json", [r["id"] for r in rows if r["id"] not in found])
    print(
        f"已导出 {len(records)}/{len(rows)} 条：自然结束 {summary['naturally_finished']}，达到长度限制 {summary['length_limited']}。",
        flush=True,
    )
    print(
        f"统计：{dest / 'summary.md'}\n全文：{dest / 'responses.md'}\n数据：{dest / 'lengths.csv'}",
        flush=True,
    )
    return summary


def owned_group(pgid, nonce):
    """A session created by this run, also carrying its unique environment marker."""
    import psutil

    for p in psutil.process_iter(["pid"]):
        try:
            if os.getpgid(p.pid) == pgid and os.getsid(p.pid) == pgid:
                if p.environ().get("QWEN_LENGTH_RUN") == nonce:
                    return True
        except (ProcessLookupError, PermissionError, psutil.Error):
            pass
    return False


def cleanup_groups(servers, nonce):
    for s in servers:
        if owned_group(s["pid"], nonce):
            try:
                os.killpg(s["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    end = time.monotonic() + 20
    while time.monotonic() < end and any(owned_group(s["pid"], nonce) for s in servers):
        time.sleep(0.25)
    for s in servers:
        if owned_group(s["pid"], nonce):
            try:
                os.killpg(s["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def stop(root):
    import psutil

    p = root / "controller.json"
    if not p.exists():
        print("No controller recorded.")
        return
    state = read(p)
    try:
        controller = psutil.Process(state["pid"])
        alive = abs(controller.create_time() - state["create_time"]) < 0.01
    except psutil.Error:
        alive = False
    if alive:
        save(root / "stop.request", {"at": utc(), "run_id": state["run_id"]})
        print(
            "Stop requested; completed records are retained. Wait for status.json and GPU release."
        )
    else:
        cleanup_groups(state.get("servers", []), state["nonce"])
        print(
            "Controller absent; cleaned only surviving server groups from that recorded run."
        )


def make_record(row, result, tok, special, eos, fingerprint, gpu, run_id, duration):
    meta = result.get("meta_info", {})
    reason = meta.get("finish_reason", {})
    ids = result.get("output_ids")
    assert isinstance(ids, list) and all(type(i) is int for i in ids), (
        "Missing actual output IDs"
    )
    assert meta.get("prompt_tokens") == row["input_tokens"], (
        "Input token count mismatch"
    )
    assert meta.get("completion_tokens") == len(ids), "Generated token count mismatch"
    kind, matched = reason.get("type"), reason.get("matched")
    if kind == "stop" and isinstance(matched, int) and matched in eos:
        assert ids and ids[-1] == matched, "EOS missing from returned IDs"
        status = "eos"
    elif kind == "length":
        status = "length_limit"
    else:
        raise ValueError(f"Unexpected finish reason: {reason}")
    response = tok.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return dict(
        **row,
        fingerprint=fingerprint,
        status=status,
        finish_reason=reason,
        output_token_ids=ids,
        generated_tokens=len(ids),
        response_tokens=sum(i not in special for i in ids),
        response_retokenized_tokens=len(tok.encode(response, add_special_tokens=False)),
        end_token_id=ids[-1] if ids else None,
        response_text=response,
        server_text=result.get("text"),
        meta_info=meta,
        gpu=gpu,
        run_id=run_id,
        elapsed_s=duration,
        completed_at=utc(),
    )


async def execute(args, rows, tok, special, eos, fingerprint, run_dir, state):
    import aiohttp

    model = args.model_dir
    found = checked_records(args.output_dir, rows, fingerprint)
    pending = [r for r in rows if r["id"] not in found]
    random.Random(20260912).shuffle(pending)
    if not pending:
        return
    hardware = cmd(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,uuid",
        "--format=csv,noheader",
    )
    atomic_text(run_dir / "gpus.csv", hardware + "\n")
    assert len(hardware.splitlines()) == 8 and all(
        "H100" in x for x in hardware.splitlines()
    ), "Expected 8 H100 GPUs"
    occupied = cmd(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    )
    atomic_text(run_dir / "existing_gpu_processes.csv", occupied + "\n")
    assert not occupied.strip(), (
        "GPUs have active compute processes; leave other tasks untouched and run when GPUs are free"
    )
    # Fail before launching anything if one of our eight local ports is in use.
    for port in range(args.base_port, args.base_port + 8):
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))
    processes, handles, worker_tasks = [], [], []
    done_before, new_done, new_tokens = len(found), 0, 0
    begun = time.monotonic()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)
    status_path = args.output_dir / "status.json"

    def status(phase, active=None):
        elapsed = time.monotonic() - begun
        save(
            status_path,
            dict(
                phase=phase,
                run_id=state["run_id"],
                timestamp=utc(),
                completed=done_before + new_done,
                expected=len(rows),
                new_completed=new_done,
                active=active,
                elapsed_s=elapsed,
                completed_response_tokens_per_s=new_tokens / max(elapsed, 1),
                note="Completed-request token rate only; unfinished long responses are not counted yet.",
            ),
        )

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    connector = aiohttp.TCPConnector(limit=8 * args.concurrency + 16, limit_per_host=0)
    try:
        for gpu in range(8):
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=str(gpu),
                QWEN_LENGTH_RUN=state["nonce"],
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="4",
                SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
            env.pop("SGLANG_MAX_NEW_TOKENS_LIMIT", None)
            command = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.base_port + gpu),
                "--tp-size",
                "1",
                "--dtype",
                "bfloat16",
                "--sampling-defaults",
                "openai",
                "--context-length",
                str(CONTEXT),
                "--mem-fraction-static",
                str(args.mem_fraction),
                "--max-running-requests",
                str(args.concurrency),
                "--cuda-graph-max-bs-decode",
                str(args.concurrency),
                "--enable-deterministic-inference",
                "--random-seed",
                str(20260912 + gpu),
                "--log-level-http",
                "warning",
            ]
            log = (run_dir / f"server_gpu{gpu}.log").open("wb")
            handles.append(log)
            p = subprocess.Popen(
                command,
                env=env,
                cwd=args.output_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            processes.append(p)
            state["servers"].append(
                dict(pid=p.pid, gpu=gpu, port=args.base_port + gpu, command=command)
            )
            save(args.output_dir / "controller.json", state)
        status("starting_servers")
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            ready = set()
            deadline = time.monotonic() + args.startup_timeout
            while len(ready) < 8:
                if stop_event.is_set() or (args.output_dir / "stop.request").exists():
                    raise InterruptedError("Stop requested")
                for gpu, p in enumerate(processes):
                    if p.poll() is not None:
                        raise RuntimeError(
                            f"GPU {gpu} server exited: read {run_dir}/server_gpu{gpu}.log"
                        )
                    if gpu in ready:
                        continue
                    try:
                        async with session.get(
                            f"http://127.0.0.1:{args.base_port + gpu}/server_info",
                            timeout=aiohttp.ClientTimeout(total=2),
                        ) as resp:
                            if resp.status == 200:
                                info = await resp.json()
                                assert (
                                    info.get("tp_size") == 1
                                    and info.get("context_length") == CONTEXT
                                    and info.get("sampling_defaults") == "openai"
                                    and not info.get("preferred_sampling_params")
                                )
                                save(run_dir / f"server_info_gpu{gpu}.json", info)
                                ready.add(gpu)
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Server startup timed out; inspect logs. No prompt was truncated."
                    )
                print(f"Servers ready: {len(ready)}/8", flush=True)
                if len(ready) < 8:
                    await asyncio.sleep(2)
            queue = asyncio.Queue()
            for r in pending:
                queue.put_nowait(r)
            active = {}

            async def worker(gpu):
                nonlocal new_done, new_tokens
                while not stop_event.is_set():
                    try:
                        row = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    start = time.monotonic()
                    active[row["id"]] = {"gpu": gpu, "started": start}
                    attempt_path = args.output_dir / "attempts" / (row["id"] + ".json")
                    attempt = dict(
                        id=row["id"],
                        run_id=state["run_id"],
                        gpu=gpu,
                        status="running",
                        started_at=utc(),
                    )
                    save(attempt_path, attempt)
                    sampling = {
                        **sampling_parameters(args.greedy),
                        "stop_token_ids": eos,
                    }
                    if not args.greedy:
                        sampling["sampling_seed"] = row["sampling_seed"]
                    payload = dict(
                        input_ids=row["input_token_ids"],
                        sampling_params=sampling,
                        stream=False,
                        return_logprob=False,
                        rid=f"{state['run_id']}-{row['id']}",
                    )
                    result = None
                    try:
                        async with session.post(
                            f"http://127.0.0.1:{args.base_port + gpu}/generate",
                            json=payload,
                        ) as resp:
                            body = await resp.text()
                            if resp.status != 200:
                                raise RuntimeError(f"HTTP {resp.status}: {body[:3000]}")
                            result = json.loads(body)
                        record = make_record(
                            row,
                            result,
                            tok,
                            special,
                            eos,
                            fingerprint,
                            gpu,
                            state["run_id"],
                            time.monotonic() - start,
                        )
                        save(
                            args.output_dir / "records" / (row["id"] + ".json"), record
                        )
                        new_done += 1
                        new_tokens += record["response_tokens"]
                    except asyncio.CancelledError:
                        save(
                            attempt_path,
                            {**attempt, "status": "interrupted", "ended_at": utc()},
                        )
                        raise
                    except Exception as exc:
                        save(
                            attempt_path,
                            {
                                **attempt,
                                "status": "failed",
                                "error": repr(exc),
                                "ended_at": utc(),
                            },
                        )
                        save(
                            run_dir / "errors" / (row["id"] + ".json"),
                            dict(
                                id=row["id"],
                                error=repr(exc),
                                raw_result=result,
                                at=utc(),
                            ),
                        )
                        raise
                    finally:
                        active.pop(row["id"], None)
                        queue.task_done()

            worker_tasks = [
                asyncio.create_task(worker(g))
                for _ in range(args.concurrency)
                for g in range(8)
            ]
            last_report = 0
            while True:
                for task in worker_tasks:
                    if task.done() and not task.cancelled() and task.exception():
                        raise task.exception()
                if all(t.done() for t in worker_tasks):
                    break
                if stop_event.is_set() or (args.output_dir / "stop.request").exists():
                    raise InterruptedError("Stop requested")
                for gpu, p in enumerate(processes):
                    if p.poll() is not None:
                        raise RuntimeError(
                            f"GPU {gpu} server exited; completed records retained"
                        )
                if time.monotonic() - last_report >= 10:
                    last_report = time.monotonic()
                    status("running", len(active))
                    oldest = max(
                        (time.monotonic() - x["started"] for x in active.values()),
                        default=0,
                    )
                    print(
                        f"{utc()} completed={done_before + new_done}/1400 active={len(active)} oldest_active={oldest:.1f}s",
                        flush=True,
                    )
                await asyncio.sleep(1)
            await asyncio.gather(*worker_tasks)
    finally:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        try:
            status("cleaning_up", 0)
        finally:
            cleanup_groups(state["servers"], state["nonce"])
        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for handle in handles:
            handle.close()


def run(args):
    import psutil

    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=True)
    with output_lock(dest):
        if (dest / "controller.json").exists():
            previous = read(dest / "controller.json")
            if any(
                owned_group(s["pid"], previous["nonce"])
                for s in previous.get("servers", [])
            ):
                raise SystemExit(
                    "Previous server groups remain: run the stop command before resuming."
                )
        (dest / "stop.request").unlink(missing_ok=True)
        rows, tok, special, eos, fingerprint = prepare(args)
        state = dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            run_id=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + "-"
            + uuid.uuid4().hex[:8],
            nonce=uuid.uuid4().hex,
            servers=[],
            started=utc(),
            concurrency=args.concurrency,
            mem_fraction=args.mem_fraction,
            base_port=args.base_port,
            python=sys.executable,
            repository=str(REPO),
            model_family=args.model_family,
            system_mode=args.system_mode,
            decoding="greedy" if args.greedy else "sampling",
            model_dir=str(args.model_dir),
            data_dir=str(args.data_dir),
            output_dir=str(args.output_dir),
        )
        save(dest / "controller.json", state)
        run_dir = dest / "runs" / state["run_id"]
        run_dir.mkdir(parents=True)
        start = time.monotonic()
        phase, error, exit_code = "completed", None, 0
        try:
            asyncio.run(
                execute(args, rows, tok, special, eos, fingerprint, run_dir, state)
            )
        except (InterruptedError, KeyboardInterrupt) as exc:
            phase, error, exit_code = "stopped", str(exc), 130
        except Exception:
            import traceback

            phase, error, exit_code = "failed", traceback.format_exc(), 1
            print(error, file=sys.stderr, flush=True)
        finally:
            state.update(
                phase=phase,
                error=error,
                finished=utc(),
                total_elapsed_s=time.monotonic() - start,
            )
            save(run_dir / "run.json", state)
            try:
                summary = export_results(args.output_dir)
                if summary["recorded"] != 1400 and phase == "completed":
                    phase, exit_code = "incomplete", 1
                if phase == "completed" and summary["length_limited"]:
                    phase = "completed_with_length_limits"
                state.update(
                    phase=phase,
                    completed=summary["recorded"],
                    naturally_finished=summary["naturally_finished"],
                    length_limited=summary["length_limited"],
                )
                save(run_dir / "run.json", state)
                save(
                    dest / "status.json",
                    dict(
                        phase=phase,
                        run_id=state["run_id"],
                        error=error,
                        total_elapsed_s=state["total_elapsed_s"],
                        completed=summary["recorded"],
                        naturally_finished=summary["naturally_finished"],
                        length_limited=summary["length_limited"],
                        expected=1400,
                        finished=utc(),
                    ),
                )
            except Exception:
                save(
                    dest / "status.json",
                    dict(phase="export_failed", run_id=state["run_id"], error=error),
                )
                raise
        raise SystemExit(exit_code)


def report(root, top, preview_chars):
    summary = read(root / "summary.json")
    if "experiment" not in summary and (root / "manifest.json").is_file():
        summary["experiment"] = experiment_metadata(
            read(root / "manifest.json")["signature"]
        )
    print(summary_markdown(summary), end="")
    with (root / "longest_responses.jsonl").open(encoding="utf-8") as f:
        for _, line in zip(range(top), f):
            r = json.loads(line)
            print(
                f"\n{r['id']}：raw={r['raw_tokens']}，response={r['response_tokens']}，{STATUS_LABELS[r['status']]}"
            )
            print("原始 prompt：", r["raw_prompt"])
            print("回复预览：", r["response_text"][:preview_chars])
    print(f"\n完整回复文件：{root / 'responses.md'}")


def set_runtime_directories(root):
    cache = root / "cache"
    locations = {
        "TMPDIR": root / "tmp",
        "HF_HOME": cache / "huggingface",
        "XDG_CACHE_HOME": cache,
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
    }
    for key, path in locations.items():
        if path.resolve().is_relative_to(REPO):
            raise ValueError(
                f"Runtime cache must be outside the source checkout: {path}"
            )
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    os.environ.pop("SGLANG_MAX_NEW_TOKENS_LIMIT", None)
    sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=["prepare", "run", "export", "stop", "status", "report"]
    )
    p.add_argument(
        "--model-family",
        choices=tuple(MODEL_SPECS),
        default="qwen2.5-7b",
        help="Qwen checkpoint identity; existing calls default to 7B",
    )
    p.add_argument(
        "--system-mode",
        choices=("helpful", "none"),
        default="helpful",
        help="none omits the entire system turn, including Qwen's automatic default",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Use temperature=0 without sampling filters, penalties or sampling seeds",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("STUDY_ROOT", "~/qwen_t2v_length_study")),
        help="External experiment root for default paths and caches (STUDY_ROOT)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=os.environ.get("MODEL_DIR"),
        help="Already downloaded Qwen model directory (MODEL_DIR)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("DATA_DIR"),
        help="T2V-CompBench V2 Git checkout, containing prompts/ (DATA_DIR)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=os.environ.get("RESULT_DIR"),
        help="Results directory outside the source checkout (RESULT_DIR)",
    )
    p.add_argument(
        "--model-revision",
        help="Optional 40-character HF commit override; otherwise read local download metadata, or record unknown",
    )
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--mem-fraction", type=float, default=0.85)
    p.add_argument("--base-port", type=int, default=31000)
    p.add_argument("--startup-timeout", type=int, default=1800)
    p.add_argument(
        "--top",
        type=int,
        default=0,
        help="Optional longest-response previews; report prints only tables by default",
    )
    p.add_argument("--preview-chars", type=int, default=500)
    args = p.parse_args()
    if not __debug__:
        p.error(
            "Run without -O or PYTHONOPTIMIZE; integrity assertions must stay enabled"
        )
    args.root = args.root.expanduser().resolve()
    model_name = MODEL_SPECS[args.model_family]["repo_id"].split("/", 1)[1]
    args.model_dir = (
        (args.model_dir or args.root / "models" / model_name)
        .expanduser()
        .resolve()
    )
    args.data_dir = (
        (args.data_dir or args.root / "data/T2V-CompBench").expanduser().resolve()
    )
    legacy_defaults = (
        args.model_family == "qwen2.5-7b"
        and args.system_mode == "helpful"
        and not args.greedy
    )
    result_name = (
        "results"
        if legacy_defaults
        else (
            f"results_{args.model_family}_{args.system_mode}_"
            f"{'greedy' if args.greedy else 'sampling'}"
        )
    )
    args.output_dir = (args.output_dir or args.root / result_name).expanduser().resolve()
    if args.output_dir.is_relative_to(REPO):
        p.error("--output-dir must be outside the source checkout")
    if not (1 <= args.concurrency <= 256 and 0.5 <= args.mem_fraction < 0.95):
        p.error("Use concurrency 1..256 and mem-fraction >=0.5 and <0.95")
    if not (
        1024 <= args.base_port <= 65528
        and args.startup_timeout > 0
        and args.top >= 0
        and args.preview_chars >= 0
    ):
        p.error("Invalid port, timeout, or report arguments")
    if args.action in {"prepare", "run", "export"} and sys.platform != "linux":
        p.error(
            "prepare/run/export use Linux process and file locking; execute on the AWS host"
        )
    if args.action in {"prepare", "run"}:
        if args.model_revision is not None and not re.fullmatch(
            r"[0-9a-f]{40}", args.model_revision
        ):
            p.error("--model-revision, when supplied, must be a 40-character HF commit")
        set_runtime_directories(args.output_dir)
    if args.action == "run":
        run(args)
    elif args.action == "prepare":
        with output_lock(args.output_dir):
            prepare(args)
    elif args.action == "export":
        with output_lock(args.output_dir):
            export_results(args.output_dir)
    elif args.action == "stop":
        stop(args.output_dir)
    elif args.action == "report":
        report(args.output_dir, args.top, args.preview_chars)
    else:
        path = args.output_dir / "status.json"
        print(path.read_text(encoding="utf-8") if path.exists() else "No status yet.")
        print("Saved records:", len(list((args.output_dir / "records").glob("*.json"))))


if __name__ == "__main__":
    main()
