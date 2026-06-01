"""Generate ShiftServe CLI and cookbook markdown artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path


def _repo_root() -> Path:
    # docs.py -> shiftserve -> sglang -> python -> repo root
    return Path(__file__).resolve().parents[3]


def _canonical_doc(name: str) -> Path:
    return _repo_root() / "docs" / name


def _copy_or_write(path: str | Path, canonical_name: str, fallback: str) -> None:
    target = Path(path)
    source = _canonical_doc(canonical_name)
    if source.exists():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        target.write_text(fallback, encoding="utf-8")


def generate_cli_markdown(path: str | Path) -> None:
    fallback = """# ShiftServe CLI 手册

请优先使用仓库中的 `docs/shiftserve_cli.md`。该 fallback 文档只保留最小命令：

```bash
python -m sglang.shiftserve.cli validate-config \
  --deployment-json ./configs/deployment.json \
  --profile-json ./configs/profile.json \
  --traffic-json ./configs/traffic.json

python -m sglang.shiftserve.cli simulate \
  --traffic-json ./configs/traffic.json \
  --mode no_flip \
  --weighted-schedule false \
  --out-dir ./outputs/no_flip_round_robin
```
"""
    _copy_or_write(path, "shiftserve_cli.md", fallback)


def generate_cookbook_markdown(path: str | Path) -> None:
    fallback = """# ShiftServe 架构与代码 Cookbook

请优先使用仓库中的 `docs/shiftserve_cookbook.md`。该 fallback 文档只说明核心入口：

- `config.py`：JSON dataclasses 和 loader。
- `scheduler.py`：round-robin/weighted scheduling 和 flip hysteresis。
- `launcher.py`：launch defaults、fixed transfer buffer、rank0 broadcast args。
- `router.py`：PE->TE->DiT->VAE request lifecycle。
- `metrics.py`：event logs 和 summary CSV。
"""
    _copy_or_write(path, "shiftserve_cookbook.md", fallback)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成 ShiftServe 中文 markdown 文档")
    parser.add_argument("--output-dir", default=str(Path.home() / "Desktop"))
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generate_cli_markdown(out / "shiftserve_cli.md")
    generate_cookbook_markdown(out / "shiftserve_cookbook.md")


if __name__ == "__main__":
    main()
