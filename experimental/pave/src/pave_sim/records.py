"""Versioned result storage; no files are opened until explicitly requested."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

SCHEMA_VERSION = 1


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with text_stream(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with text_stream(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def text_stream(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


class Journal:
    def __init__(self, path: Path, run_id: str):
        self.stream = text_stream(path, "wt")
        self.run_id, self.sequence = run_id, 0

    def emit(self, event: str, **fields) -> None:
        record = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "sequence": self.sequence, "event": event, **fields}
        self.sequence += 1
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def close(self) -> None:
        self.stream.close()
