from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from generate_dataset import _bucket_key
from tools_schema import SPIDOR_TOOLS


OUT_DIR = Path("data")
CMDS_PATH = OUT_DIR / "cmds.jsonl"
OFFTOPIC_TXT_PATH = OUT_DIR / "offtopic_phrases.txt"
OFFTOPIC_JSONL_PATH = OUT_DIR / "offtopic.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
VAL_PATH = OUT_DIR / "val.jsonl"
VAL_RATIO = 0.15
SEED = 42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def build_offtopic() -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for line in OFFTOPIC_TXT_PATH.read_text(encoding="utf-8").splitlines():
        q = " ".join(line.strip().split())
        if q and q not in seen:
            seen.add(q)
            rows.append({"query": q, "tools": SPIDOR_TOOLS, "reasoning": "-", "answers": []})
    write_jsonl(OFFTOPIC_JSONL_PATH, rows)
    return rows


rng = random.Random(SEED)
offtopic_rows = build_offtopic()
cmd_rows = read_jsonl(CMDS_PATH)

offtopic_queries = {r["query"] for r in offtopic_rows}
cmd_rows = [r for r in cmd_rows if r["query"] not in offtopic_queries]

buckets: dict[str, list[dict[str, Any]]] = {}
for r in cmd_rows:
    buckets.setdefault(_bucket_key(r), []).append(r)
buckets["offtopic"] = offtopic_rows

train_rows: list[dict[str, Any]] = []
val_rows: list[dict[str, Any]] = []
for items in buckets.values():
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * VAL_RATIO))) if len(items) > 1 else 0
    val_rows.extend(items[:n_val])
    train_rows.extend(items[n_val:])

rng.shuffle(train_rows)
rng.shuffle(val_rows)

write_jsonl(TRAIN_PATH, train_rows)
write_jsonl(VAL_PATH, val_rows)
print(
    f"cmds={len(cmd_rows)}, offtopic={len(offtopic_rows)} | "
    f"train={len(train_rows)}, val={len(val_rows)}"
)
