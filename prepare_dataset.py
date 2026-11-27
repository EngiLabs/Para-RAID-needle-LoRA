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
TRAIN_FILTER_PATH = OUT_DIR / "train_filter.jsonl"
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

cmd_buckets: dict[str, list[dict[str, Any]]] = {}
for r in cmd_rows:
    cmd_buckets.setdefault(_bucket_key(r), []).append(r)

train_cmds: list[dict[str, Any]] = []
val_cmds: list[dict[str, Any]] = []
for items in cmd_buckets.values():
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * VAL_RATIO))) if len(items) > 1 else 0
    val_cmds.extend(items[:n_val])
    train_cmds.extend(items[n_val:])

rng.shuffle(offtopic_rows)
n_off_val = max(1, int(round(len(offtopic_rows) * VAL_RATIO))) if len(offtopic_rows) > 1 else 0
val_offtopic = offtopic_rows[:n_off_val]
train_offtopic = offtopic_rows[n_off_val:]

train_filter_rows = train_cmds + train_offtopic
val_rows = val_cmds + val_offtopic

rng.shuffle(train_cmds)
rng.shuffle(train_filter_rows)
rng.shuffle(val_rows)

write_jsonl(TRAIN_PATH, train_cmds)
write_jsonl(TRAIN_FILTER_PATH, train_filter_rows)
write_jsonl(VAL_PATH, val_rows)
print(
    f"cmds={len(cmd_rows)}, offtopic={len(offtopic_rows)} | "
    f"train.jsonl (Needle 3 cmds)={len(train_cmds)}, "
    f"train_filter.jsonl (Layer 1)={len(train_filter_rows)}, "
    f"val.jsonl={len(val_rows)}"
)
