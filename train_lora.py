from __future__ import annotations

import os
import re
from pathlib import Path

from cactus.cli.logging import configure
from cactus.cli.parser import parser
from cactus.model.build import build_main
from cactus.model.finetune import finetune_local

DATASET_PATH = Path("data/train.jsonl")
ADAPTER_PATH = Path("models/spidor_lora.safetensors")
OUTPUT_CACT = Path("models/spidor_needle3_ru.cact")
LOCAL_BASE_CACT = Path("checkpoints/needle3.cact")

EPOCHS = 12
BATCH_SIZE = 4
LEARNING_RATE = "1e-4"
LORA_RANK = 32
LORA_ALPHA = 64
MAX_LEN = 1280

STEP_RE = re.compile(r"\bstep\s+(\d+)/(\d+)\b")
EPOCH_RE = re.compile(r"\bepoch\s+(\d+)/(\d+)\b")


def report_progress(msg: str) -> None:
    m = STEP_RE.search(msg) or (EPOCH_RE.search(msg) if "loss" in msg else None)
    if m and int(m.group(2)) > 0:
        pct = 100.0 * int(m.group(1)) / int(m.group(2))
        print(f"  {'progress':<9} {pct:5.1f}%", flush=True)


if not DATASET_PATH.exists():
    raise FileNotFoundError(f"No dataset {DATASET_PATH}")

ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)
if LOCAL_BASE_CACT.exists():
    os.environ.setdefault("CACTUS_NEEDLE_WEIGHTS", str(LOCAL_BASE_CACT.resolve()))

configure()
cli = parser()

finetune_args = cli.parse_args([
    "finetune", str(DATASET_PATH),
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--lr", LEARNING_RATE,
    "--lora-rank", str(LORA_RANK),
    "--lora-alpha", str(LORA_ALPHA),
    "--max-len", str(MAX_LEN),
    "--out", str(ADAPTER_PATH),
])
finetune_args.score = False
finetune_local(finetune_args, progress=report_progress)

build_args = cli.parse_args([
    "build",
    "--lora", str(ADAPTER_PATH),
    "--out", str(OUTPUT_CACT),
])
build_main(build_args)
print(f"Done: {OUTPUT_CACT}")
