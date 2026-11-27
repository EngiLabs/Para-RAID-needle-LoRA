from __future__ import annotations

import json
import random
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

from generate_dataset import _bucket_key
from offtopic_filter import BASE_MODEL, CLF_PATH, OfftopicFilter, atomic_label_for_row


TRAIN_FILTER_PATH = Path("data/train_filter.jsonl")
VAL_PATH = Path("data/val.jsonl")
SIM_RAW_PATH = Path("data/sim_raw.json")
BUCKET_CAP = 45
SEED = 42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def balance_for_filter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    live_queries: set[str] = set()
    if SIM_RAW_PATH.exists():
        raw = json.loads(SIM_RAW_PATH.read_text(encoding="utf-8"))
        live_queries = {" ".join(c["query"].strip().split()) for c in raw.get("commands", [])}

    offtopic = [r for r in rows if not r.get("answers")]
    cmd_buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("answers"):
            cmd_buckets.setdefault(_bucket_key(r), []).append(r)

    selected_cmds: list[dict[str, Any]] = []
    for items in cmd_buckets.values():
        live = [r for r in items if r["query"] in live_queries]
        synth = [r for r in items if r["query"] not in live_queries]
        rng.shuffle(synth)
        merged = live + synth
        selected_cmds.extend(merged[: max(BUCKET_CAP, len(live))])

    out = selected_cmds + offtopic
    rng.shuffle(out)
    return out


t0 = time.perf_counter()
flt = OfftopicFilter(model_name=BASE_MODEL, clf_path=None)

raw_rows = read_jsonl(TRAIN_FILTER_PATH)
balanced_rows = balance_for_filter(raw_rows)

texts = [r["query"] for r in balanced_rows]
y_bin = [1 if r.get("answers") else 0 for r in balanced_rows]
vecs = flt.embed(texts)

binary_clf = SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)
binary_clf.fit(vecs, y_bin)

cmd_indices = [i for i, y in enumerate(y_bin) if y == 1]
cmd_vecs = vecs[cmd_indices]

intent_labels: list[str] = []
intent_map: dict[str, list[dict[str, Any]]] = {}
for i in cmd_indices:
    lbl, ans = atomic_label_for_row(balanced_rows[i])
    intent_labels.append(lbl)
    if ans is not None and lbl not in intent_map:
        intent_map[lbl] = ans

intent_clf = SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)
intent_clf.fit(cmd_vecs, intent_labels)

CLF_PATH.parent.mkdir(parents=True, exist_ok=True)
bundle = {
    "binary_clf": binary_clf,
    "intent_clf": intent_clf,
    "intent_map": intent_map,
}
joblib.dump(bundle, CLF_PATH)
print(
    f"Trained {CLF_PATH} on {len(balanced_rows)} balanced samples "
    f"(cmds={len(cmd_indices)}, offtopic={len(balanced_rows) - len(cmd_indices)}, "
    f"atomic_classes={len(intent_map)}) in {time.perf_counter() - t0:.1f}s"
)

if VAL_PATH.exists():
    flt_eval = OfftopicFilter(model_name=BASE_MODEL, clf_path=CLF_PATH)
    val_rows = read_jsonl(VAL_PATH)
    val_vecs = flt_eval.embed([r["query"] for r in val_rows])
    for th in (0.25, 0.35, 0.45):
        flt_eval.threshold = th
        fp = 0
        fn = 0
        n_off = 0
        n_cmd = 0
        for r, v in zip(val_rows, val_vecs):
            ok, _ = flt_eval.is_command_vec(v[None, :])
            if not r.get("answers"):
                n_off += 1
                fp += int(ok)
            else:
                n_cmd += 1
                fn += int(not ok)
        print(f"  val @ th={th:.2f}: Offtopic FP={100*fp/max(1, n_off):.2f}% ({fp}/{n_off}) | Cmd Missed={100*fn/max(1, n_cmd):.2f}% ({fn}/{n_cmd})")
