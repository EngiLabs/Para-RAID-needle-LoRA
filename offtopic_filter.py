from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


BASE_MODEL = "cointegrated/rubert-tiny2"
CLF_PATH = Path("models/offtopic_clf.joblib")

COMPLEX_PATTERNS = re.compile(
    r"(\d|"
    r"\b(?:метр|полметр|сантиметр|см|мм|шаг|секунд|полсекунд|мс|градус|прямой\s+угол|девяносто|"
    r"координат|точк|маршрут|затем|потом)\b|"
    r"\sи\s|,)",
    re.IGNORECASE,
)


def is_complex_query(query: str) -> bool:
    return bool(COMPLEX_PATTERNS.search(query))


def atomic_label_for_row(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
    ans = row.get("answers") or []
    if len(ans) != 1 or is_complex_query(row["query"]):
        return "COMPLEX", None
    call = ans[0]
    name = call["name"]
    args = call.get("arguments") or {}
    if name in {"move_distance", "move_duration", "navigate_to"}:
        return "COMPLEX", None
    if name == "turn_robot" and float(args.get("angle_deg", 90.0)) not in (90.0, 180.0):
        return "COMPLEX", None
    if name == "set_speed" and int(args.get("speed", 180)) not in (100, 180, 255):
        return "COMPLEX", None
    label = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    return label, ans


class OfftopicFilter:
    def __init__(
        self,
        model_name: str = BASE_MODEL,
        clf_path: Path | str | None = CLF_PATH,
        threshold: float = 0.35,
        atomic_threshold: float = 0.58,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.threshold = threshold
        self.atomic_threshold = atomic_threshold

        loaded = joblib.load(clf_path) if clf_path and Path(clf_path).exists() else None
        if isinstance(loaded, dict):
            self.binary_clf = loaded.get("binary_clf")
            self.intent_clf = loaded.get("intent_clf")
            self.intent_map: dict[str, list[dict[str, Any]]] = loaded.get("intent_map") or {}
        else:
            self.binary_clf = loaded
            self.intent_clf = None
            self.intent_map = {}

    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        clean = [str(t).encode("utf-8", "ignore").decode("utf-8").strip() or "-" for t in texts]
        vecs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(clean), batch_size):
                enc = self.tokenizer(
                    clean[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                out = self.model(**enc)
                cls = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
                vecs.append(cls.cpu().numpy())
        return np.vstack(vecs) if vecs else np.empty((0, 312), dtype=np.float32)

    def is_command_vec(self, vec: np.ndarray) -> tuple[bool, float]:
        if self.binary_clf is None:
            raise RuntimeError("Classifier not loaded; run train_filter.py first.")
        p_cmd = float(self.binary_clf.predict_proba(vec)[0, 1])
        return p_cmd >= self.threshold, p_cmd

    def prob_command(self, query: str) -> float:
        _, p_cmd = self.is_command_vec(self.embed([query]))
        return p_cmd

    def is_command(self, query: str) -> bool:
        ok, _ = self.is_command_vec(self.embed([query]))
        return ok

    def predict_atomic_vec(
        self,
        query: str,
        vec: np.ndarray,
        min_prob: float | None = None,
    ) -> list[dict[str, Any]] | None:
        if self.intent_clf is None or is_complex_query(query):
            return None
        probs = self.intent_clf.predict_proba(vec)[0]
        idx = int(np.argmax(probs))
        label = str(self.intent_clf.classes_[idx])
        prob = float(probs[idx])
        th = self.atomic_threshold if min_prob is None else min_prob
        if label != "COMPLEX" and prob >= th and label in self.intent_map:
            return self.intent_map[label]
        return None

    def route(self, query: str, agent: Any) -> tuple[list[dict[str, Any]], str]:
        vec = self.embed([query])
        ok, _ = self.is_command_vec(vec)
        if not ok:
            return [], "filtered"

        atomic = self.predict_atomic_vec(query, vec)
        if atomic is not None:
            return atomic, "fast_atomic"

        agent.reset()
        resp = agent.complete(query)
        native = resp.get("function_calls") or []
        if native:
            return native, "function_calls"
        suppressed = resp.get("suppressed_calls") or []
        if suppressed:
            return suppressed, "suppressed_calls"

        fallback = self.predict_atomic_vec(query, vec, min_prob=0.25)
        if fallback is not None:
            return fallback, "fast_atomic"
        return [], "empty"
