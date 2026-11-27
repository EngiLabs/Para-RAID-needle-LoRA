from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


BASE_MODEL = "cointegrated/rubert-tiny2"
CLF_PATH = Path("models/offtopic_clf.joblib")


class OfftopicFilter:
    def __init__(
        self,
        model_name: str = BASE_MODEL,
        clf_path: Path | str | None = CLF_PATH,
        threshold: float = 0.5,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.threshold = threshold
        self.clf = joblib.load(clf_path) if clf_path and Path(clf_path).exists() else None

    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        vecs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(
                    texts[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                out = self.model(**enc)
                cls = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
                vecs.append(cls.cpu().numpy())
        return np.vstack(vecs) if vecs else np.empty((0, 312), dtype=np.float32)

    def prob_command(self, query: str) -> float:
        if self.clf is None:
            raise RuntimeError("classifier not loaded")
        vec = self.embed([query])
        return float(self.clf.predict_proba(vec)[0, 1])

    def is_command(self, query: str) -> bool:
        return self.prob_command(query) >= self.threshold
