from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import SBERT_MODEL_NAME
from .data_pipeline import clean_text


@dataclass
class EngineInfo:
    backend: str
    model_name: str
    message: str


class SemanticEngine:
    def __init__(self, backend: str = "auto", model_name: str = SBERT_MODEL_NAME):
        self.requested_backend = backend
        self.model_name = model_name
        self.backend = "tfidf"
        self.model = None
        self.message = "Fallback lexical local actif."
        self._initialise()

    def _initialise(self) -> None:
        if self.requested_backend not in {"auto", "sbert"}:
            self.backend = "tfidf"
            self.message = "Fallback lexical local actif."
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.backend = "sbert"
            self.message = f"SBERT local active via {self.model_name}."
        except Exception as exc:  # pragma: no cover
            self.backend = "tfidf"
            self.message = f"Fallback lexical local actif car SBERT n a pas pu se charger : {exc}"

    def info(self) -> EngineInfo:
        return EngineInfo(backend=self.backend, model_name=self.model_name, message=self.message)

    def pairwise_similarity(self, left_texts: list[str], right_texts: list[str]) -> np.ndarray:
        if not left_texts or not right_texts:
            return np.zeros((len(left_texts), len(right_texts)))

        left = [clean_text(text) or "vide" for text in left_texts]
        right = [clean_text(text) or "vide" for text in right_texts]

        if self.backend == "sbert" and self.model is not None:
            left_embeddings = np.asarray(self.model.encode(left, normalize_embeddings=True))
            right_embeddings = np.asarray(self.model.encode(right, normalize_embeddings=True))
            return np.clip(left_embeddings @ right_embeddings.T, 0.0, 1.0)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(left + right)
        left_matrix = matrix[: len(left)]
        right_matrix = matrix[len(left) :]
        return np.clip(cosine_similarity(left_matrix, right_matrix), 0.0, 1.0)
