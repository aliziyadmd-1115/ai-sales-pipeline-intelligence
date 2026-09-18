from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import redact_pii

RETRIEVAL_COLUMNS = ["opportunity_id", "industry", "product_line", "outcome", "notes", "close_reason"]


@dataclass
class RetrievedOpportunity:
    opportunity_id: str
    industry: str
    product_line: str
    outcome: str
    notes: str
    close_reason: str
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[RetrievedOpportunity]: ...


def _prepare_documents(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if df.empty:
        raise ValueError("Cannot build a retriever with no historical opportunities")
    work = df[RETRIEVAL_COLUMNS].fillna("").astype(str).copy()
    if work["opportunity_id"].str.strip().eq("").any() or work["opportunity_id"].duplicated().any():
        raise ValueError("Retrieval requires nonempty, unique opportunity IDs")
    work = work.sort_values("opportunity_id").reset_index(drop=True)
    for col in ("notes", "close_reason"):
        work[col] = work[col].map(redact_pii)
    # Rank on pre-close context. Return close reasons as evidence after retrieval.
    documents = (
        work["notes"] + "\nIndustry: " + work["industry"] + "\nProduct: " + work["product_line"]
    ).tolist()
    return work, documents


def _check_search(query: str, k: int) -> str:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return redact_pii(str(query).strip())


class TfidfRetriever:
    """Deterministic lexical retrieval; abstains on zero or weak overlap."""

    def __init__(self, df: pd.DataFrame, min_similarity: float = 0.10):
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1")
        self.df, corpus = _prepare_documents(df)
        self.min_similarity = min_similarity
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_features=25000, stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 3) -> list[RetrievedOpportunity]:
        query = _check_search(query, k)
        q = self.vectorizer.transform([query])
        if q.nnz == 0:
            return []
        scores = cosine_similarity(q, self.matrix)[0]
        best = scores.argsort(kind="stable")[::-1][:k]
        return [
            RetrievedOpportunity(
                **self.df.iloc[int(idx)].to_dict(), score=round(float(scores[idx]), 4),
            )
            for idx in best if scores[idx] > 0 and scores[idx] >= self.min_similarity
        ]


class ChromaRetriever:
    """Persistent semantic index names include a content and embedding fingerprint."""

    def __init__(
        self, df: pd.DataFrame, persist_dir: Path = Path("artifacts/chroma_db"),
        min_similarity: float = 0.25,
    ):
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1")
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError as exc:
            raise RuntimeError("Chroma backend requires requirements-vector.txt") from exc
        self.df, documents = _prepare_documents(df)
        self.min_similarity = min_similarity
        self.by_id = self.df.set_index("opportunity_id")
        embedding_model = "all-MiniLM-L6-v2"
        fingerprint = hashlib.sha256(
            (embedding_model + "|preclose-v2|" + self.df.to_json(orient="records")).encode()
        ).hexdigest()[:24]
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=f"sales_{fingerprint}", embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        # Idempotent upserts also repair an interrupted initial indexing run.
        if self.collection.count() != len(self.df):
            ids = self.df["opportunity_id"].tolist()
            metadata = self.df[["industry", "product_line", "outcome"]].to_dict(orient="records")
            for start in range(0, len(ids), 200):
                end = start + 200
                self.collection.upsert(
                    ids=ids[start:end], documents=documents[start:end], metadatas=metadata[start:end],
                )

    def search(self, query: str, k: int = 3) -> list[RetrievedOpportunity]:
        query = _check_search(query, k)
        if not query:
            return []
        out = self.collection.query(query_texts=[query], n_results=min(k, len(self.df)))
        results = []
        for opp_id, distance in zip(out["ids"][0], out["distances"][0]):
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            if score < self.min_similarity:
                continue
            row = self.by_id.loc[opp_id]
            results.append(RetrievedOpportunity(
                opportunity_id=opp_id, **row.to_dict(), score=round(score, 4),
            ))
        return results


def build_retriever(df: pd.DataFrame, backend: str = "tfidf") -> Retriever:
    backend = backend.lower().strip()
    if backend == "chroma":
        return ChromaRetriever(df)
    if backend != "tfidf":
        raise ValueError("backend must be 'tfidf' or 'chroma'")
    return TfidfRetriever(df)
