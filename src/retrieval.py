from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


class TfidfRetriever:
    """Lightweight retrieval backend used for local development and CI."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        corpus = (
            self.df["notes"] + " Industry: " + self.df["industry"] +
            " Product: " + self.df["product_line"] + " Close reason: " + self.df["close_reason"]
        ).tolist()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=25000)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 3) -> list[RetrievedOpportunity]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        results = []
        for idx in best:
            row = self.df.iloc[int(idx)]
            results.append(RetrievedOpportunity(
                opportunity_id=str(row.opportunity_id),
                industry=str(row.industry),
                product_line=str(row.product_line),
                outcome=str(row.outcome),
                notes=str(row.notes),
                close_reason=str(row.close_reason),
                score=round(float(scores[idx]), 4),
            ))
        return results


class ChromaRetriever:
    """Semantic vector database backend using ChromaDB + sentence-transformers."""

    def __init__(self, df: pd.DataFrame, persist_dir: Path = Path("artifacts/chroma_db")):
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError as exc:
            raise RuntimeError("Chroma backend requires requirements-vector.txt") from exc

        self.df = df.reset_index(drop=True).copy()
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name="historical_sales_opportunities",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() == 0:
            documents = (
                self.df["notes"] + "\nIndustry: " + self.df["industry"] +
                "\nProduct: " + self.df["product_line"] +
                "\nClose reason: " + self.df["close_reason"]
            ).tolist()
            ids = self.df["opportunity_id"].astype(str).tolist()
            metadata = self.df[["industry", "product_line", "outcome"]].astype(str).to_dict(orient="records")
            for start in range(0, len(ids), 200):
                end = start + 200
                self.collection.add(ids=ids[start:end], documents=documents[start:end], metadatas=metadata[start:end])

    def search(self, query: str, k: int = 3) -> list[RetrievedOpportunity]:
        out = self.collection.query(query_texts=[query], n_results=k)
        results = []
        for opp_id, document, meta, distance in zip(
            out["ids"][0], out["documents"][0], out["metadatas"][0], out["distances"][0]
        ):
            notes, _, rest = document.partition("\nIndustry: ")
            close_reason = rest.split("\nClose reason: ", 1)[-1] if "\nClose reason: " in rest else ""
            results.append(RetrievedOpportunity(
                opportunity_id=opp_id,
                industry=str(meta.get("industry", "")),
                product_line=str(meta.get("product_line", "")),
                outcome=str(meta.get("outcome", "")),
                notes=notes,
                close_reason=close_reason,
                score=round(1.0 - float(distance), 4),
            ))
        return results


def build_retriever(df: pd.DataFrame, backend: str = "tfidf") -> Retriever:
    backend = backend.lower().strip()
    if backend == "chroma":
        return ChromaRetriever(df)
    if backend != "tfidf":
        raise ValueError("backend must be 'tfidf' or 'chroma'")
    return TfidfRetriever(df)
