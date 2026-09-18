import pandas as pd
import pytest

from src.retrieval import TfidfRetriever, build_retriever


def test_tfidf_retrieval_returns_results():
    df = pd.DataFrame([
        {"opportunity_id": "OPP-1", "industry": "technology", "product_line": "analytics", "outcome": "won", "notes": "Executive sponsor approved analytics proposal", "close_reason": "Strong sponsorship"},
        {"opportunity_id": "OPP-2", "industry": "retail", "product_line": "security", "outcome": "lost", "notes": "Retail customer delayed security purchase", "close_reason": "Budget timing"},
    ])
    hits = TfidfRetriever(df).search("analytics proposal with executive sponsor", k=1)
    assert hits[0].opportunity_id == "OPP-1"


def test_retriever_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        build_retriever(pd.DataFrame(), backend="unknown")


@pytest.fixture
def history():
    return pd.DataFrame([
        {"opportunity_id": "OPP-1", "industry": "technology", "product_line": "analytics",
         "outcome": "won", "notes": "Executive approved analytics proposal",
         "close_reason": "Contact sponsor@example.com"},
        {"opportunity_id": "OPP-2", "industry": "retail", "product_line": "security",
         "outcome": "lost", "notes": "Retail security deployment delayed",
         "close_reason": "Budget timing"},
    ])


def test_no_evidence_for_out_of_vocabulary_or_stopword_queries(history):
    retriever = TfidfRetriever(history)
    assert retriever.search("volcanology tectonic seismograph") == []
    assert retriever.search("the and it of for") == []


def test_retrieval_redacts_email_in_historical_close_reason(history):
    hits = TfidfRetriever(history).search("executive analytics proposal")
    assert hits
    assert "sponsor@example.com" not in hits[0].close_reason


def test_retrieval_does_not_rank_on_post_close_outcome_reason(history):
    history.loc[1, "close_reason"] = "quasar astrophysics telescope"
    assert TfidfRetriever(history).search("quasar astrophysics telescope") == []


def test_empty_history_and_invalid_limit_fail_clearly(history):
    with pytest.raises(ValueError, match="no historical"):
        TfidfRetriever(history.iloc[:0])
    with pytest.raises(ValueError, match="positive integer"):
        TfidfRetriever(history).search("analytics proposal", k=0)


def test_chroma_refreshes_changed_data_and_repairs_partial_index(monkeypatch, tmp_path, history):
    """Exercise persistence contracts without downloading an embedding model."""
    import sys
    from types import SimpleNamespace
    from src.retrieval import ChromaRetriever

    collections = {}

    class Collection:
        def __init__(self):
            self.rows = {}
            self.requested = None

        def count(self):
            return len(self.rows)

        def upsert(self, ids, documents, metadatas):
            self.rows.update(dict(zip(ids, documents)))

        def query(self, query_texts, n_results):
            self.requested = n_results
            ids = list(self.rows)[:n_results]
            return {"ids": [ids], "distances": [[0.1] * len(ids)]}

    class Client:
        def get_or_create_collection(self, name, **kwargs):
            return collections.setdefault(name, Collection())

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **kw: Client()))
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", SimpleNamespace(
        SentenceTransformerEmbeddingFunction=lambda **kw: object(),
    ))
    original = ChromaRetriever(history, persist_dir=tmp_path)
    original.collection.rows.pop("OPP-2")
    repaired = ChromaRetriever(history, persist_dir=tmp_path)
    assert repaired.collection.count() == 2
    assert len(repaired.search("analytics", k=10)) == 2
    assert repaired.collection.requested == 2

    changed = history.iloc[:1].copy()
    changed["notes"] = "A revised analytics opportunity"
    updated = ChromaRetriever(changed, persist_dir=tmp_path)
    assert updated.collection is not original.collection
    assert updated.search("analytics")[0].notes == "A revised analytics opportunity"
    assert len(updated.search("analytics", k=10)) == 1
