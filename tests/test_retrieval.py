import pandas as pd
from src.retrieval import TfidfRetriever


def test_tfidf_retrieval_returns_results():
    df = pd.DataFrame([
        {"opportunity_id": "OPP-1", "industry": "technology", "product_line": "analytics", "outcome": "won", "notes": "Executive sponsor approved analytics proposal", "close_reason": "Strong sponsorship"},
        {"opportunity_id": "OPP-2", "industry": "retail", "product_line": "security", "outcome": "lost", "notes": "Retail customer delayed security purchase", "close_reason": "Budget timing"},
    ])
    hits = TfidfRetriever(df).search("analytics proposal with executive sponsor", k=1)
    assert hits[0].opportunity_id == "OPP-1"
