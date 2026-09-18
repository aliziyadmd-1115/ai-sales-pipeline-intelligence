from pathlib import Path
import os
import json
from pydantic import ValidationError

import pandas as pd
import streamlit as st

from src.model import load_model, predict_win_probability
from src.rag import answer_query
from src.retrieval import build_retriever

st.set_page_config(page_title="AI Sales Pipeline Intelligence", layout="wide")
st.title("AI Sales Pipeline Intelligence Platform")
st.caption("Synthetic portfolio demo • scores are estimates, not validated production sales forecasts")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/opportunities_clean.csv"
MODEL = ROOT / "artifacts/win_model.joblib"

if not DATA.exists() or not MODEL.exists():
    st.error("Project artifacts are missing. Run the setup commands in README.md first.")
    st.stop()

@st.cache_resource
def load_resources():
    df = pd.read_csv(DATA)
    model = load_model(MODEL)
    retriever = build_retriever(df, backend=os.getenv("RETRIEVAL_BACKEND", "tfidf"))
    return df, model, retriever

try:
    _, model, retriever = load_resources()
except (ValueError, RuntimeError, OSError):
    st.error("Could not load the model or retrieval backend. Check setup and restart the app.")
    st.stop()

metrics_path = ROOT / "artifacts/metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    with st.expander("Evaluation evidence — synthetic holdout"):
        a, b, c = st.columns(3)
        a.metric("Holdout ROC-AUC", f"{metrics['roc_auc']:.3f}")
        b.metric("Brier score (lower is better)", f"{metrics['brier_score']:.3f}")
        c.metric("Test opportunities", metrics["test_rows"])
        st.caption("Thresholds were compared on a separate validation set. Full split and predictions are in artifacts/.")

left, right = st.columns(2)

with left:
    st.subheader("1. Predict opportunity win probability")
    region = st.selectbox("Region", ["Northeast", "South", "Midwest", "West"])
    industry = st.selectbox("Industry", ["technology", "financial_services", "healthcare", "retail", "manufacturing"])
    segment = st.selectbox("Customer segment", ["SMB", "Mid-Market", "Enterprise"])
    product = st.selectbox("Product line", ["analytics", "automation", "data_platform", "security"])
    stage = st.selectbox("Sales stage", ["qualification", "discovery", "proposal", "negotiation"])
    value = st.number_input("Estimated value ($)", min_value=1000.0, max_value=1000000.0, value=75000.0)
    engagement = st.slider("Engagement score", 0, 100, 68)
    days = st.number_input("Days in pipeline", min_value=0, max_value=1000, value=52)
    meetings = st.number_input("Meetings", min_value=0, max_value=100, value=4)
    competitor = st.checkbox("Competitor present", value=True)
    proposal = st.checkbox("Proposal sent", value=True)
    discount = st.slider("Discount", 0.0, 0.40, 0.12, 0.01)
    threshold = st.slider(
        "Decision threshold",
        0.20,
        0.80,
        0.50,
        0.05,
        help="Lower thresholds prioritize recall; higher thresholds prioritize precision.",
    )
    notes = st.text_area("Opportunity notes", "Decision makers are engaged and requested a final proposal, but a competitor is also being evaluated.")
    if st.button("Predict win probability"):
        try:
            result = predict_win_probability(
                model, notes=notes, region=region, industry=industry, customer_segment=segment,
                product_line=product, sales_stage=stage, estimated_value=value,
                days_in_pipeline=days, engagement_score=engagement, meetings_count=meetings,
                competitor_present=competitor, discount_pct=discount, proposal_sent=proposal,
                decision_threshold=threshold,
            )
        except ValidationError:
            st.error("Enter at least 20 characters of opportunity notes and valid field values.")
            st.stop()
        probability = result["win_probability"]
        st.metric("Predicted win probability", f"{probability:.1%}")
        st.progress(probability)
        st.success(f"Decision: {result['predicted_outcome'].upper()} at a {threshold:.0%} threshold")
        with st.expander("Prediction response"):
            st.json(result)

with right:
    st.subheader("2. Retrieve historical evidence and analyze")
    query = st.text_area(
        "Business question",
        "What do similar analytics opportunities suggest when executive engagement is strong but a competitor is present?",
    )
    use_llm = st.checkbox("Use local Ollama LLM if available", value=False)
    if st.button("Generate grounded analysis"):
        try:
            result = answer_query(query, retriever, use_llm=use_llm, k=3)
        except ValidationError:
            st.error("Enter a business question between 10 and 2,000 characters.")
            st.stop()
        st.write(result["answer"])
        st.subheader("Retrieved opportunity citations")
        if result["citations"]:
            st.dataframe(pd.DataFrame(result["citations"]), width="stretch")
        if result.get("warning"):
            st.warning(result["warning"])
