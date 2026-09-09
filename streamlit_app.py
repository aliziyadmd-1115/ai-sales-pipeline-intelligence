from pathlib import Path
import os

import pandas as pd
import streamlit as st

from src.model import load_model, predict_win_probability
from src.rag import answer_query
from src.retrieval import build_retriever

st.set_page_config(page_title="AI Sales Pipeline Intelligence", layout="wide")
st.title("AI Sales Pipeline Intelligence Platform")
st.caption("Win-probability modeling + historical opportunity retrieval + grounded RAG-style business insights")

DATA = Path("data/opportunities_clean.csv")
MODEL = Path("artifacts/win_model.joblib")

if not DATA.exists() or not MODEL.exists():
    st.error("Project artifacts are missing. Run the setup commands in README.md first.")
    st.stop()

@st.cache_resource
def load_resources():
    df = pd.read_csv(DATA)
    model = load_model(MODEL)
    retriever = build_retriever(df, backend=os.getenv("RETRIEVAL_BACKEND", "tfidf"))
    return df, model, retriever

_, model, retriever = load_resources()
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
    notes = st.text_area("Opportunity notes", "Decision makers are engaged and requested a final proposal, but a competitor is also being evaluated.")
    if st.button("Predict win probability"):
        st.json(predict_win_probability(
            model, notes=notes, region=region, industry=industry, customer_segment=segment,
            product_line=product, sales_stage=stage, estimated_value=value,
            days_in_pipeline=days, engagement_score=engagement, meetings_count=meetings,
            competitor_present=competitor, discount_pct=discount, proposal_sent=proposal,
        ))

with right:
    st.subheader("2. Retrieve historical evidence and analyze")
    query = st.text_area(
        "Business question",
        "What do similar analytics opportunities suggest when executive engagement is strong but a competitor is present?",
    )
    use_llm = st.checkbox("Use local Ollama LLM if available", value=False)
    if st.button("Generate grounded analysis"):
        result = answer_query(query, retriever, use_llm=use_llm, k=3)
        st.write(result["answer"])
        st.subheader("Retrieved opportunity citations")
        st.dataframe(pd.DataFrame(result["citations"]), use_container_width=True)
