"""PS69 - Customer Feedback Classifier (Streamlit UI)."""
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from classifier import CATEGORIES, classify_dataframe, used_model

load_dotenv()
st.set_page_config(page_title="Customer Feedback Classifier", layout="wide")
st.title("Customer Feedback Classifier")
st.caption("Upload a feedback CSV. The LLM tags each row with a category, priority and sentiment.")

api_key = (os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
           or st.sidebar.text_input("API key (Groq gsk_... or Gemini)", type="password"))
batch_size = st.sidebar.slider("Rows per LLM call", 5, 20, 10)

up = st.file_uploader("Feedback CSV", type="csv")
use_sample = st.button("Use sample data")
if use_sample:
    st.session_state["df"] = pd.read_csv("sample_feedback.csv")
elif up is not None:
    st.session_state["df"] = pd.read_csv(up)

if "df" in st.session_state:
    df = st.session_state["df"]
    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)
    text_col = st.selectbox("Column containing feedback text", df.columns,
                            index=list(df.columns).index("feedback") if "feedback" in df.columns else 0)

    if st.button("Classify", type="primary"):
        if not api_key:
            st.error("Enter an API key in the sidebar (or set GROQ_API_KEY in .env).")
            st.stop()
        bar = st.progress(0.0)
        with st.spinner("Classifying..."):
            st.session_state["result"] = classify_dataframe(
                df, text_col, api_key, batch_size, progress=bar.progress)
        bar.empty()
        st.success(f"Done. Model used: {used_model()}")

if "result" in st.session_state:
    res = st.session_state["result"]
    st.subheader("Results")
    c1, c2 = st.columns(2)
    cats = c1.multiselect("Filter by category", CATEGORIES, default=CATEGORIES)
    pris = c2.multiselect("Filter by priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    view = res[res["category"].isin(cats) & res["priority"].isin(pris)]
    order = {"High": 0, "Medium": 1, "Low": 2}
    view = view.sort_values("priority", key=lambda s: s.map(order))
    st.dataframe(view, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.bar_chart(res["category"].value_counts())
    m2.bar_chart(res["priority"].value_counts())

    st.download_button("Download results CSV", res.to_csv(index=False).encode("utf-8"),
                       "classified_feedback.csv", "text/csv")
