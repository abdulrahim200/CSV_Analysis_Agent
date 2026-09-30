from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.csv_analyst_agent import ask_data_analyst_agent, load_data


load_dotenv()

st.set_page_config(page_title="CSV Analyst Agent", page_icon=":material/table_chart:", layout="wide")

st.title("CSV Data Analyst Agent")
st.caption("Agno agent with Groq, DuckDB, and pandas tools.")

with st.sidebar:
    st.header("Groq")
    groq_api_key = os.getenv("GROQ_API_KEY") or ""
    model_id = st.text_input("Model", value="openai/gpt-oss-20b")
    if groq_api_key:
        st.success("Groq key loaded from environment")
    else:
        st.warning("GROQ_API_KEY is missing from .env")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
df = load_data(uploaded_file)

left, right = st.columns(2)
left.metric("Rows", len(df))
right.metric("Columns", len(df.columns))

with st.expander("Preview data", expanded=True):
    st.dataframe(df.head(20), width="stretch")

st.subheader("Ask a question")
question = st.text_input(
    "Business question",
    value="Which product generated the most revenue?",
)

if st.button("Ask agent", type="primary"):
    if not groq_api_key:
        st.error("GROQ_API_KEY is missing from the environment file.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Agent is analyzing your data..."):
            try:
                answer = ask_data_analyst_agent(df, question, groq_api_key, model_id)
                st.markdown(answer)
            except Exception as error:
                st.error(f"Agent failed: {error}")
                st.info("Try a simpler question, check the Groq key in .env, or verify the dataset format.")

st.caption("Try: columns in this dataset, monthly trend, most profitable region, top product by revenue.")
