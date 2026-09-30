# CSV Analyst Agent

A simple Streamlit project for an internship demo.

The user uploads a CSV or Excel file, writes a business question, and an **Agno Data Analyst Agent** answers.

The agent uses **Agno + Gemini + DuckDB/Pandas tools**, adapted from the repo-style data analyst agent.

## What It Can Answer

- What columns are in this dataset?
- Which product generated the most revenue?
- Show monthly sales trend.
- Which region is most profitable?
- Compare category performance.

## Project Files

- `app.py` - the Streamlit interface
- `agents/csv_analyst_agent.py` - Agno Gemini data analyst agent
- `data/sample_sales.csv` - sample dataset for demo
- `requirements.txt` - Python packages

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Demo Pitch

> This is a CSV Business Analyst Agent. It uses Agno with Gemini, DuckDB, and pandas tools to answer natural-language questions over uploaded CSV or Excel data.
