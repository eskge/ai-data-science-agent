# 🔬 AI Data Science Agent

A production-ready data science automation tool built with **Streamlit** and **Groq API (LLaMA 3.3 70B)**. Upload any CSV and get automated profiling, intelligent cleaning, EDA, ML readiness assessment, and model recommendations — all in one dashboard.

---

## Core Architecture

```
LLM  (Groq / LLaMA 3.3)  →  planner, analyst, explainer
Pandas / Scikit-learn     →  executor, transformer
```

**The LLM never touches your data directly.** It proposes plans and generates insights. All actual data operations are deterministic, reproducible, and auditable.

---

## Features

### 16-Step Pipeline

| Step | Feature | Engine |
|------|---------|--------|
| 1 | CSV Upload (UTF-8 + latin-1 fallback) | — |
| 2–3 | Dataset Profiling + Health Score (0–100) | Deterministic |
| 4 | Business-friendly dataset summary | LLM |
| 5–6 | AI Cleaning Plan → User Approval → Execution | LLM plan + Pandas exec |
| 7 | Automated EDA (distributions, correlations, target analysis) | Deterministic + LLM insights |
| 8 | Auto target detection (classification / regression) | Heuristic |
| 9 | ML Readiness Score with radar chart | Deterministic |
| 10 | Data leakage detection + cross-column anomaly detection | Deterministic |
| 11 | Feature importance (mutual information + correlation) | Scikit-learn |
| 12 | Model recommendations with pros/cons | LLM |
| 13 | Senior Data Scientist review notes | LLM |
| 14 | Natural language data chat with code execution | LLM + Pandas |
| 15–16 | 8 downloadable outputs (CSV, reports, notebook, starter code) | Mixed |

### What the Profiler catches
- Missing values, duplicate rows, duplicate columns
- Constant and near-constant columns
- High-cardinality text columns (ML encoding risk)
- Type mismatches — numeric data stored as strings
- Range values like `"1000-1500"` in numeric columns → parsed to midpoints
- Outlier detection via IQR and Z-score (lower bounds clamped to 0 for non-negative columns)
- Cross-column anomalies (e.g. bathrooms > bedrooms + 2)
- Price/sqft sanity checks for housing data

### Cleaning actions available
- Remove duplicate rows
- Fill missing values — median (numeric) or mode (categorical)
- Drop columns with >30% missing high-cardinality text, or >60% missing any type
- Parse numeric ranges to midpoints + unit conversion (Sq. Meter → sqft)
- Cap outliers using IQR bounds
- Drop constant columns

### Downloadable outputs
1. Cleaned CSV
2. Data Quality Report (Markdown)
3. EDA Report (Markdown)
4. ML Recommendation Report (Markdown)
5. Cleaning Log (JSON)
6. Starter ML training script (`.py`)
7. Jupyter Notebook (`.ipynb`)
8. Full Analysis Summary (JSON)

---

## Project Structure

```
ds_agent/
├── app.py                  # Main Streamlit application
├── requirements.txt
└── utils/
    ├── __init__.py
    ├── profiler.py         # Deterministic dataset profiling engine
    ├── cleaner.py          # Pandas-only cleaning executor
    ├── charts.py           # Plotly visualisation library
    ├── llm.py              # Groq API integration (planner only)
    └── reports.py          # Report and notebook generators
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-ds-agent.git
cd ai-ds-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas numpy scikit-learn plotly groq scipy openpyxl nbformat
```

### 3. Get a Groq API key

Sign up for free at [console.groq.com](https://console.groq.com). The free tier is sufficient for all features.

### 4. Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) and paste your Groq API key in the sidebar.

> **Note:** The app works without an API key — profiling, cleaning, EDA, and downloads are fully deterministic. The LLM features (summaries, insights, model recommendations, chat) require the key.

---

## Demo

Upload any CSV and the agent will:

1. **Instantly profile** every column — types, missing values, cardinality, outliers
2. **Score dataset health** from 0–100 with a breakdown by category
3. **Propose a cleaning plan** with risk labels — you approve or reject each step individually
4. **Execute cleaning** deterministically via Pandas and log every action
5. **Generate EDA charts** with one-line AI insights per chart
6. **Detect your target variable** and assess ML readiness
7. **Rank features** by mutual information + correlation
8. **Recommend models** with pros, cons, and expected performance
9. **Write a Senior Data Scientist review** with actionable notes
10. **Answer questions** about your data in natural language

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Groq API — LLaMA 3.3 70B Versatile |
| Data | Pandas, NumPy, SciPy |
| ML | Scikit-learn |
| Visualisation | Plotly |
| Reports | Markdown, nbformat |

---

## Design Principles

**Determinism over magic** — every data transformation is a traceable Pandas operation. No black-box LLM mutations.

**Approval-gated cleaning** — the agent never modifies your data without explicit step-by-step user approval.

**Fail gracefully** — the app works fully without an API key. LLM features degrade to deterministic fallbacks, not crashes.

**Auditable outputs** — every cleaning action is logged with before/after statistics and downloadable as JSON.

---

## Tested On

| Dataset | Rows | Columns | Notes |
|---------|------|---------|-------|
| Bengaluru House Prices | 13,320 | 9 | Mixed types, range values, high-cardinality location |
| Telco Customer Churn | 7,043 | 21 | Binary classification, class imbalance |
| titanic.csv | 891 | 12 | Missing values, categorical encoding |

---

## Known Limitations

- Feature engineering (extracting BHK from "2 BHK", binning dates) is suggested in the Senior Review but not yet automated
- Chat code execution runs in a sandboxed local namespace but has no timeout guard on large datasets
- Datasets above ~500k rows may be slow to profile; consider sampling first

---

## Author

**Shailesh** — Data Scientist  
B.Tech Data Science, Manipal Institute of Technology  
Warner Bros. Discovery — Content Data Service

---

## License

MIT License. Free to use, modify, and distribute.
