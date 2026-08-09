"""
AI Data Science Agent
Built with Streamlit + Groq API (LLaMA 3.3 70B)

Architecture:
  LLM  = planner, analyst, explainer  (Groq API)
  Code = executor, transformer         (Pandas / Scikit-learn)

The LLM never touches data directly.
All cleaning and transformations are deterministic and auditable.

Steps:
  1.  CSV Upload
  2-3. Dataset Profiling & Health Score
  4.  AI Dataset Summary
  5-6. AI Cleaning Plan & Execution
  7.  Automated EDA
  8.  Auto Target Detection
  9-11. ML Readiness · Leakage · Cross-Column Anomalies · Feature Importance
  12. Model Recommendations
  13. Senior Data Scientist Review
  14. AI Data Chat
  15-16. Download Centre

Requires:
  pip install streamlit pandas numpy scikit-learn plotly groq scipy
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Optional

# ─── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="AI Data Science Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports ────────────────────────────────────────────────────────────────
from utils.profiler import (
    profile_dataset, detect_target_candidates,
    assess_ml_readiness, detect_leakage, compute_feature_importance,
    detect_cross_column_anomalies
)
from utils.cleaner import generate_cleaning_plan, execute_cleaning_plan
from utils.charts import (
    plot_distribution, plot_categorical, plot_correlation_heatmap,
    plot_missing_values, plot_target_distribution,
    plot_feature_importance, plot_scatter, plot_ml_readiness
)
from utils.reports import (
    generate_quality_report, generate_ml_report,
    generate_jupyter_notebook, generate_eda_report
)
from utils import llm as llm_module

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --primary: #6366f1; --primary-dim: rgba(99,102,241,0.15);
    --secondary: #8b5cf6; --accent: #06b6d4;
    --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
    --bg: #0b0f1a; --surface: #131929; --surface2: #1a2235;
    --border: rgba(99,102,241,0.2); --text: #e2e8f0; --muted: #64748b; --radius: 12px;
}
html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
.card-header { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.card-value { font-size: 2rem; font-weight: 700; color: var(--text); line-height: 1.1; }
.card-sub { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }
.insight-box { background: var(--primary-dim); border-left: 3px solid var(--primary); border-radius: 0 8px 8px 0; padding: 0.9rem 1.2rem; margin: 0.75rem 0; font-size: 0.88rem; color: var(--text); line-height: 1.6; }
.warn-box { background: rgba(245,158,11,0.1); border-left: 3px solid var(--warning); border-radius: 0 8px 8px 0; padding: 0.9rem 1.2rem; margin: 0.5rem 0; font-size: 0.85rem; }
.danger-box { background: rgba(239,68,68,0.1); border-left: 3px solid var(--danger); border-radius: 0 8px 8px 0; padding: 0.9rem 1.2rem; margin: 0.5rem 0; font-size: 0.85rem; }
.success-box { background: rgba(16,185,129,0.1); border-left: 3px solid var(--success); border-radius: 0 8px 8px 0; padding: 0.9rem 1.2rem; margin: 0.5rem 0; font-size: 0.85rem; }
.step-badge { display: inline-block; background: var(--primary-dim); color: var(--primary); border: 1px solid var(--primary); border-radius: 999px; padding: 0.15rem 0.75rem; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.5rem; }
.section-title { font-size: 1.4rem; font-weight: 700; color: var(--text); margin: 0.25rem 0 1rem 0; }
.risk-low { color: var(--success); font-size: 0.75rem; font-weight: 600; }
.risk-med { color: var(--warning); font-size: 0.75rem; font-weight: 600; }
.risk-high { color: var(--danger); font-size: 0.75rem; font-weight: 600; }
.chat-user { background: var(--primary-dim); border-radius: 12px 12px 4px 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; text-align: right; }
.chat-agent { background: var(--surface2); border: 1px solid var(--border); border-radius: 12px 12px 12px 4px; padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; border: none !important; padding: 0.6rem 1.2rem !important; font-size: 0.85rem !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important; }
.stButton > button { background: linear-gradient(135deg, var(--primary), var(--secondary)) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background: var(--surface2) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }
.stSelectbox > div > div { background: var(--surface2) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }
.stProgress > div > div { background-color: var(--primary) !important; }
.senior-review { background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08)); border: 1px solid rgba(139,92,246,0.3); border-radius: var(--radius); padding: 1.5rem; font-size: 0.88rem; line-height: 1.8; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
[data-testid="stFileUploader"] { background: var(--surface2) !important; border: 2px dashed var(--border) !important; border-radius: var(--radius) !important; }
.stDownloadButton > button { background: var(--surface2) !important; color: var(--accent) !important; border: 1px solid var(--accent) !important; font-size: 0.82rem !important; }
.hero { background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(6,182,212,0.06) 100%); border: 1px solid var(--border); border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem; text-align: center; }
.hero-title { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, var(--primary), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
.hero-sub { font-size: 1rem; color: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "df_raw": None, "df_clean": None, "filename": None, "profile": None,
        "cleaning_plan": None, "approved_steps": [], "cleaning_log": [],
        "dataset_summary": None, "target_col": None, "task_type": None,
        "top_features": [], "ml_readiness": None, "leakage_warnings": [],
        "model_recs": None, "senior_review": None, "starter_code": None,
        "chat_history": [], "groq_client": None, "api_key": "",
        "_cleaning_insight": None, "_client_key": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── UI helpers ─────────────────────────────────────────────────────────────
def card(header, value, sub=""):
    st.markdown(
        f'<div class="card"><div class="card-header">{header}</div>'
        f'<div class="card-value">{value}</div>'
        f'{"<div class=card-sub>" + sub + "</div>" if sub else ""}</div>',
        unsafe_allow_html=True,
    )

def insight(text):
    if text:
        st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)

def danger_box(text):
    st.markdown(f'<div class="danger-box">🚨 {text}</div>', unsafe_allow_html=True)

def success_box(text):
    st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)

def section(badge, title):
    st.markdown(f'<div class="step-badge">{badge}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def risk_badge(risk):
    cls = {"Low": "risk-low", "Medium": "risk-med", "High": "risk-high"}.get(risk, "risk-low")
    return f'<span class="{cls}">● {risk} Risk</span>'


# ─── Groq client ─────────────────────────────────────────────────────────────
def get_client():
    """Return validated Groq client. Validates key on first use; caches per key."""
    if st.session_state.groq_client and st.session_state.get("_client_key") == st.session_state.api_key:
        return st.session_state.groq_client
    if st.session_state.api_key:
        try:
            client = llm_module.get_client(st.session_state.api_key)
            client.models.list()  # raises 401 on bad key
            st.session_state.groq_client = client
            st.session_state["_client_key"] = st.session_state.api_key
            return client
        except Exception as e:
            st.session_state.groq_client = None
            st.session_state["_client_key"] = None
            err = str(e)
            if "401" in err or "invalid_api_key" in err.lower() or "authentication" in err.lower():
                st.sidebar.error("❌ Invalid Groq API key — check console.groq.com")
            else:
                st.sidebar.warning(f"⚠️ Groq connection failed: {err}")
            return None
    return None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 DS Agent")
    st.markdown("---")

    api_key = st.text_input(
        "Groq API Key", value=st.session_state.api_key,
        type="password", placeholder="gsk_...",
        help="Get your free key at console.groq.com",
    )
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.session_state.groq_client = None
        st.session_state["_client_key"] = None

    if api_key and st.session_state.get("_client_key") == api_key:
        st.markdown('<div class="success-box" style="font-size:0.75rem;padding:0.5rem 0.8rem">✓ Groq connected</div>', unsafe_allow_html=True)
    elif api_key:
        st.markdown('<div style="font-size:0.75rem;padding:0.5rem 0.8rem;color:var(--muted)">⏳ Key entered — validates on next action</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box" style="font-size:0.75rem;padding:0.5rem 0.8rem">Enter Groq key for AI features</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.df_raw is not None:
        df_w = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
        st.markdown(f"📊 **{st.session_state.filename}**")
        st.markdown(f"<small style='color:var(--muted)'>{len(df_w):,} rows × {df_w.shape[1]} cols</small>", unsafe_allow_html=True)
        if st.session_state.df_clean is not None:
            st.markdown("<small style='color:var(--success)'>✓ Data cleaned</small>", unsafe_allow_html=True)
        if st.session_state.target_col:
            st.markdown(f"<small style='color:var(--accent)'>🎯 Target: {st.session_state.target_col}</small>", unsafe_allow_html=True)
        if st.session_state.task_type:
            st.markdown(f"<small style='color:var(--muted)'>{st.session_state.task_type}</small>", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🗑️ Reset / New Dataset", use_container_width=True):
            for _k in list(st.session_state.keys()):
                del st.session_state[_k]
            st.rerun()

    st.markdown("<small style='color:var(--muted)'>LLM = planner & analyst<br>Pandas = executor<br>Deterministic & auditable</small>", unsafe_allow_html=True)


# ─── Hero ────────────────────────────────────────────────────────────────────
if st.session_state.df_raw is None:
    st.markdown(
        '<div class="hero"><div class="hero-title">🔬 AI Data Science Agent</div>'
        '<div class="hero-sub">Upload a CSV to automatically profile, clean, visualise, '
        'and get ML recommendations — powered by Groq LLaMA 3.3.</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 1", "Upload Dataset")
uploaded = st.file_uploader("Drop a CSV file here", type=["csv"], label_visibility="collapsed")

if uploaded is not None and (st.session_state.filename != uploaded.name or st.session_state.df_raw is None):
    with st.spinner("Reading file…"):
        try:
            try:
                df = pd.read_csv(uploaded, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, encoding="latin-1")

            # Reset all state for new file
            reset_keys = [
                "df_raw", "df_clean", "profile", "cleaning_plan", "dataset_summary",
                "target_col", "task_type", "model_recs", "senior_review",
                "starter_code", "ml_readiness", "_cleaning_insight",
            ]
            for key in reset_keys:
                st.session_state[key] = None
            st.session_state["approved_steps"] = []
            st.session_state["cleaning_log"] = []
            st.session_state["top_features"] = []
            st.session_state["leakage_warnings"] = []
            st.session_state["chat_history"] = []
            # Clear EDA insight cache
            for k in list(st.session_state.keys()):
                if k.startswith("_eda_insight") or k.startswith("_cached_"):
                    del st.session_state[k]
            st.session_state.df_raw = df
            st.session_state.filename = uploaded.name
        except Exception as e:
            st.error(f"Could not read file: {e}")

if st.session_state.df_raw is None:
    st.stop()

df_raw = st.session_state.df_raw
df_work = st.session_state.df_clean if st.session_state.df_clean is not None else df_raw

# Overview cards
c1, c2, c3, c4 = st.columns(4)
with c1: card("Rows", f"{len(df_work):,}", "records")
with c2: card("Columns", str(df_work.shape[1]), "features")
with c3: card("Memory", f"{df_work.memory_usage(deep=True).sum()/1024/1024:.1f} MB", "in memory")
with c4:
    mp = round(df_work.isna().sum().sum() / (df_work.shape[0] * df_work.shape[1]) * 100, 1)
    card("Missing", f"{mp}%", "of all cells")

with st.expander("📋 Data Preview", expanded=False):
    st.dataframe(df_work.head(10), use_container_width=True)
    st.caption(f"Showing first 10 of {len(df_work):,} rows")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2–3 — Profiling & Health Score
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 2–3", "Dataset Profiling & Health Score")

if st.session_state.profile is None:
    with st.spinner("Profiling dataset…"):
        st.session_state.profile = profile_dataset(df_work)

profile = st.session_state.profile
health = profile["health_score"]
score = health["score"]
color = "#10b981" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
label = "Excellent" if score >= 80 else "Good" if score >= 70 else "Fair" if score >= 50 else "Poor"

col_l, col_r = st.columns([1, 2])
with col_l:
    st.markdown(
        f'<div class="card" style="text-align:center;border-color:{color}40">'
        f'<div class="card-header">Dataset Health Score</div>'
        f'<div style="font-size:3.5rem;font-weight:800;color:{color};line-height:1.1">{score}</div>'
        f'<div style="font-size:2rem;color:{color};font-weight:300">/ 100</div>'
        f'<div style="color:var(--muted);font-size:0.85rem;margin-top:0.5rem">{label}</div></div>',
        unsafe_allow_html=True,
    )

with col_r:
    rows_html = ""
    for cat, info in health["breakdown"].items():
        bc = {"Excellent": "#10b981", "Good": "#06b6d4", "Moderate": "#f59e0b", "Poor": "#ef4444"}.get(info["label"], "#6366f1")
        dt = f"-{info['deduction']}" if info["deduction"] > 0 else "✓"
        rows_html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:0.4rem 0;border-bottom:1px solid var(--border)">'
            f'<span style="font-size:0.85rem">{cat}</span>'
            f'<span style="display:flex;gap:0.75rem;align-items:center">'
            f'<span style="font-size:0.75rem;color:{bc};font-weight:600">{info["label"]}</span>'
            f'<span style="font-size:0.75rem;color:var(--muted)">{info["detail"]}</span>'
            f'<span style="font-size:0.75rem;color:{bc};font-weight:700;min-width:24px;text-align:right">{dt}</span>'
            f'</span></div>'
        )
    st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

# Column profiles expander
with st.expander("🔍 Column Profiles", expanded=False):
    col_names = list(profile["columns"].keys())
    search_col = st.selectbox("Select column", col_names, key="col_select")
    if search_col:
        info = profile["columns"][search_col]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Type", info["col_type"])
        with c2: st.metric("Missing", f"{info['missing_pct']}%")
        with c3: st.metric("Unique", info["unique_count"])
        with c4: st.metric("Cardinality", f"{info['cardinality_pct']}%")
        if "mean" in info:
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Mean", info["mean"])
            with c2: st.metric("Std", info["std"])
            with c3: st.metric("Min", info["min"])
            with c4: st.metric("Max", info["max"])
            if info.get("skewness") is not None:
                st.caption(f"Skewness: {info['skewness']}")
            st.plotly_chart(plot_distribution(df_work, search_col), use_container_width=True, key="pc_1")
        elif info["col_type"] in ("categorical", "high_cardinality_text", "text"):
            st.plotly_chart(plot_categorical(df_work, search_col), use_container_width=True, key="pc_2")

# Missing values bar chart
if any(i["missing_count"] > 0 for i in profile["columns"].values()):
    fig = plot_missing_values(profile)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="pc_3")

# Outlier bounds detail
if profile.get("outliers"):
    with st.expander("📐 Outlier Bounds Detail", expanded=False):
        for col_o, o_info in profile["outliers"].items():
            lb, ub = o_info["lower_bound"], o_info["upper_bound"]
            clamp_note = "  *(lower bound clamped to 0 — values are non-negative)*" if lb == 0.0 else ""
            st.markdown(
                f"**{col_o}**: {o_info['iqr_outliers']} outliers ({o_info['iqr_pct']}%)"
                f" — IQR bounds: [{lb}, {ub}]{clamp_note}"
            )

# Quality flags
flags = profile.get("quality_flags", {})
if flags:
    st.markdown("**Quality Flags Detected**")
    for fn, fc in flags.items():
        if fc:
            lbl = fn.replace("_", " ").title()
            if "constant" in fn or "mismatch" in fn:
                danger_box(f"**{lbl}**: {', '.join(str(c) for c in fc)}")
            elif "high_missing" in fn or "high_cardinality" in fn:
                warn(f"**{lbl}**: {', '.join(str(c) for c in fc)}")
            else:
                insight(f"**{lbl}**: {', '.join(str(c) for c in fc)}")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — AI Dataset Summary
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 4", "AI Dataset Summary")
client = get_client()

if client:
    if st.session_state.dataset_summary is None:
        with st.spinner("Generating AI summary…"):
            st.session_state.dataset_summary = llm_module.generate_dataset_summary(
                client, profile, df_work.head(5).to_string()
            )
    if st.session_state.dataset_summary:
        st.markdown(
            f'<div class="insight-box" style="font-size:0.95rem;line-height:1.8">'
            f'{st.session_state.dataset_summary}</div>',
            unsafe_allow_html=True,
        )
else:
    warn("Enter a Groq API key in the sidebar to enable AI summaries.")
    n_miss = sum(1 for c in profile["columns"].values() if c["missing_count"] > 0)
    insight(
        f"Dataset has **{len(df_work):,} rows** and **{df_work.shape[1]} columns**. "
        f"**{n_miss}** columns contain missing values. Health score: **{health['score']}/100**."
    )

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5–6 — Cleaning Plan & Execution
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 5–6", "AI Cleaning Plan & Execution")

if st.session_state.cleaning_plan is None:
    st.session_state.cleaning_plan = generate_cleaning_plan(df_raw, profile)

plan = st.session_state.cleaning_plan

if not plan:
    success_box("No cleaning steps needed — your dataset looks clean!")
else:
    # LLM cleaning insight (cached)
    if client:
        if not st.session_state._cleaning_insight:
            with st.spinner("Analysing cleaning strategy…"):
                st.session_state._cleaning_insight = llm_module.generate_cleaning_insights(client, plan, profile)
        insight(st.session_state._cleaning_insight)

    st.markdown(f"**{len(plan)} cleaning actions proposed.** Review and approve below.")
    cb1, cb2, _ = st.columns([1, 1, 3])
    with cb1:
        if st.button("✅ Approve All"):
            st.session_state.approved_steps = plan.copy()
    with cb2:
        if st.button("❌ Reject All"):
            st.session_state.approved_steps = []

    approved_ids = {s["id"] for s in st.session_state.approved_steps}
    new_approved = []

    for step in plan:
        approved = step["id"] in approved_ids
        border = "var(--success)" if approved else "var(--border)"
        st.markdown(
            f'<div class="card" style="margin-bottom:0.6rem;border-color:{border}40">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
            f'<div><span style="font-size:0.9rem;font-weight:600">{step["title"]}</span>'
            f'<div style="font-size:0.78rem;color:var(--muted);margin-top:0.3rem">{step["reason"]}</div>'
            f'<div style="font-size:0.76rem;color:var(--muted);margin-top:0.15rem">Impact: {step["impact"]}</div></div>'
            f'<div style="text-align:right;min-width:80px">{risk_badge(step["risk"])}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if st.checkbox("Approve", value=approved, key=f"step_{step['id']}"):
            new_approved.append(step)

    st.session_state.approved_steps = new_approved

    if st.session_state.approved_steps:
        st.markdown(f"**{len(st.session_state.approved_steps)}/{len(plan)} steps approved.**")
        if st.button("🚀 Execute Cleaning Plan", type="primary"):
            with st.spinner("Executing cleaning…"):
                df_cleaned, log = execute_cleaning_plan(df_raw, st.session_state.approved_steps)
                st.session_state.df_clean = df_cleaned
                st.session_state.cleaning_log = log
                st.session_state.profile = profile_dataset(df_cleaned)
                # Reset downstream state
                for k in ["dataset_summary", "cleaning_plan", "model_recs", "senior_review",
                          "ml_readiness", "_cleaning_insight", "_cached_quality_report", "_cached_eda_report"]:
                    st.session_state[k] = None
                st.session_state["approved_steps"] = []
                st.session_state["top_features"] = []
                st.session_state["leakage_warnings"] = []
                # Clear EDA insight cache
                for k in list(st.session_state.keys()):
                    if k.startswith("_eda_insight"):
                        del st.session_state[k]
                st.rerun()

# Cleaning log
if st.session_state.cleaning_log:
    st.markdown("**Cleaning Log**")
    for entry in st.session_state.cleaning_log:
        icon = "✅" if entry["status"] == "Success" else "❌"
        success_box(f"{icon} **Step {entry['step']}**: {entry['detail']}")

    if st.session_state.df_clean is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before Cleaning**")
            st.metric("Rows", f"{len(df_raw):,}")
            st.metric("Missing cells", f"{df_raw.isna().sum().sum():,}")
        with c2:
            st.markdown("**After Cleaning**")
            after_miss = st.session_state.df_clean.isna().sum().sum()
            st.metric("Rows", f"{len(st.session_state.df_clean):,}",
                      delta=f"{len(st.session_state.df_clean) - len(df_raw)}")
            _miss_delta = after_miss - df_raw.isna().sum().sum()
            st.metric("Missing cells", f"{after_miss:,}",
                      delta=f"{_miss_delta:+,}", delta_color="inverse")

df_work = st.session_state.df_clean if st.session_state.df_clean is not None else df_raw
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Automated EDA
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 7", "Automated Exploratory Data Analysis")

eda_tabs = st.tabs(["📊 Distributions", "📦 Categorical", "🔥 Correlations", "🎯 Target Analysis"])
numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_work.select_dtypes(include=["object", "category"]).columns.tolist()
# Include potential_datetime cols in categorical tab
cat_cols = cat_cols + [
    c for c in df_work.columns
    if profile["columns"].get(c, {}).get("col_type") == "potential_datetime" and c not in cat_cols
]

with eda_tabs[0]:
    if numeric_cols:
        sel_num = st.selectbox("Select numeric column", numeric_cols, key="eda_num")
        if sel_num:
            st.plotly_chart(plot_distribution(df_work, sel_num), use_container_width=True, key="pc_4")
            if client:
                _eda_key = f"_eda_insight_num_{sel_num}"
                if _eda_key not in st.session_state:
                    ci = profile["columns"].get(sel_num, {})
                    stats_s = f"mean={ci.get('mean')}, std={ci.get('std')}, skew={ci.get('skewness')}"
                    st.session_state[_eda_key] = llm_module.generate_eda_insight(
                        client, "histogram + box plot", sel_num, stats_s, st.session_state.target_col
                    )
                insight(st.session_state.get(_eda_key))
    else:
        warn("No numeric columns found.")

with eda_tabs[1]:
    if cat_cols:
        sel_cat = st.selectbox("Select categorical column", cat_cols, key="eda_cat")
        if sel_cat:
            st.plotly_chart(plot_categorical(df_work, sel_cat), use_container_width=True, key="pc_5")
            if client:
                _eda_key = f"_eda_insight_cat_{sel_cat}"
                if _eda_key not in st.session_state:
                    vc = df_work[sel_cat].value_counts()
                    stats_s = f"top values: {vc.head(3).to_dict()}, {df_work[sel_cat].nunique()} unique"
                    st.session_state[_eda_key] = llm_module.generate_eda_insight(
                        client, "bar chart", sel_cat, stats_s, st.session_state.target_col
                    )
                insight(st.session_state.get(_eda_key))
    else:
        warn("No categorical columns found.")

with eda_tabs[2]:
    fig_corr = plot_correlation_heatmap(df_work)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True, key="pc_6")
        num_df = df_work.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            corr = num_df.corr().abs()
            np.fill_diagonal(corr.values, 0)
            stacked = corr.stack()
            top_pairs = stacked[stacked > 0].nlargest(5)
            if not top_pairs.empty:
                st.markdown("**Top correlated pairs:**")
                for (c1c, c2c), val in top_pairs.items():
                    if c1c != c2c:
                        st.markdown(f"- `{c1c}` ↔ `{c2c}`: **{val:.3f}**")
    else:
        warn("Need at least 2 numeric columns for correlation analysis.")

with eda_tabs[3]:
    if st.session_state.target_col and st.session_state.target_col in df_work.columns:
        target = st.session_state.target_col
        st.plotly_chart(plot_target_distribution(df_work, target), use_container_width=True, key="pc_7")
        avail = [c for c in numeric_cols if c != target]
        if avail:
            feat = st.selectbox("Feature vs target", avail, key="tgt_feat")
            if feat:
                _sc_y = target if target in numeric_cols else feat
                _sc_c = target if df_work[target].nunique() <= 10 else None
                st.plotly_chart(
                    plot_scatter(df_work, feat, _sc_y, _sc_c),
                    use_container_width=True, key="pc_scatter_tgt",
                )
    else:
        warn("Select a target variable (Step 8) to enable target analysis.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Auto Target Detection
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 8", "Auto Target Detection")

candidates = detect_target_candidates(df_work, profile)
if candidates:
    st.markdown("**Detected potential target variables:**")
    for cand in candidates[:5]:
        cc = "#10b981" if cand["confidence"] >= 80 else "#f59e0b" if cand["confidence"] >= 50 else "#94a3b8"
        reasons_html = "".join([
            f'<span style="margin-left:0.5rem;font-size:0.72rem;color:var(--muted)">• {r}</span>'
            for r in cand["reasons"]
        ])
        st.markdown(
            f'<div class="card" style="margin-bottom:0.5rem">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div><span style="font-weight:600">{cand["column"]}</span>'
            f'<span style="margin-left:0.75rem;font-size:0.78rem;color:var(--muted)">{cand["task_type"]}</span>'
            f'{reasons_html}</div>'
            f'<span style="font-size:0.82rem;font-weight:700;color:{cc}">{cand["confidence"]}% confidence</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

all_cols = ["(none)"] + list(df_work.columns)
default_idx = 0
if candidates:
    try:
        default_idx = all_cols.index(candidates[0]["column"])
    except ValueError:
        pass

sel_target = st.selectbox("Select target variable", all_cols, index=default_idx, key="target_select")
if sel_target != "(none)":
    st.session_state.target_col = sel_target
    _t = df_work[sel_target]
    if _t.nunique() == 2:
        st.session_state.task_type = "Binary Classification"
    elif _t.nunique() <= 20 and (not pd.api.types.is_numeric_dtype(_t) or _t.nunique() <= 10):
        st.session_state.task_type = "Multi-class Classification"
    elif pd.api.types.is_numeric_dtype(_t) and _t.nunique() > 20:
        st.session_state.task_type = "Regression"
    else:
        st.session_state.task_type = "Classification"
    success_box(f"Target set to **{sel_target}** → **{st.session_state.task_type}**")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9–11 — ML Readiness · Leakage · Cross-Column Anomalies · Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 9–11", "ML Readiness · Leakage Detection · Feature Importance")

if st.session_state.target_col:
    target = st.session_state.target_col

    # ML Readiness
    ml_readiness = assess_ml_readiness(df_work, profile, target)
    st.session_state.ml_readiness = ml_readiness

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**ML Readiness**")
        overall = ml_readiness.get("overall", 0)
        oc = "#10b981" if overall >= 80 else "#f59e0b" if overall >= 60 else "#ef4444"
        st.markdown(
            f'<div style="font-size:2rem;font-weight:700;color:{oc};margin-bottom:0.75rem">{overall}/100</div>',
            unsafe_allow_html=True,
        )
        for k, v in ml_readiness.items():
            if k == "overall":
                continue
            bc = "#10b981" if v["score"] >= 80 else "#f59e0b" if v["score"] >= 60 else "#ef4444"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:0.35rem 0;'
                f'border-bottom:1px solid var(--border)">'
                f'<span style="font-size:0.82rem">{k.replace("_"," ").title()}</span>'
                f'<span style="font-size:0.78rem;color:{bc};font-weight:600">{v["label"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(v["detail"])

    with c2:
        st.plotly_chart(plot_ml_readiness(ml_readiness), use_container_width=True, key="pc_9")

    # Leakage detection
    st.markdown("**Data Leakage Detection**")
    leakage = detect_leakage(df_work, profile, target)
    st.session_state.leakage_warnings = leakage
    if leakage:
        for w in leakage:
            if w["severity"] == "High":
                danger_box(f"**{w['type']} — {w['column']}**: {w['message']}")
            else:
                warn(f"**{w['type']} — {w['column']}**: {w['message']}")
    else:
        success_box("No obvious data leakage detected.")

    # Cross-column anomaly detection
    st.markdown("**Cross-Column Anomaly Detection**")
    cross_anomalies = detect_cross_column_anomalies(df_work)
    if cross_anomalies:
        for a in cross_anomalies:
            cols_str = " & ".join(f"`{c}`" for c in a["columns"])
            if a["severity"] == "High":
                danger_box(f"**{a['type']}** ({cols_str}): {a['message']}")
            else:
                warn(f"**{a['type']}** ({cols_str}): {a['message']}")
    else:
        success_box("No cross-column anomalies detected.")

    # Feature importance
    st.markdown("**Feature Importance Estimation**")
    if st.button("⚡ Compute Feature Importance", key="fi_btn"):
        with st.spinner("Computing (mutual information + correlation)…"):
            st.session_state.top_features = compute_feature_importance(df_work, target)

    if st.session_state.top_features:
        fig_fi = plot_feature_importance(st.session_state.top_features)
        if fig_fi:
            st.plotly_chart(fig_fi, use_container_width=True, key="pc_10")
        if client:
            _fi_key = f"_eda_insight_fi_{target}"
            if _fi_key not in st.session_state:
                top_names = [f["feature"] for f in st.session_state.top_features[:3]]
                st.session_state[_fi_key] = llm_module.generate_eda_insight(
                    client, "feature importance chart",
                    ", ".join(top_names), f"top features: {top_names}", target,
                )
            insight(st.session_state.get(_fi_key))
else:
    warn("Select a target variable (Step 8) to run ML readiness checks.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 12 — Model Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 12", "Model Recommendations")

if st.session_state.task_type and st.session_state.target_col:
    if client:
        if st.session_state.model_recs is None:
            if st.button("🤖 Generate Model Recommendations", key="mr_btn"):
                with st.spinner("Analysing best models for your task…"):
                    top_names = [f["feature"] for f in st.session_state.top_features[:5]]
                    st.session_state.model_recs = llm_module.generate_model_recommendations(
                        client, st.session_state.task_type, len(df_work),
                        df_work.shape[1], st.session_state.target_col, top_names,
                    )
        if st.session_state.model_recs:
            st.markdown(f'<div class="card">{st.session_state.model_recs}</div>', unsafe_allow_html=True)
    else:
        task = st.session_state.task_type
        recs = (
            [("Random Forest", "Strong baseline, handles mixed data"),
             ("XGBoost / LightGBM", "Excellent tabular performance"),
             ("Logistic Regression", "Fast, interpretable baseline")]
            if "Classification" in task else
            [("Random Forest Regressor", "Handles non-linearity well"),
             ("XGBoost Regressor", "Top tabular performance"),
             ("Ridge Regression", "Fast, interpretable")]
        )
        for name, desc in recs:
            st.markdown(
                f'<div class="card"><strong>{name}</strong><br>'
                f'<span style="color:var(--muted);font-size:0.85rem">{desc}</span></div>',
                unsafe_allow_html=True,
            )
else:
    warn("Select a target variable (Step 8) to get model recommendations.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 13 — Senior Data Scientist Review
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 13", "Senior Data Scientist Review")
st.markdown(
    '<div style="font-size:0.8rem;color:var(--muted);margin-bottom:1rem">'
    'AI-powered review from the perspective of an experienced data scientist</div>',
    unsafe_allow_html=True,
)

if client and st.session_state.target_col:
    if st.session_state.senior_review is None:
        if st.button("🧠 Generate Senior Review", key="sr_btn"):
            with st.spinner("Thinking like a senior DS…"):
                top_names = [f["feature"] for f in st.session_state.top_features[:8]]
                st.session_state.senior_review = llm_module.generate_senior_review(
                    client, profile,
                    st.session_state.task_type or "Unknown",
                    st.session_state.target_col, top_names,
                    st.session_state.ml_readiness or {},
                    st.session_state.leakage_warnings or [],
                )
    if st.session_state.senior_review:
        st.markdown(
            f'<div class="senior-review">{st.session_state.senior_review}</div>',
            unsafe_allow_html=True,
        )
elif not client:
    warn("Add a Groq API key to enable the Senior Data Scientist Review.")
else:
    warn("Select a target variable (Step 8) to generate a Senior Review.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 14 — AI Data Chat
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 14", "AI Data Chat")

if client:
    for msg in st.session_state.chat_history:
        cls = "chat-user" if msg["role"] == "user" else "chat-agent"
        icon = "🙋" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{cls}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        suggestions = [
            "What factors most influence the target variable?",
            "Which column has the most missing values?",
            "Show the distribution of the target variable.",
            "Are there strong correlations between features?",
        ]
        sc1, sc2 = st.columns(2)
        for i, s in enumerate(suggestions):
            with (sc1 if i % 2 == 0 else sc2):
                if st.button(s, key=f"sug_{i}"):
                    st.session_state._pending_chat = s

    user_input = st.chat_input("Ask anything about your data…")
    if not user_input and hasattr(st.session_state, "_pending_chat"):
        user_input = st.session_state._pending_chat
        del st.session_state._pending_chat

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking…"):
            df_info = f"Shape: {df_work.shape}, Target: {st.session_state.target_col}, Task: {st.session_state.task_type}"
            schema = df_work.dtypes.to_string() + "\n\n" + df_work.describe().to_string()
            result = llm_module.chat_with_data(
                client, user_input, df_info, schema, st.session_state.chat_history[:-1]
            )
            answer = result["answer"]
            if result.get("code"):
                try:
                    local_ns = {"df": df_work.copy(), "pd": pd, "np": np}
                    exec(result["code"], local_ns)  # noqa: S102
                    cr = local_ns.get("result", None)
                    if cr is not None:
                        if isinstance(cr, (pd.DataFrame, pd.Series)):
                            answer += f"\n\n*Result:*\n{cr.to_string()}"
                        else:
                            answer += f"\n\n*Result:* {cr}"
                except Exception as e:
                    answer += f"\n\n*(Computation note: {e})*"
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
else:
    warn("Enter a Groq API key to enable the AI Data Chat.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 15–16 — Download Centre
# ═══════════════════════════════════════════════════════════════════════════════
section("Step 15–16", "Download Centre")

df_export = st.session_state.df_clean if st.session_state.df_clean is not None else df_raw

dl1, dl2, dl3, dl4 = st.columns(4)

with dl1:
    st.download_button(
        "📥 Cleaned CSV",
        data=df_export.to_csv(index=False).encode(),
        file_name=f"cleaned_{st.session_state.filename}",
        mime="text/csv",
        use_container_width=True,
    )

with dl2:
    if "_cached_quality_report" not in st.session_state:
        st.session_state["_cached_quality_report"] = generate_quality_report(
            profile, st.session_state.cleaning_log
        )
    st.download_button(
        "📋 Quality Report",
        data=st.session_state["_cached_quality_report"].encode(),
        file_name="quality_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

with dl3:
    if "_cached_eda_report" not in st.session_state:
        st.session_state["_cached_eda_report"] = generate_eda_report(
            profile, st.session_state.top_features, st.session_state.dataset_summary or ""
        )
    st.download_button(
        "📊 EDA Report",
        data=st.session_state["_cached_eda_report"].encode(),
        file_name="eda_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

with dl4:
    ml_md = generate_ml_report(
        st.session_state.task_type or "Unknown",
        st.session_state.target_col or "Unknown",
        st.session_state.top_features or [],
        st.session_state.ml_readiness or {},
        st.session_state.leakage_warnings or [],
        st.session_state.model_recs or "Not generated",
        st.session_state.senior_review or "Not generated",
    )
    st.download_button(
        "🤖 ML Report",
        data=ml_md.encode(),
        file_name="ml_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

dl5, dl6, dl7, dl8 = st.columns(4)

with dl5:
    log_json = json.dumps(st.session_state.cleaning_log, indent=2, default=str)
    st.download_button(
        "🧹 Cleaning Log",
        data=log_json.encode(),
        file_name="cleaning_log.json",
        mime="application/json",
        use_container_width=True,
    )

with dl6:
    if client and st.session_state.target_col:
        if st.session_state.starter_code is None:
            if st.button("Generate Starter Code", key="gen_sc"):
                with st.spinner("Generating…"):
                    top_names = [f["feature"] for f in st.session_state.top_features[:10]]
                    st.session_state.starter_code = llm_module.generate_starter_code(
                        client, st.session_state.target_col,
                        st.session_state.task_type or "Classification",
                        top_names, st.session_state.filename or "dataset.csv",
                    )
        if st.session_state.starter_code:
            st.download_button(
                "🐍 Starter .py",
                data=st.session_state.starter_code.encode(),
                file_name="ml_starter.py",
                mime="text/plain",
                use_container_width=True,
            )
    else:
        st.button("🐍 Starter .py", disabled=True, use_container_width=True,
                  help="Set target + API key first")

with dl7:
    if st.session_state.target_col:
        nb = generate_jupyter_notebook(
            st.session_state.filename or "dataset.csv",
            st.session_state.target_col,
            st.session_state.task_type or "Classification",
            st.session_state.top_features or [],
            st.session_state.starter_code or "",
        )
        st.download_button(
            "📓 Notebook .ipynb",
            data=json.dumps(nb, indent=2).encode(),
            file_name="ds_agent_notebook.ipynb",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.button("📓 Notebook", disabled=True, use_container_width=True,
                  help="Select target first")

with dl8:
    full = {
        "filename": st.session_state.filename,
        "shape": profile["shape"],
        "health_score": health["score"],
        "target_col": st.session_state.target_col,
        "task_type": st.session_state.task_type,
        "top_features": st.session_state.top_features[:10] if st.session_state.top_features else [],
        "leakage_warnings": st.session_state.leakage_warnings,
        "cleaning_steps": len(st.session_state.cleaning_log),
    }
    st.download_button(
        "📦 Analysis JSON",
        data=json.dumps(full, indent=2, default=str).encode(),
        file_name="full_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

if st.session_state.starter_code:
    with st.expander("👁️ Preview Starter Code"):
        st.code(st.session_state.starter_code, language="python")

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:var(--muted);font-size:0.75rem;padding:1rem 0">'
    'AI Data Science Agent · LLM = planner · Pandas = executor · Deterministic & auditable'
    '</div>',
    unsafe_allow_html=True,
)
