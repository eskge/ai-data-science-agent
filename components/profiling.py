import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.profiler import profile_dataset

DARK_LAYOUT = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font_color="#c9d1d9", margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
)

def render_profiling():
    st.markdown('<div class="section-tag">STEP 2</div>', unsafe_allow_html=True)
    st.markdown("## 🔬 Dataset Profiling")

    df = st.session_state.df_raw
    if df is None:
        st.warning("Upload a dataset first.")
        return

    if st.session_state.profile is None:
        with st.spinner("Running deterministic dataset analysis..."):
            st.session_state.profile = profile_dataset(df)

    profile = st.session_state.profile

    # ── Quality Flags Overview ──────────────────────────────────────────────
    st.markdown("### 🚦 Quality Flags")
    flags = profile.get("quality_flags", {})
    
    flag_cols = st.columns(4)
    flag_items = [
        ("missing_cols", "Missing Value Columns", "⚠️", "#d29922"),
        ("duplicate_rows", "Duplicate Rows", "🔁", "#f85149"),
        ("constant_cols", "Constant Columns", "📌", "#d29922"),
        ("high_cardinality_cols", "High-Cardinality Cols", "🔑", "#58a6ff"),
    ]
    for i, (key, label, icon, color) in enumerate(flag_items):
        val = flags.get(key, [])
        count = len(val) if isinstance(val, list) else (int(val) if val else 0)
        with flag_cols[i % 4]:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="color:{color}; font-size:24px;">{icon} {count}</div>
            </div>""", unsafe_allow_html=True)

    # ── Column-by-column table ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Column Analysis")

    col_rows = []
    for col, info in profile["columns"].items():
        col_rows.append({
            "Column": col,
            "Type": info["col_type"],
            "Missing": f"{info['missing_count']} ({info['missing_pct']}%)",
            "Unique": info["unique_count"],
            "Cardinality %": f"{info['cardinality_pct']}%",
            "Sample": str(info["sample_values"][:2]),
        })
    col_df = pd.DataFrame(col_rows)
    st.dataframe(col_df, use_container_width=True, hide_index=True)

    # ── Missing values chart ─────────────────────────────────────────────────
    missing_data = {
        col: info["missing_pct"]
        for col, info in profile["columns"].items()
        if info["missing_pct"] > 0
    }
    if missing_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🕳️ Missing Values by Column")
        sorted_missing = dict(sorted(missing_data.items(), key=lambda x: -x[1]))
        fig = go.Figure(go.Bar(
            x=list(sorted_missing.keys()),
            y=list(sorted_missing.values()),
            marker_color=["#f85149" if v > 30 else "#d29922" if v > 10 else "#58a6ff" for v in sorted_missing.values()],
        ))
        fig.update_layout(
            **DARK_LAYOUT,
            yaxis_title="Missing %",
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Outlier Report ───────────────────────────────────────────────────────
    outliers = profile.get("outliers", {})
    if outliers:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Outlier Detection")
        out_rows = []
        for col, odata in outliers.items():
            out_rows.append({
                "Column": col,
                "IQR Outliers": odata.get("iqr_count", 0),
                "IQR %": f"{odata.get('iqr_pct', 0)}%",
                "Z-Score Outliers": odata.get("zscore_count", 0),
                "Z-Score %": f"{odata.get('zscore_pct', 0)}%",
            })
        if out_rows:
            st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

    # ── Duplicates ───────────────────────────────────────────────────────────
    dupes = profile.get("duplicates", {})
    dup_rows = dupes.get("duplicate_rows", 0)
    dup_cols = dupes.get("duplicate_cols", [])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        color = "#f85149" if dup_rows > 0 else "#3fb950"
        st.markdown(f"""<div class="metric-card">
            <div class="label">Duplicate Rows</div>
            <div class="value" style="color:{color};">{dup_rows:,}</div>
            <div class="sub">{dup_rows/profile['shape']['rows']*100:.1f}% of dataset</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        color = "#f85149" if dup_cols else "#3fb950"
        st.markdown(f"""<div class="metric-card">
            <div class="label">Duplicate Columns</div>
            <div class="value" style="color:{color};">{len(dup_cols)}</div>
            <div class="sub">{', '.join(dup_cols[:3]) if dup_cols else 'None found'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶ Continue to Health Score →", type="primary"):
        st.session_state.active_step = 3
        st.rerun()
