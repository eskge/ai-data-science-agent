import streamlit as st
import plotly.graph_objects as go
from utils.profiler import profile_dataset

def render_health_score():
    st.markdown('<div class="section-tag">STEP 3</div>', unsafe_allow_html=True)
    st.markdown("## 💯 Dataset Health Score")

    df = st.session_state.df_raw
    if df is None:
        st.warning("Upload a dataset first.")
        return

    if st.session_state.profile is None:
        with st.spinner("Profiling dataset..."):
            st.session_state.profile = profile_dataset(df)

    profile = st.session_state.profile
    hs = profile["health_score"]
    score = hs["score"]
    breakdown = hs.get("breakdown", {})

    # Score color
    if score >= 80:
        score_color = "#3fb950"
        grade = "Excellent"
    elif score >= 60:
        score_color = "#d29922"
        grade = "Good"
    elif score >= 40:
        score_color = "#f0883e"
        grade = "Moderate"
    else:
        score_color = "#f85149"
        grade = "Poor"

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"color": score_color, "size": 48}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#7d8590", "tickfont": {"color": "#7d8590"}},
            "bar": {"color": score_color, "thickness": 0.3},
            "bgcolor": "#161b22",
            "bordercolor": "#21262d",
            "steps": [
                {"range": [0, 40], "color": "#3b0d0c"},
                {"range": [40, 60], "color": "#2d1e00"},
                {"range": [60, 80], "color": "#1a2a1a"},
                {"range": [80, 100], "color": "#0d2818"},
            ],
            "threshold": {
                "line": {"color": score_color, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        height=260,
        margin=dict(l=30, r=30, t=20, b=10),
    )

    col_gauge, col_details = st.columns([1, 1])

    with col_gauge:
        st.markdown(f"""<div class="metric-card" style="text-align:center; padding:24px;">
            <div class="label" style="text-align:center; margin-bottom:8px;">Dataset Health Score</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div style="text-align:center; font-size:18px; font-weight:600; color:{score_color}; margin-top:-16px; padding-bottom:16px;">
            {grade}
        </div></div>""", unsafe_allow_html=True)

    with col_details:
        st.markdown("**Score Breakdown**")
        st.markdown("<br>", unsafe_allow_html=True)

        breakdown_items = [
            ("missing_values", "Missing Values"),
            ("duplicates", "Duplicates"),
            ("outliers", "Outliers"),
            ("data_consistency", "Data Consistency"),
            ("column_quality", "Column Quality"),
        ]

        for key, label in breakdown_items:
            item = breakdown.get(key, {})
            rating = item.get("rating", "Unknown")
            deduction = item.get("deduction", 0)
            detail = item.get("detail", "")

            if rating == "Excellent":
                icon, color = "🟢", "#3fb950"
            elif rating == "Good":
                icon, color = "🟡", "#d29922"
            elif rating == "Moderate":
                icon, color = "🟠", "#f0883e"
            else:
                icon, color = "🔴", "#f85149"

            st.markdown(f"""<div style="display:flex; align-items:flex-start; justify-content:space-between;
                padding: 8px 0; border-bottom: 1px solid #21262d;">
                <div>
                    <span style="font-size:13px; font-weight:500;">{icon} {label}</span>
                    <div style="font-size:11px; color:#7d8590; margin-top:2px;">{detail}</div>
                </div>
                <div style="text-align:right; flex-shrink:0; margin-left:12px;">
                    <span style="font-size:12px; font-weight:600; color:{color};">{rating}</span>
                    {f'<div style="font-size:10px; color:#f85149;">-{deduction} pts</div>' if deduction > 0 else ''}
                </div>
            </div>""", unsafe_allow_html=True)

    # Deductions explanation
    deductions = hs.get("deductions", [])
    if deductions:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📉 Score Deductions**")
        for d in deductions:
            st.markdown(f"""<div class="warning-box">
                ⚠️ <strong>{d.get('reason', '')}</strong><br>
                <span style="font-size:12px;">{d.get('detail', '')} — <strong>−{d.get('points', 0)} points</strong></span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶ Continue to AI Summary →", type="primary"):
        st.session_state.active_step = 4
        st.rerun()
