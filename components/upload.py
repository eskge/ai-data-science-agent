import streamlit as st
import pandas as pd
import io

def render_upload():
    st.markdown('<div class="section-tag">STEP 1</div>', unsafe_allow_html=True)
    st.markdown("## 📁 Upload Your Dataset")
    st.markdown("Upload a CSV file to begin the automated analysis pipeline.")

    uploaded = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        help="Supports CSV files up to 200MB"
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df_raw = df
            st.session_state.df_clean = df.copy()
            # Reset downstream state
            st.session_state.profile = None
            st.session_state.cleaning_plan = None
            st.session_state.cleaning_approvals = {}
            st.session_state.cleaning_log = []
            st.session_state.ai_summary = None
            st.session_state.ds_notes = None
            st.session_state.chat_history = []
            st.session_state.selected_target = None
            st.session_state.eda_insights = {}

            mem_kb = df.memory_usage(deep=True).sum() / 1024

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Rows</div>
                    <div class="value">{df.shape[0]:,}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Columns</div>
                    <div class="value">{df.shape[1]}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Memory</div>
                    <div class="value">{mem_kb:.1f}<span style='font-size:16px;font-weight:400;'> KB</span></div>
                </div>""", unsafe_allow_html=True)
            with c4:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Missing Values</div>
                    <div class="value">{missing_pct:.1f}<span style='font-size:16px;font-weight:400;'>%</span></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**📋 Data Preview** — first 10 rows")
            st.dataframe(df.head(10), use_container_width=True, hide_index=False)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🗂️ Column Data Types**")
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Non-Null Count": df.notnull().sum().values,
                "Null Count": df.isnull().sum().values,
                "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
                "Sample Values": [str(df[c].dropna().unique()[:3].tolist()) for c in df.columns]
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div class="success-box">
                ✅ Dataset loaded successfully. Proceed to <strong>Dataset Profiling</strong> to begin the analysis pipeline.
            </div>""", unsafe_allow_html=True)

            if st.button("▶ Continue to Dataset Profiling →", type="primary"):
                st.session_state.active_step = 2
                st.rerun()

        except Exception as e:
            st.markdown(f"""<div class="danger-box">
                ❌ Failed to read CSV: {str(e)}<br>
                Make sure the file is a valid UTF-8 encoded CSV.
            </div>""", unsafe_allow_html=True)

    elif st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.markdown("""<div class="ai-insight">
            ℹ️ Dataset already loaded. Use the sidebar to navigate between pipeline steps.
        </div>""", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding: 60px 20px; color: #7d8590;'>
            <div style='font-size: 48px; margin-bottom: 16px;'>📂</div>
            <div style='font-size: 16px; margin-bottom: 8px; color: #c9d1d9;'>No dataset loaded</div>
            <div style='font-size: 13px;'>Upload a CSV file above to start the AI analysis pipeline</div>
        </div>
        """, unsafe_allow_html=True)
