import streamlit as st
from utils.profiler import profile_dataset
from utils.llm import get_client, generate_dataset_summary

def render_ai_summary():
    st.markdown('<div class="section-tag">STEP 4</div>', unsafe_allow_html=True)
    st.markdown("## 🤖 AI Dataset Summary")
    st.markdown("The LLM reads your dataset profile and generates a plain-English business summary.")

    df = st.session_state.df_raw
    if df is None:
        st.warning("Upload a dataset first.")
        return

    if st.session_state.profile is None:
        with st.spinner("Profiling dataset..."):
            st.session_state.profile = profile_dataset(df)

    profile = st.session_state.profile

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.session_state.ai_summary:
            st.markdown("**🧠 AI-Generated Summary**")
            st.markdown(f'<div class="ai-insight">{st.session_state.ai_summary}</div>', unsafe_allow_html=True)

    with col2:
        if not st.session_state.groq_key:
            st.markdown("""<div class="warning-box">
                ⚙️ Add your Groq API key in the sidebar to enable AI summaries.
            </div>""", unsafe_allow_html=True)
        else:
            if st.button("🤖 Generate AI Summary", type="primary"):
                with st.spinner("AI is reading your dataset..."):
                    try:
                        client = get_client(st.session_state.groq_key)
                        df_preview = df.head(5).to_string(max_cols=10)
                        summary = generate_dataset_summary(client, profile, df_preview)
                        st.session_state.ai_summary = summary
                        st.rerun()
                    except Exception as e:
                        st.error(f"LLM error: {e}")

    if not st.session_state.ai_summary:
        st.markdown("""<div class="ai-insight">
            Click "Generate AI Summary" to get an AI-powered overview of your dataset in plain English.
        </div>""", unsafe_allow_html=True)

    # Show deterministic stats as context
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📊 Profile at a Glance**")

    col_types = {}
    for col, info in profile["columns"].items():
        t = info["col_type"]
        col_types[t] = col_types.get(t, 0) + 1

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Numerical Cols</div>
            <div class="value">{col_types.get('numerical', 0) + col_types.get('categorical_numeric', 0)}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Categorical Cols</div>
            <div class="value">{col_types.get('categorical', 0)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Text / High-Card</div>
            <div class="value">{col_types.get('high_cardinality_text', 0) + col_types.get('text', 0)}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Date Cols</div>
            <div class="value">{col_types.get('datetime', 0) + col_types.get('potential_datetime', 0)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶ Continue to AI Cleaning Plan →", type="primary"):
        st.session_state.active_step = 5
        st.rerun()
