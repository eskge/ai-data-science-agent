"""
LLM Planning Layer — Groq API
The LLM is only ever used to:
  - Generate human-readable summaries
  - Propose cleaning plans (executed deterministically)
  - Explain insights
  - Chat about the data
  - Write code recommendations
It NEVER touches the data directly.
"""
import json
import re
from typing import Dict, Any, List, Optional
from groq import Groq


def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _chat(client: Groq, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """Low-level chat completion."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _safe_chat(client, system, user, **kwargs):
    """Wrapper that returns None on error instead of surfacing raw error strings."""
    try:
        return _chat(client, system, user, **kwargs)
    except Exception as e:
        import streamlit as st
        err = str(e)
        if "rate_limit" in err.lower() or "429" in err:
            st.warning("⚠️ Groq rate limit hit — wait a moment and try again.")
        elif "401" in err or "invalid_api_key" in err.lower():
            st.error("❌ Invalid API key — re-enter it in the sidebar.")
        else:
            st.warning(f"⚠️ LLM call failed: {err}")
        return None


def generate_dataset_summary(client: Groq, profile: Dict[str, Any], df_preview: str) -> str:
    system = """You are a senior data scientist writing a concise, business-friendly dataset summary.
Write in plain English. Avoid jargon. Be specific about what you observe.
Keep it to 3–5 sentences. Mention: what the dataset likely represents, key quality concerns, and a hint at the likely prediction target if obvious."""
    
    col_summary = []
    for col, info in list(profile["columns"].items())[:20]:
        col_summary.append(f"{col}: {info['col_type']}, {info['missing_pct']}% missing, {info['unique_count']} unique values")
    
    user = f"""Dataset shape: {profile['shape']['rows']} rows × {profile['shape']['cols']} columns
Health Score: {profile['health_score']['score']}/100

Column overview:
{chr(10).join(col_summary)}

Quality flags: {json.dumps({k: v for k, v in profile.get('quality_flags', {}).items() if v}, indent=2)}

First few rows preview:
{df_preview}

Write a business-friendly summary of this dataset."""
    
    return _safe_chat(client, system, user)


def generate_cleaning_insights(client: Groq, cleaning_plan: List[Dict], profile: Dict) -> str:
    system = """You are a data scientist reviewing a data cleaning plan.
Provide a brief paragraph (2-3 sentences) explaining the overall cleaning strategy and any risks.
Be practical and specific."""
    
    plan_text = "\n".join([f"- {s['title']}: {s['reason']}" for s in cleaning_plan])
    user = f"""Cleaning plan for a {profile['shape']['rows']}-row dataset:
{plan_text}

Briefly explain the overall approach and key considerations."""
    
    return _safe_chat(client, system, user)


def generate_eda_insight(client: Groq, chart_type: str, column: str, stats_summary: str, target_col: Optional[str] = None) -> str:
    system = """You are a data scientist providing a one-sentence insight about a chart.
Be specific, actionable, and highlight what's most important for prediction tasks.
One sentence only. No preamble."""
    
    target_hint = f" The prediction target is '{target_col}'." if target_col else ""
    user = f"""Chart: {chart_type} of '{column}'.{target_hint}
Statistics: {stats_summary}
Provide one specific insight."""
    
    return _safe_chat(client, system, user, max_tokens=150)


def generate_model_recommendations(client: Groq, task_type: str, n_rows: int, n_features: int, target_col: str, top_features: List[str]) -> str:
    system = """You are a senior ML engineer recommending models.
For each recommended model, provide:
- Name
- 2-3 word reason why it fits
- Main pros (1-2 bullet points)  
- Main cons (1 bullet point)
- Expected performance range
Format as clean markdown."""
    
    user = f"""Task: {task_type}
Dataset: {n_rows} rows, {n_features} features
Target: {target_col}
Top features: {', '.join(top_features[:5]) if top_features else 'unknown'}

Recommend 3 models with brief analysis."""
    
    return _safe_chat(client, system, user, max_tokens=800)


def generate_senior_review(client: Groq, profile: Dict, task_type: str, target_col: str, 
                           top_features: List[str], ml_readiness: Dict, leakage_warnings: List) -> str:
    system = """You are a senior data scientist with 15 years of experience writing a peer review of a dataset.
Write in first person. Be direct, practical, and specific. 
Format as 6–8 bullet points using markdown. Each bullet = one actionable insight or warning.
Focus on: what to watch out for, feature engineering ideas, modeling strategy, and gotchas."""
    
    readiness_text = "\n".join([f"  {k}: {v['label']} ({v['detail']})" for k, v in ml_readiness.items() if k != "overall"])
    leakage_text = "\n".join([f"  {w['column']}: {w['message']}" for w in leakage_warnings]) if leakage_warnings else "  None detected"
    
    user = f"""Dataset: {profile['shape']['rows']} rows × {profile['shape']['cols']} columns
Health Score: {profile['health_score']['score']}/100
Task Type: {task_type}
Target Variable: {target_col}
Top Features: {', '.join(top_features[:8]) if top_features else 'Not yet analyzed'}

ML Readiness:
{readiness_text}

Leakage Warnings:
{leakage_text}

Quality Issues: {json.dumps({k: v for k, v in profile.get('quality_flags', {}).items() if v})}

Write your senior data scientist review."""
    
    return _safe_chat(client, system, user, max_tokens=1000, temperature=0.4)


def chat_with_data(client: Groq, question: str, df_info: str, schema: str, conversation_history: List[Dict]) -> Dict[str, Any]:
    """
    Chat interface. LLM analyzes the question and returns:
    - answer text
    - optional pandas code for computation (executed deterministically by caller)
    """
    system = """You are a helpful data science assistant. You have access to a pandas DataFrame called `df`.

When answering questions:
1. If computation is needed, provide Python/Pandas code in a ```python block that stores results in a variable called `result`
2. Always explain what the code does and what it means
3. Be specific with numbers when available
4. If you can answer from the schema/stats alone, do so without code

Available context: dataset schema and basic statistics are provided.
Always be concise and direct."""
    
    messages = [{"role": "system", "content": system}]
    
    context = f"""Dataset info:
{df_info}

Schema and stats:
{schema}"""
    
    messages.append({"role": "user", "content": context})
    messages.append({"role": "assistant", "content": "I have the dataset context. What would you like to know?"})
    
    for msg in conversation_history[-6:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": question})
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()
        code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None
        return {"answer": content, "code": code}
    except Exception as e:
        err = str(e)
        if "rate_limit" in err.lower() or "429" in err:
            msg = "Rate limit hit — please wait a moment and try again."
        elif "401" in err or "invalid_api_key" in err.lower():
            msg = "Invalid API key — re-enter it in the sidebar."
        else:
            msg = f"LLM error: {err}"
        return {"answer": f"⚠️ {msg}", "code": None}


def generate_starter_code(client: Groq, target_col: str, task_type: str, 
                           top_features: List[str], dataset_name: str) -> str:
    system = """You are a senior ML engineer writing production-ready Python code.
Generate complete, runnable scikit-learn code with:
- Data loading
- Preprocessing with Pipeline
- Train/test split (stratified if classification)
- Recommended model
- Evaluation metrics appropriate to the task
- Feature importance plot
- Comments explaining key choices
Use best practices. Return only the Python code, no markdown."""
    
    user = f"""Dataset: {dataset_name}
Target: {target_col}
Task: {task_type}
Top features: {', '.join(top_features[:10]) if top_features else 'all available'}

Generate production-ready ML training code."""
    
    return _safe_chat(client, system, user, max_tokens=2000)
