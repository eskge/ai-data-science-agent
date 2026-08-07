"""
Report & Code Generation — downloadable deliverables.
"""
import pandas as pd
import numpy as np
import json
import io
from typing import Dict, Any, List, Optional
from datetime import datetime


def generate_quality_report(profile: Dict[str, Any], cleaning_log: List[Dict] = None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    health = profile["health_score"]
    lines = [
        "# Data Quality Report",
        f"Generated: {ts}",
        "",
        "## Dataset Overview",
        f"- Rows: {profile['shape']['rows']:,}",
        f"- Columns: {profile['shape']['cols']}",
        f"- Memory: {profile['memory_mb']} MB",
        "",
        f"## Health Score: {health['score']}/100",
        "",
        "### Breakdown",
    ]
    for cat, info in health["breakdown"].items():
        lines.append(f"- **{cat}**: {info['label']} ({info['detail']})")

    lines += ["", "## Column Profiles", ""]
    for col, info in profile["columns"].items():
        lines.append(f"### {col}")
        lines.append(f"- Type: {info['col_type']} ({info['dtype']})")
        lines.append(f"- Missing: {info['missing_count']} ({info['missing_pct']}%)")
        lines.append(f"- Unique values: {info['unique_count']} ({info['cardinality_pct']}%)")
        if "mean" in info:
            lines.append(f"- Mean: {info['mean']}, Std: {info['std']}, Min: {info['min']}, Max: {info['max']}")
        lines.append("")

    if profile.get("duplicates", {}).get("duplicate_rows", 0) > 0:
        lines += [
            "## Duplicates",
            f"- Duplicate rows: {profile['duplicates']['duplicate_rows']}",
            "",
        ]

    if profile.get("outliers"):
        lines += ["## Outliers", ""]
        for col, out in profile["outliers"].items():
            lines.append(f"- **{col}**: {out['iqr_outliers']} outliers ({out['iqr_pct']}%) via IQR")
        lines.append("")

    if profile.get("quality_flags"):
        lines += ["## Quality Flags", ""]
        for flag, cols in profile["quality_flags"].items():
            if cols:
                lines.append(f"- **{flag.replace('_', ' ').title()}**: {', '.join(str(c) for c in cols)}")
        lines.append("")

    if cleaning_log:
        lines += ["## Cleaning Log", ""]
        for entry in cleaning_log:
            status = "✅" if entry["status"] == "Success" else "❌"
            lines.append(f"{status} Step {entry['step']}: {entry['title']}")
            lines.append(f"   - {entry['detail']}")
            if entry.get("rows_affected"):
                lines.append(f"   - Rows affected: {entry['rows_affected']}")
        lines.append("")

    return "\n".join(lines)


def generate_ml_report(task_type: str, target_col: str, top_features: List[Dict],
                       ml_readiness: Dict, leakage_warnings: List, model_recs: str,
                       senior_review: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# ML Recommendation Report",
        f"Generated: {ts}",
        "",
        "## Problem Definition",
        f"- Task Type: {task_type}",
        f"- Target Variable: {target_col}",
        "",
        "## ML Readiness",
    ]
    for k, v in ml_readiness.items():
        if k != "overall":
            lines.append(f"- **{k}**: {v['label']} — {v['detail']}")
    lines.append(f"\n**Overall Readiness Score: {ml_readiness.get('overall', 'N/A')}/100**")

    if leakage_warnings:
        lines += ["", "## ⚠️ Data Leakage Warnings", ""]
        for w in leakage_warnings:
            lines.append(f"- [{w['severity']}] **{w['column']}** ({w['type']}): {w['message']}")

    lines += ["", "## Top Predictive Features", ""]
    for i, f in enumerate(top_features[:10], 1):
        lines.append(f"{i}. **{f['feature']}** — MI: {f['mutual_info']:.4f}, Correlation: {f['correlation']:.4f}")

    lines += ["", "## Model Recommendations", "", model_recs, "", "## Senior Data Scientist Review", "", senior_review]

    return "\n".join(lines)


def generate_jupyter_notebook(df_name: str, target_col: str, task_type: str,
                               top_features: List[Dict], starter_code: str) -> dict:
    """Returns a nbformat-compatible dict."""
    feature_names = [f["feature"] for f in top_features[:10]]

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": f"# AI Data Science Agent — {df_name}\n\nAuto-generated notebook"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.metrics import classification_report, mean_squared_error\n\npd.set_option('display.max_columns', None)\nplt.style.use('dark_background')"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Load Dataset"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"df = pd.read_csv('{df_name}')\nprint(f'Shape: {{df.shape}}')\ndf.head()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Basic Profiling"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "print('Missing values:')\nprint(df.isnull().sum())\nprint(f'\\nDuplicate rows: {df.duplicated().sum()}')\ndf.describe()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": f"## Target Variable: {target_col}\n\nTask: **{task_type}**"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"print(df['{target_col}'].value_counts())\ndf['{target_col}'].value_counts().plot(kind='bar')\nplt.title('Target Distribution')\nplt.show()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Feature Analysis"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"top_features = {feature_names}\ndf[top_features].hist(bins=20, figsize=(15, 10))\nplt.tight_layout()\nplt.show()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## ML Training Code"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": starter_code if starter_code else "# Starter code not generated yet"
        },
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    return notebook


def generate_eda_report(profile: Dict, top_features: List[Dict], dataset_summary: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Exploratory Data Analysis Report",
        f"Generated: {ts}",
        "",
        "## Dataset Summary",
        dataset_summary,
        "",
        f"## Shape: {profile['shape']['rows']:,} rows × {profile['shape']['cols']} columns",
        "",
        "## Column Summary",
        "",
        "| Column | Type | Missing% | Unique | Notes |",
        "|--------|------|----------|--------|-------|",
    ]
    for col, info in profile["columns"].items():
        note = ""
        if info["missing_pct"] > 30:
            note = "⚠️ High missing"
        elif info["col_type"] == "high_cardinality_text":
            note = "High cardinality"
        lines.append(f"| {col} | {info['col_type']} | {info['missing_pct']}% | {info['unique_count']} | {note} |")

    if top_features:
        lines += ["", "## Top Predictive Features", ""]
        for i, f in enumerate(top_features[:10], 1):
            lines.append(f"{i}. **{f['feature']}** (score: {f['combined_score']:.4f})")

    return "\n".join(lines)
