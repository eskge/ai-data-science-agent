"""
Deterministic Data Cleaning Engine
All cleaning is done via Pandas only.
LLM proposes; this module executes.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats


def generate_cleaning_plan(df: pd.DataFrame, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a deterministic cleaning plan based on profiling results.
    Returns a list of proposed actions (NOT yet executed).
    """
    plan = []
    step_id = 1

    # 1. Remove duplicate rows
    dup_count = profile["duplicates"]["duplicate_rows"]
    if dup_count > 0:
        plan.append({
            "id": step_id,
            "action": "remove_duplicates",
            "title": f"Remove {dup_count} duplicate rows",
            "reason": f"Duplicate rows can bias model training and inflate evaluation metrics.",
            "impact": f"Dataset will shrink from {len(df)} to {len(df) - dup_count} rows.",
            "risk": "Low",
            "params": {},
        })
        step_id += 1

    # 2. Handle missing values per column
    for col, col_info in profile["columns"].items():
        if col_info["missing_count"] == 0:
            continue
        missing_pct = col_info["missing_pct"]
        col_type = col_info["col_type"]

        if missing_pct > 60:
            plan.append({
                "id": step_id,
                "action": "drop_column",
                "title": f"Drop column '{col}' ({missing_pct:.1f}% missing)",
                "reason": f"Over 60% of values are missing — column is unlikely to provide reliable signal.",
                "impact": f"Column '{col}' will be removed.",
                "risk": "Medium",
                "params": {"column": col},
            })
            step_id += 1
        elif col_type == "numerical":
            plan.append({
                "id": step_id,
                "action": "fill_median",
                "title": f"Fill '{col}' missing values with median",
                "reason": f"{missing_pct:.1f}% missing. Median imputation is robust to outliers.",
                "impact": f"{col_info['missing_count']} values will be filled.",
                "risk": "Low",
                "params": {"column": col},
            })
            step_id += 1
        elif col_type in ("categorical", "categorical_numeric", "potential_datetime", "text"):
            plan.append({
                "id": step_id,
                "action": "fill_mode",
                "title": f"Fill '{col}' missing values with mode",
                "reason": f"{missing_pct:.1f}% missing. Mode imputation preserves distribution for categorical/text data.",
                "impact": f"{col_info['missing_count']} values will be filled.",
                "risk": "Low",
                "params": {"column": col},
            })
            step_id += 1

    # 3. Fix type mismatches
    flags = profile.get("quality_flags", {})
    for col in flags.get("type_mismatch_cols", []):
        plan.append({
            "id": step_id,
            "action": "convert_numeric",
            "title": f"Convert '{col}' from text to numeric",
            "reason": f"Column contains numeric data stored as strings. Conversion needed for ML.",
            "impact": f"Non-numeric values in '{col}' will become NaN.",
            "risk": "Low",
            "params": {"column": col},
        })
        step_id += 1

    # 4. Cap outliers using IQR
    for col, outlier_info in profile.get("outliers", {}).items():
        if outlier_info["iqr_pct"] > 5:
            plan.append({
                "id": step_id,
                "action": "cap_outliers_iqr",
                "title": f"Cap outliers in '{col}' using IQR method",
                "reason": f"{outlier_info['iqr_pct']:.1f}% of values are outliers. Capping prevents extreme values from dominating models.",
                "impact": f"Values below {outlier_info['lower_bound']:.2f} and above {outlier_info['upper_bound']:.2f} will be clipped.",
                "risk": "Medium",
                "params": {
                    "column": col,
                    "lower": outlier_info["lower_bound"],
                    "upper": outlier_info["upper_bound"],
                },
            })
            step_id += 1

    # 5. Drop constant columns
    for col in flags.get("constant_cols", []):
        plan.append({
            "id": step_id,
            "action": "drop_column",
            "title": f"Drop constant column '{col}'",
            "reason": "Column has only one unique value and provides no predictive information.",
            "impact": f"Column '{col}' will be removed.",
            "risk": "Low",
            "params": {"column": col},
        })
        step_id += 1

    return plan


def execute_cleaning_plan(df: pd.DataFrame, approved_steps: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Execute approved cleaning steps using Pandas only.
    Returns (cleaned_df, cleaning_log).
    """
    cleaned_df = df.copy()
    log = []

    for step in approved_steps:
        action = step["action"]
        params = step.get("params", {})
        before_shape = cleaned_df.shape
        rows_affected = 0
        cols_affected = []
        detail = ""

        try:
            if action == "remove_duplicates":
                before = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
                rows_affected = before - len(cleaned_df)
                detail = f"Removed {rows_affected} duplicate rows"

            elif action == "drop_column":
                col = params["column"]
                if col in cleaned_df.columns:
                    cleaned_df = cleaned_df.drop(columns=[col])
                    cols_affected = [col]
                    detail = f"Dropped column '{col}'"

            elif action == "fill_median":
                col = params["column"]
                if col in cleaned_df.columns:
                    median_val = cleaned_df[col].median()
                    rows_affected = int(cleaned_df[col].isna().sum())
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                    cols_affected = [col]
                    detail = f"Filled {rows_affected} missing values with median ({median_val:.4f})"

            elif action == "fill_mode":
                col = params["column"]
                if col in cleaned_df.columns:
                    mode_val = cleaned_df[col].mode()
                    if len(mode_val) > 0:
                        mode_val = mode_val[0]
                        rows_affected = int(cleaned_df[col].isna().sum())
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                        cols_affected = [col]
                        detail = f"Filled {rows_affected} missing values with mode ('{mode_val}')"

            elif action == "convert_numeric":
                col = params["column"]
                if col in cleaned_df.columns:
                    before_nulls = cleaned_df[col].isna().sum()
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
                    after_nulls = cleaned_df[col].isna().sum()
                    rows_affected = int(after_nulls - before_nulls)
                    cols_affected = [col]
                    detail = f"Converted '{col}' to numeric ({rows_affected} non-numeric values → NaN)"

            elif action == "cap_outliers_iqr":
                col = params["column"]
                lower, upper = params["lower"], params["upper"]
                if col in cleaned_df.columns:
                    before_outliers = ((cleaned_df[col] < lower) | (cleaned_df[col] > upper)).sum()
                    cleaned_df[col] = cleaned_df[col].clip(lower=lower, upper=upper)
                    rows_affected = int(before_outliers)
                    cols_affected = [col]
                    detail = f"Clipped {rows_affected} outlier values in '{col}' to [{lower:.2f}, {upper:.2f}]"

            log.append({
                "step": step["id"],
                "title": step["title"],
                "status": "Success",
                "rows_affected": rows_affected,
                "cols_affected": cols_affected,
                "before_shape": before_shape,
                "after_shape": cleaned_df.shape,
                "detail": detail,
            })

        except Exception as e:
            log.append({
                "step": step["id"],
                "title": step["title"],
                "status": f"Error: {str(e)}",
                "rows_affected": 0,
                "cols_affected": [],
                "before_shape": before_shape,
                "after_shape": cleaned_df.shape,
                "detail": str(e),
            })

    return cleaned_df, log
