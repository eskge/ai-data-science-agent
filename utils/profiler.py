"""
Deterministic Data Profiling Engine
All operations here are rule-based - no LLM involvement.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Full dataset profile - deterministic."""
    profile = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "columns": {},
        "duplicates": {
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_cols": _find_duplicate_columns(df),
        },
        "quality_flags": {},
    }

    for col in df.columns:
        profile["columns"][col] = _profile_column(df, col)

    profile["quality_flags"] = _detect_quality_issues(df, profile["columns"])
    profile["outliers"] = _detect_outliers(df)
    profile["health_score"] = _compute_health_score(df, profile)

    return profile


def _profile_column(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    series = df[col]
    n = len(series)
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))

    info = {
        "dtype": str(series.dtype),
        "missing_count": missing,
        "missing_pct": round(missing / n * 100, 2) if n > 0 else 0,
        "unique_count": unique,
        "cardinality_pct": round(unique / (n - missing) * 100, 2) if (n - missing) > 0 else 0,
        "sample_values": series.dropna().head(5).tolist(),
        "col_type": _classify_column(series),
    }

    if pd.api.types.is_numeric_dtype(series):
        try:
            desc = series.describe()
        except Exception:
            desc = pd.Series(dtype=float)

        def _safe(key):
            val = desc.get(key, None)
            if val is None:
                return None
            try:
                f = float(val)
                return None if pd.isna(f) else round(f, 4)
            except (TypeError, ValueError):
                return None

        skew_val = None
        try:
            s = series.skew()
            skew_val = round(float(s), 4) if not pd.isna(s) else None
        except Exception:
            pass

        info.update({
            "mean":     _safe("mean"),
            "std":      _safe("std"),
            "min":      _safe("min"),
            "max":      _safe("max"),
            "q25":      _safe("25%"),
            "median":   _safe("50%"),
            "q75":      _safe("75%"),
            "skewness": skew_val,
        })
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        vc = series.value_counts()
        info["top_values"] = vc.head(5).to_dict()

    return info


def _classify_column(series: pd.Series) -> str:
    # Boolean columns — treat as categorical
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique <= 2:
            return "categorical_numeric"
        if n_unique <= 10:
            try:
                if series.dropna().apply(lambda x: float(x) == int(float(x))).all():
                    return "categorical_numeric"
            except Exception:
                pass
        return "numerical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Try parsing as date
    if pd.api.types.is_object_dtype(series):
        sample = series.dropna().head(50)
        if len(sample) >= 5:
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.8:   # 80%+ must parse as dates
                    return "potential_datetime"
            except Exception:
                pass
        # Check cardinality for categorical
        if series.nunique() / max(len(series), 1) < 0.05:
            return "categorical"
        if series.nunique() / max(len(series), 1) > 0.9:
            return "high_cardinality_text"
    return "text"


def _find_duplicate_columns(df: pd.DataFrame) -> List[str]:
    dupes = []
    seen = {}
    # Use a fast hash first (sample), then verify with full column only if hash matches
    for col in df.columns:
        try:
            sample = df[col].fillna("__NA__").astype(str)
            # Quick fingerprint: hash of first 500 + last 500 + value_counts hash
            fingerprint = hash((tuple(sample.iloc[:500]), tuple(sample.iloc[-500:]), tuple(sample.value_counts().items())))
            if fingerprint in seen:
                # Full verification before flagging
                if (df[col].fillna("__NA__").astype(str) == df[seen[fingerprint]].fillna("__NA__").astype(str)).all():
                    dupes.append(f"{col} == {seen[fingerprint]}")
            else:
                seen[fingerprint] = col
        except Exception:
            continue
    return dupes


def _detect_quality_issues(df: pd.DataFrame, col_profiles: Dict) -> Dict[str, List[str]]:
    flags = {
        "constant_cols": [],
        "near_constant_cols": [],
        "high_cardinality_cols": [],
        "high_missing_cols": [],
        "potential_id_cols": [],
        "potential_target_cols": [],
        "type_mismatch_cols": [],
    }
    for col, info in col_profiles.items():
        if info["unique_count"] <= 1:
            flags["constant_cols"].append(col)
        elif info["unique_count"] == 2:
            pass  # Binary columns are valid — don't flag as near-constant
        elif info["cardinality_pct"] < 1.0 and info["unique_count"] > 1:
            flags["near_constant_cols"].append(col)
        if info["cardinality_pct"] > 95 and info["unique_count"] > 100:
            flags["high_cardinality_cols"].append(col)
            if "id" in col.lower() or "key" in col.lower() or "uuid" in col.lower():
                flags["potential_id_cols"].append(col)
        if info["missing_pct"] > 30:
            flags["high_missing_cols"].append(col)
        # Detect numeric stored as string
        if info["col_type"] == "text" and info["missing_pct"] < 50:
            sample = df[col].dropna().head(50)
            numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
            if numeric_count / len(sample) > 0.8:
                flags["type_mismatch_cols"].append(col)
        # Potential binary target
        if info["col_type"] in ("categorical", "categorical_numeric") and info["unique_count"] == 2:
            name_lower = col.lower()
            if any(k in name_lower for k in ["churn", "target", "label", "fraud", "default", "outcome", "status", "flag"]):
                flags["potential_target_cols"].append(col)

    return {k: v for k, v in flags.items() if v}


def _detect_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    results = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        # Skip boolean-like columns
        if series.nunique() <= 2:
            continue
        try:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            iqr_outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

            z_scores = np.abs(stats.zscore(series))
            z_outliers = int((z_scores > 3).sum())

            if iqr_outliers > 0 or z_outliers > 0:
                results[col] = {
                    "iqr_outliers": iqr_outliers,
                    "iqr_pct": round(iqr_outliers / len(series) * 100, 2),
                    "zscore_outliers": z_outliers,
                    "zscore_pct": round(z_outliers / len(series) * 100, 2),
                    "lower_bound": round(float(q1 - 1.5 * iqr), 4),
                    "upper_bound": round(float(q3 + 1.5 * iqr), 4),
                }
        except Exception:
            continue
    return results


def _compute_health_score(df: pd.DataFrame, profile: Dict) -> Dict[str, Any]:
    score = 100
    breakdown = {}
    n_rows, n_cols = profile["shape"]["rows"], profile["shape"]["cols"]

    # Missing values (up to -25)
    total_cells = n_rows * n_cols
    missing_cells = sum(c["missing_count"] for c in profile["columns"].values())
    missing_pct = missing_cells / max(total_cells, 1) * 100
    if missing_pct == 0:
        missing_deduct = 0
        missing_label = "Excellent"
    elif missing_pct < 5:
        missing_deduct = 5
        missing_label = "Good"
    elif missing_pct < 15:
        missing_deduct = 12
        missing_label = "Moderate"
    else:
        missing_deduct = 25
        missing_label = "Poor"
    score -= missing_deduct
    breakdown["Missing Values"] = {"label": missing_label, "deduction": missing_deduct, "detail": f"{missing_pct:.1f}% cells missing"}

    # Duplicates (up to -15)
    dup_pct = profile["duplicates"]["duplicate_rows"] / max(n_rows, 1) * 100
    if dup_pct == 0:
        dup_deduct = 0
        dup_label = "Excellent"
    elif dup_pct < 2:
        dup_deduct = 5
        dup_label = "Good"
    elif dup_pct < 10:
        dup_deduct = 10
        dup_label = "Moderate"
    else:
        dup_deduct = 15
        dup_label = "Poor"
    score -= dup_deduct
    breakdown["Duplicates"] = {"label": dup_label, "deduction": dup_deduct, "detail": f"{profile['duplicates']['duplicate_rows']} duplicate rows"}

    # Outliers (up to -15)
    outlier_cols = len(profile["outliers"])
    if outlier_cols == 0:
        out_deduct = 0
        out_label = "Excellent"
    elif outlier_cols <= 2:
        out_deduct = 5
        out_label = "Good"
    elif outlier_cols <= 5:
        out_deduct = 10
        out_label = "Moderate"
    else:
        out_deduct = 15
        out_label = "Poor"
    score -= out_deduct
    breakdown["Outliers"] = {"label": out_label, "deduction": out_deduct, "detail": f"{outlier_cols} columns with outliers"}

    # Column quality (up to -15)
    flags = profile.get("quality_flags", {})
    quality_issues = len(flags.get("constant_cols", [])) + len(flags.get("type_mismatch_cols", []))
    if quality_issues == 0:
        col_deduct = 0
        col_label = "Excellent"
    elif quality_issues <= 2:
        col_deduct = 5
        col_label = "Good"
    else:
        col_deduct = 15
        col_label = "Poor"
    score -= col_deduct
    breakdown["Column Quality"] = {"label": col_label, "deduction": col_deduct, "detail": f"{quality_issues} quality issues"}

    # Data consistency (up to -10)
    high_card = len(flags.get("high_cardinality_cols", []))
    high_missing = len(flags.get("high_missing_cols", []))
    consistency_issues = high_card + high_missing
    if consistency_issues == 0:
        cons_deduct = 0
        cons_label = "Excellent"
    elif consistency_issues <= 2:
        cons_deduct = 5
        cons_label = "Good"
    else:
        cons_deduct = 10
        cons_label = "Moderate"
    score -= cons_deduct
    breakdown["Data Consistency"] = {"label": cons_label, "deduction": cons_deduct, "detail": f"{consistency_issues} consistency concerns"}

    return {"score": max(score, 0), "breakdown": breakdown}


def detect_target_candidates(df: pd.DataFrame, profile: Dict) -> List[Dict[str, Any]]:
    """Heuristic target variable detection."""
    candidates = []
    for col, info in profile["columns"].items():
        score = 0
        task_type = None
        reason = []

        name_lower = col.lower()
        binary_keywords = ["churn", "fraud", "default", "target", "label", "outcome", "flag", "status", "result", "approved", "converted", "cancelled"]
        reg_keywords = ["revenue", "price", "amount", "salary", "cost", "sales", "income", "spend", "value", "score", "rate"]
        multiclass_keywords = ["category", "segment", "class", "type", "tier", "group", "cluster"]

        if info["missing_pct"] > 40:
            continue

        if info["col_type"] == "numerical" and info["unique_count"] > 20:
            if any(k in name_lower for k in reg_keywords):
                score += 3
                task_type = "Regression"
                reason.append("Name suggests a continuous target")
            else:
                score += 1
                task_type = "Regression"
        elif info["col_type"] in ("categorical", "categorical_numeric"):
            if info["unique_count"] == 2:
                if any(k in name_lower for k in binary_keywords):
                    score += 4
                    task_type = "Binary Classification"
                    reason.append("Binary column with target-like name")
                else:
                    score += 2
                    task_type = "Binary Classification"
                    reason.append("Binary column")
            elif 2 < info["unique_count"] <= 20:
                if any(k in name_lower for k in multiclass_keywords):
                    score += 3
                    task_type = "Multi-class Classification"
                    reason.append("Multi-class column with segment-like name")
                else:
                    score += 1
                    task_type = "Multi-class Classification"

        if any(k in name_lower for k in binary_keywords):
            score += 2
            reason.append("Column name matches common target patterns")

        if score >= 2:
            candidates.append({
                "column": col,
                "task_type": task_type or "Unknown",
                "confidence": min(score * 20, 100),
                "reasons": reason,
                "unique_values": info["unique_count"],
                "missing_pct": info["missing_pct"],
            })

    return sorted(candidates, key=lambda x: -x["confidence"])


def assess_ml_readiness(df: pd.DataFrame, profile: Dict, target_col: str = None) -> Dict[str, Any]:
    """ML readiness assessment."""
    n_rows = profile["shape"]["rows"]
    n_cols = profile["shape"]["cols"]
    assessment = {}

    # Sample size
    if n_rows < 100:
        assessment["sample_size"] = {"score": 20, "label": "Insufficient", "detail": f"{n_rows} rows — minimum 100+ recommended"}
    elif n_rows < 1000:
        assessment["sample_size"] = {"score": 60, "label": "Borderline", "detail": f"{n_rows} rows — 1,000+ preferred for robust ML"}
    elif n_rows < 10000:
        assessment["sample_size"] = {"score": 80, "label": "Adequate", "detail": f"{n_rows} rows — good for most algorithms"}
    else:
        assessment["sample_size"] = {"score": 100, "label": "Strong", "detail": f"{n_rows:,} rows — ample training data"}

    # Missing values
    total_cells = n_rows * n_cols
    missing_cells = sum(c["missing_count"] for c in profile["columns"].values())
    missing_pct = missing_cells / max(total_cells, 1) * 100
    if missing_pct < 5:
        assessment["missing_values"] = {"score": 100, "label": "Minimal", "detail": f"{missing_pct:.1f}% missing — low impact on ML"}
    elif missing_pct < 15:
        assessment["missing_values"] = {"score": 70, "label": "Manageable", "detail": f"{missing_pct:.1f}% missing — imputation required"}
    else:
        assessment["missing_values"] = {"score": 40, "label": "High", "detail": f"{missing_pct:.1f}% missing — may degrade model quality"}

    # Class imbalance (classification only)
    if target_col and target_col in df.columns:
        is_classif = df[target_col].nunique() <= 20 or not pd.api.types.is_numeric_dtype(df[target_col])
        if is_classif:
            vc = df[target_col].value_counts(normalize=True)
            min_class = vc.min()
            if min_class > 0.35:
                assessment["class_balance"] = {"score": 100, "label": "Balanced", "detail": f"Minority class: {min_class*100:.1f}%"}
            elif min_class > 0.15:
                assessment["class_balance"] = {"score": 70, "label": "Mild Imbalance", "detail": f"Minority class: {min_class*100:.1f}% — consider oversampling"}
            else:
                assessment["class_balance"] = {"score": 40, "label": "Imbalanced", "detail": f"Minority class: {min_class*100:.1f}% — SMOTE or class weights needed"}
        else:
            # Regression target: check for distribution skew instead
            try:
                skew = abs(float(df[target_col].skew()))
                if skew < 1:
                    assessment["target_distribution"] = {"score": 100, "label": "Normal-ish", "detail": f"Skewness {skew:.2f} — good for regression"}
                elif skew < 2:
                    assessment["target_distribution"] = {"score": 70, "label": "Skewed", "detail": f"Skewness {skew:.2f} — log transform may help"}
                else:
                    assessment["target_distribution"] = {"score": 40, "label": "Highly Skewed", "detail": f"Skewness {skew:.2f} — log/sqrt transform recommended"}
            except Exception:
                pass

    # Feature quality
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 3:
        assessment["feature_quality"] = {"score": 80, "label": "Good", "detail": f"{len(numeric_cols)} numeric features available"}
    elif len(numeric_cols) >= 1:
        assessment["feature_quality"] = {"score": 60, "label": "Moderate", "detail": f"{len(numeric_cols)} numeric features — encoding may be needed"}
    else:
        assessment["feature_quality"] = {"score": 30, "label": "Limited", "detail": "No numeric features — heavy encoding required"}

    overall = int(np.mean([v["score"] for v in assessment.values()]))
    assessment["overall"] = overall
    return assessment


def detect_leakage(df: pd.DataFrame, profile: Dict, target_col: str = None) -> List[Dict[str, Any]]:
    """Data leakage detection."""
    warnings_list = []
    for col, info in profile["columns"].items():
        name_lower = col.lower()
        # ID/unique identifier
        if info["cardinality_pct"] > 95 and info["unique_count"] > 50:
            warnings_list.append({
                "column": col,
                "type": "Unique Identifier",
                "severity": "High",
                "message": f"'{col}' appears to be a unique identifier and should be excluded from ML features.",
            })
        # Target leakage keywords
        leakage_keywords = ["payment_completed", "paid", "churn_date", "cancel_date", "outcome_date"]
        if any(k in name_lower for k in leakage_keywords):
            warnings_list.append({
                "column": col,
                "type": "Potential Target Leakage",
                "severity": "High",
                "message": f"'{col}' may contain future information that leaks the target variable.",
            })
        # High correlation with target (if provided)
        if target_col and target_col in df.columns and target_col != col:
            try:
                target_numeric = pd.to_numeric(df[target_col], errors="coerce")
                col_numeric = pd.to_numeric(df[col], errors="coerce")
                if col_numeric.notna().sum() > 10 and target_numeric.notna().sum() > 10:
                    corr = abs(col_numeric.corr(target_numeric))
                    if corr > 0.95:
                        warnings_list.append({
                            "column": col,
                            "type": "Suspicious Correlation",
                            "severity": "Medium",
                            "message": f"'{col}' has {corr:.2f} correlation with target — check for data leakage.",
                        })
            except Exception:
                pass

    return warnings_list


def compute_feature_importance(df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
    """Feature importance using correlation and mutual information heuristics."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder

    features = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if not numeric_cols:
        return []

    X = df[numeric_cols].copy()
    y = df[target_col].copy()

    # Fill missing values for computation
    X = X.fillna(X.median())
    y_encoded = y.copy()
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str).fillna("missing")))

    # Determine task type
    is_classification = y.nunique() <= 20 or y.dtype == object

    try:
        if is_classification:
            mi = mutual_info_classif(X, y_encoded.fillna(0), random_state=42)
        else:
            mi = mutual_info_regression(X, y_encoded.fillna(0), random_state=42)

        corrs = [abs(X[col].corr(y_encoded.fillna(0))) for col in numeric_cols]

        for i, col in enumerate(numeric_cols):
            features.append({
                "feature": col,
                "mutual_info": round(float(mi[i]), 4),
                "correlation": round(float(corrs[i]) if not np.isnan(corrs[i]) else 0, 4),
                "combined_score": round((float(mi[i]) + (corrs[i] if not np.isnan(corrs[i]) else 0)) / 2, 4),
            })
    except Exception:
        for col in numeric_cols:
            try:
                corr = abs(df[col].corr(y_encoded.fillna(0)))
            except Exception:
                corr = 0
            features.append({
                "feature": col,
                "mutual_info": 0,
                "correlation": round(float(corr) if not np.isnan(corr) else 0, 4),
                "combined_score": round(float(corr) if not np.isnan(corr) else 0, 4),
            })

    return sorted(features, key=lambda x: -x["combined_score"])[:15]
