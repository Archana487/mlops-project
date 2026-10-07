"""Data validation gate. Fails the pipeline early instead of training on garbage."""
from __future__ import annotations
import pandas as pd

RANGES = {
    "age": (18, 110), "resting_bp": (60, 260), "cholesterol": (80, 600),
    "fasting_glucose": (40, 600), "max_heart_rate": (40, 230), "bmi": (10, 80),
    "st_depression": (-1, 10),
}
CATEGORIES = {"sex": {"M", "F"},
              "chest_pain_type": {"typical_angina", "atypical_angina", "non_anginal", "asymptomatic"}}
MAX_MISSING = {"cholesterol": 0.40, "fasting_glucose": 0.20}   # others must be ~complete


class DataValidationError(ValueError):
    pass


def validate_dataframe(df: pd.DataFrame, cfg: dict, require_target: bool = True) -> dict:
    f = cfg["features"]
    cols = f["numeric"] + f["binary"] + f["categorical"]
    errors, warnings = [], []
    required = cols + ([cfg["target"]] if require_target else []) + ["admission_date"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing columns: {missing_cols}")
    if len(df) < 1000:
        errors.append(f"Too few rows ({len(df)})")
    if df.get("patient_id") is not None and df["patient_id"].duplicated().any():
        errors.append("Duplicate patient_id values")
    for c, (lo, hi) in RANGES.items():
        bad = ((df[c] < lo) | (df[c] > hi)).sum()
        if bad / len(df) > 0.001:
            errors.append(f"{c}: {bad} values outside [{lo},{hi}]")
    for c, allowed in CATEGORIES.items():
        unk = set(df[c].dropna().unique()) - allowed
        if unk:
            errors.append(f"{c}: unexpected categories {unk}")
    for c in f["binary"]:
        if not set(df[c].dropna().unique()) <= {0, 1}:
            errors.append(f"{c}: not binary")
    for c in cols:
        rate = df[c].isna().mean()
        if rate > MAX_MISSING.get(c, 0.02):
            errors.append(f"{c}: missing rate {rate:.1%} exceeds limit")
    if require_target:
        prev = df[cfg["target"]].mean()
        if not 0.02 <= prev <= 0.60:
            errors.append(f"Target prevalence {prev:.1%} implausible")
        # leakage guard: nothing but the target may be a near-perfect predictor
        for c in f["numeric"] + f["binary"]:
            corr = df[[c, cfg["target"]]].corr().iloc[0, 1]
            if abs(corr) > 0.95:
                errors.append(f"{c}: |corr|={corr:.2f} with target -> possible leakage")
    report = {"n_rows": len(df), "errors": errors, "warnings": warnings, "passed": not errors}
    if errors:
        raise DataValidationError("; ".join(errors))
    return report
