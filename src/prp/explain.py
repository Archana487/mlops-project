"""Model-agnostic per-patient explanation by occlusion: swap one feature for its reference
(population-typical) value and measure the change in predicted risk. Cheap, no extra deps,
and good enough to give clinicians the 'why'. Not causal."""
from __future__ import annotations
import pandas as pd

LABELS = {"age": "Age", "resting_bp": "Resting BP", "cholesterol": "Cholesterol",
          "fasting_glucose": "Fasting glucose", "max_heart_rate": "Max heart rate", "bmi": "BMI",
          "st_depression": "ST depression", "smoker": "Smoking", "diabetes": "Diabetes",
          "family_history": "Family history", "exercise_angina": "Exercise angina",
          "sex": "Sex", "chest_pain_type": "Chest pain type"}


def explain_rows(model, X: pd.DataFrame, baseline: dict, top_k: int = 4) -> list[list[dict]]:
    base = model.predict_proba(X)[:, 1]
    deltas = {}
    for c in X.columns:
        Xo = X.copy()
        Xo[c] = baseline[c]
        deltas[c] = base - model.predict_proba(Xo)[:, 1]
    out = []
    for i in range(len(X)):
        items = [(c, float(deltas[c][i])) for c in X.columns]
        items.sort(key=lambda t: abs(t[1]), reverse=True)
        row = []
        for c, d in items[:top_k]:
            if abs(d) <= 1e-4:
                continue
            v = X.iloc[i][c]
            v = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
            row.append({"feature": LABELS.get(c, c), "value": v, "effect": round(d, 4),
                        "direction": "raises risk" if d > 0 else "lowers risk"})
        out.append(row)
    return out
