"""Synthetic patient data generator.

IMPORTANT: this is SYNTHETIC data so the project runs anywhere with no PHI/licensing issues.
It deliberately encodes realistic equity problems (differential missingness, under-diagnosis)
so the fairness tooling has something real to detect. To use real data, produce a CSV with the
same columns (see config.yaml) and point `data.path` at it -- nothing else changes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

GROUPS = ["Group A", "Group B", "Group C", "Group D"]
GROUP_P = [0.55, 0.20, 0.15, 0.10]
CHEST_PAIN = ["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n: int = 20000, seed: int = 42, drift: bool = False,
                     start: str = "2023-01-01", days: int = 730) -> pd.DataFrame:
    """drift=True simulates covariate shift (older/heavier cohort, more missing labs)
    AND concept drift (risk factors weigh differently), for monitoring demos."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(60 if drift else 54, 13, n), 20, 90).round()
    sex = rng.choice(["M", "F"], n)
    grp = rng.choice(GROUPS, n, p=GROUP_P)
    bmi = np.clip(rng.normal(31.5 if drift else 28, 5.5, n) + 0.02 * (age - 54), 16, 55)
    resting_bp = np.clip(105 + 0.45 * age + 0.3 * (bmi - 25) + rng.normal(0, 14, n), 80, 220)
    cholesterol = np.clip(170 + 0.6 * age + rng.normal(0, 35, n), 100, 450)
    diabetes = rng.binomial(1, _sigmoid(-4.2 + 0.03 * age + 0.07 * (bmi - 25)))
    fasting_glucose = np.clip(88 + 40 * diabetes + rng.normal(0, 14, n), 55, 400)
    smoker = rng.binomial(1, 0.22, n)
    family_history = rng.binomial(1, 0.25, n)
    max_hr = np.clip(208 - 0.7 * age + rng.normal(0, 12, n), 70, 210)
    exercise_angina = rng.binomial(1, _sigmoid(-2.2 + 0.02 * (age - 50)))
    st_dep = np.clip(rng.exponential(0.9, n), 0, 6)
    cp = rng.choice(CHEST_PAIN, n, p=[0.15, 0.25, 0.30, 0.30])
    cp_eff = pd.Series(cp).map({"typical_angina": 1.1, "atypical_angina": -0.2,
                                "non_anginal": 0.0, "asymptomatic": 0.6}).to_numpy()

    w = 1.25 if drift else 1.0   # concept drift: risk factors matter more/differently
    logit = (0.065 * (age - 54) + 0.6 * (sex == "M") + 0.018 * (resting_bp - 125) * w
             + 0.008 * (cholesterol - 200) + 0.5 * smoker + 0.7 * diabetes
             + 0.5 * family_history + 0.45 * (bmi > 30) + cp_eff + 0.9 * exercise_angina
             + 0.7 * st_dep - 0.015 * (max_hr - 150))
    # shift intercept so prevalence is ~20%
    lo, hi = -10, 10
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(logit + mid).mean() < 0.20 else (lo, mid)
    true_y = rng.binomial(1, _sigmoid(logit + (lo + hi) / 2))

    # --- realistic structural inequities -------------------------------------------------
    is_c = grp == "Group C"
    # (1) under-diagnosis: true disease frequently never recorded for Group C
    recorded = true_y.copy()
    missed = is_c & (true_y == 1) & (rng.random(n) < 0.15)
    recorded[missed] = 0
    # (2) differential missingness: lab not ordered as often (access to care)
    chol_obs = cholesterol.copy()
    chol_obs[rng.random(n) < np.where(is_c, 0.25, 0.05)] = np.nan
    if drift:
        chol_obs[rng.random(n) < 0.15] = np.nan
    gl_obs = fasting_glucose.copy()
    gl_obs[rng.random(n) < 0.04] = np.nan

    dates = pd.to_datetime(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    df = pd.DataFrame({
        "patient_id": [f"P{seed}-{i:07d}" for i in range(n)],
        "admission_date": dates,
        "age": age, "sex": sex, "race_ethnicity": grp, "bmi": bmi.round(1),
        "resting_bp": resting_bp.round(), "cholesterol": np.round(chol_obs),
        "fasting_glucose": np.round(gl_obs), "max_heart_rate": max_hr.round(),
        "st_depression": st_dep.round(1), "smoker": smoker, "diabetes": diabetes,
        "family_history": family_history, "exercise_angina": exercise_angina,
        "chest_pain_type": cp,
        "heart_disease": recorded,          # what the EHR says (biased label)
        "true_disease": true_y,             # oracle: only exists in synthetic data
    })
    return df.sort_values("admission_date").reset_index(drop=True)


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_band"] = pd.cut(df["age"], [0, 44, 59, 74, 200],
                            labels=["<45", "45-59", "60-74", "75+"]).astype(str)
    return df


def temporal_split(df: pd.DataFrame, train: float, valid: float):
    df = df.sort_values("admission_date").reset_index(drop=True)
    i, j = int(len(df) * train), int(len(df) * (train + valid))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()
