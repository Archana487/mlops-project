"""Subgroup performance + fairness metrics (dependency-free; mirrors Fairlearn definitions)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def group_metrics(y, p, thr: float, groups) -> pd.DataFrame:
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rows = []
    for g in sorted(set(groups)):
        m = groups == g
        yt, pt = y[m], p[m]
        pred = pt >= thr
        tp, fp = int((pred & (yt == 1)).sum()), int((pred & (yt == 0)).sum())
        fn, tn = int((~pred & (yt == 1)).sum()), int((~pred & (yt == 0)).sum())
        rows.append({
            "group": g, "n": int(m.sum()), "prevalence": yt.mean(),
            "selection_rate": pred.mean(),
            "tpr": tp / (tp + fn) if tp + fn else np.nan,
            "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "auc": roc_auc_score(yt, pt) if len(set(yt)) > 1 else np.nan,
            "mean_pred": pt.mean(),
            "calibration_gap": pt.mean() - yt.mean(),
        })
    return pd.DataFrame(rows)


def disparities(gm: pd.DataFrame, min_n: int = 1) -> dict:
    g = gm[gm["n"] >= min_n]
    rng = lambda c: float(g[c].max() - g[c].min()) if len(g) > 1 else 0.0
    return {
        "demographic_parity_gap": rng("selection_rate"),
        "equal_opportunity_gap": rng("tpr"),
        "equalized_odds_gap": max(rng("tpr"), rng("fpr")),
        "auc_gap": rng("auc"),
        "max_abs_calibration_gap": float(g["calibration_gap"].abs().max()),
        "min_group_auc": float(g["auc"].min()),
        "worst_tpr_group": str(g.loc[g["tpr"].idxmin(), "group"]) if len(g) else None,
    }


def fairness_gate(gm: pd.DataFrame, gates: dict, policy: str = "full") -> tuple[bool, list[str], dict]:
    d = disparities(gm, gates["min_group_size"])
    fails = []
    if policy == "full" and d["equal_opportunity_gap"] > gates["max_equal_opportunity_gap"]:
        fails.append(f"equal-opportunity gap {d['equal_opportunity_gap']:.3f} > {gates['max_equal_opportunity_gap']}"
                     f" (worst TPR: {d['worst_tpr_group']})")
    if d["max_abs_calibration_gap"] > gates["max_calibration_gap"]:
        fails.append(f"calibration gap {d['max_abs_calibration_gap']:.3f} > {gates['max_calibration_gap']}")
    if d["min_group_auc"] < gates["min_group_auc"]:
        fails.append(f"min group AUC {d['min_group_auc']:.3f} < {gates['min_group_auc']}")
    small = gm[gm["n"] < gates["min_group_size"]]["group"].tolist()
    return (not fails), fails, {**d, "groups_too_small_to_assess": small}
