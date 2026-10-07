"""Production monitoring: data drift (PSI), score drift, delayed-label performance,
subgroup performance, and human-in-the-loop health (override rate, review backlog)."""
from __future__ import annotations
import html, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import store
from .config import load_config, reports_dir
from .fairness import expected_calibration_error, group_metrics, disparities


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10, eps: float = 1e-4) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if ref.empty or cur.empty:
        return float("nan")
    if ref.dtype == object or ref.nunique() <= 5:
        cats = sorted(set(ref) | set(cur))
        r = ref.value_counts(normalize=True).reindex(cats, fill_value=0) + eps
        c = cur.value_counts(normalize=True).reindex(cats, fill_value=0) + eps
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = pd.Series(np.histogram(ref, edges)[0] / len(ref)) + eps
        c = pd.Series(np.histogram(cur, edges)[0] / len(cur)) + eps
    return float(((c - r) * np.log(c / r)).sum())


def status(v: float, warn: float, alert: float) -> str:
    return "alert" if v >= alert else "warn" if v >= warn else "ok"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> dict:
    m = cfg["monitoring"]
    cols = cfg["features"]["numeric"] + cfg["features"]["binary"] + cfg["features"]["categorical"] + ["score"]
    out = {}
    for c in cols:
        if c in current and c in reference:
            v = psi(reference[c], current[c])
            out[c] = {"psi": round(v, 4), "status": status(v, m["psi_warn"], m["psi_alert"])}
    # missingness drift (labs not ordered is a classic silent failure)
    for c in ("cholesterol", "fasting_glucose"):
        if c in current:
            rm, cm = reference[c].isna().mean(), current[c].isna().mean()
            out[c]["missing_rate"] = {"reference": round(rm, 3), "current": round(cm, 3)}
            if abs(cm - rm) > 0.10:
                out[c]["status"] = "alert"
    return out


def performance_report(bundle_meta: dict, cfg: dict) -> dict:
    m, thr = cfg["monitoring"], bundle_meta["thresholds"]["high"]
    ldf = store.labeled_df()
    if len(ldf) < m["min_labeled_for_performance"]:
        return {"status": "insufficient_labels", "n_labeled": int(len(ldf)),
                "needed": m["min_labeled_for_performance"]}
    y, p = ldf["label"].to_numpy(), ldf["score"].to_numpy()
    pred = p >= thr
    sens = float((pred & (y == 1)).sum() / max((y == 1).sum(), 1))
    auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    base = bundle_meta["test_metrics"]
    alerts = []
    if base["auc"] - auc > m["auc_drop_alert"]:
        alerts.append(f"AUC dropped {base['auc']:.3f}->{auc:.3f}")
    if base["sensitivity"] - sens > m["sensitivity_drop_alert"]:
        alerts.append(f"sensitivity dropped {base['sensitivity']:.3f}->{sens:.3f}")
    rep = {"n_labeled": int(len(ldf)), "auc": round(auc, 4), "sensitivity": round(sens, 4),
           "ece": round(expected_calibration_error(y, p), 4), "baseline_auc": round(base["auc"], 4),
           "baseline_sensitivity": round(base["sensitivity"], 4), "alerts": alerts,
           "status": "alert" if alerts else "ok"}
    g = ldf.dropna(subset=["race_ethnicity"])
    if len(g):
        gm = group_metrics(g["label"], g["score"], thr, g["race_ethnicity"])
        rep["subgroups"] = gm.round(4).to_dict("records")
        rep["fairness"] = disparities(gm, min_n=30)
    return rep


def hitl_report(cfg: dict) -> dict:
    m = cfg["monitoring"]
    preds, rev = store.predictions_df(), store.reviews_df()
    need = preds[preds["requires_review"] == 1] if len(preds) else preds
    rep = {"n_predictions": int(len(preds)), "n_requiring_review": int(len(need)),
           "n_reviewed": int(len(rev)), "backlog": int(len(need) - len(rev)),
           "tier_mix": preds["tier"].value_counts(normalize=True).round(3).to_dict() if len(preds) else {}}
    if len(rev):
        dec = rev["decision"].value_counts(normalize=True)
        override = float(dec.filter(like="override").sum())
        lat = (pd.to_datetime(rev["ts"]) - pd.to_datetime(rev["pred_ts"])).dt.total_seconds() / 3600
        rep.update(override_rate=round(override, 3), decision_mix=dec.round(3).to_dict(),
                   median_review_hours=round(float(lat.median()), 3),
                   status="alert" if override > m["max_override_rate"] else "ok")
        if override > m["max_override_rate"]:
            rep["alert"] = "Clinicians override the model unusually often -- investigate model/data."
    else:
        rep["status"] = "no_reviews"
    return rep


def run_monitoring(bundle: dict, cfg: dict | None = None, write: bool = True) -> dict:
    cfg = cfg or load_config()
    meta, ref = bundle["meta"], bundle["reference"]
    cur = store.predictions_df()
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model_version": meta["version"], "n_current": int(len(cur))}
    report["drift"] = drift_report(ref, cur, cfg) if len(cur) >= 100 else {"status": "insufficient_data"}
    report["performance"] = performance_report(meta, cfg)
    report["human_in_the_loop"] = hitl_report(cfg)
    flagged = [k for k, v in report["drift"].items() if isinstance(v, dict) and v.get("status") == "alert"]
    perf_alert = report["performance"].get("status") == "alert"
    report["overall_status"] = "alert" if flagged or perf_alert or report["human_in_the_loop"].get("status") == "alert" else "ok"
    report["recommendation"] = ("Investigate drift in: " + ", ".join(flagged) + ". Consider retraining on recent data." if flagged
                                else "Performance degraded vs baseline: retrain/review." if perf_alert
                                else "No action needed.")
    if write:
        d = reports_dir()
        (d / "monitoring_latest.json").write_text(json.dumps(report, indent=2, default=float))
        (d / "monitoring_latest.html").write_text(render_html(report))
    return report


def render_html(r: dict) -> str:
    color = {"ok": "#1a7f37", "warn": "#9a6700", "alert": "#cf222e"}
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['psi']}</td>"
                   f"<td style='color:{color[v['status']]};font-weight:600'>{v['status']}</td></tr>"
                   for k, v in r["drift"].items() if isinstance(v, dict) and "psi" in v)
    perf = html.escape(json.dumps({k: v for k, v in r["performance"].items() if k != "subgroups"}, indent=1))
    hitl = html.escape(json.dumps(r["human_in_the_loop"], indent=1))
    return f"""<!doctype html><meta charset=utf-8><title>Monitoring</title>
<body style="font:14px system-ui;max-width:860px;margin:2rem auto">
<h1>Model monitoring &mdash; <span style="color:{color[r['overall_status']]}">{r['overall_status'].upper()}</span></h1>
<p>Model <b>{r['model_version']}</b> &middot; {r['generated_at']} &middot; {r['n_current']} predictions</p>
<p><b>Recommendation:</b> {html.escape(r['recommendation'])}</p>
<h2>Feature drift (PSI: &lt;0.1 ok, 0.1-0.25 warn, &gt;0.25 alert)</h2>
<table border=1 cellpadding=6 style="border-collapse:collapse"><tr><th>feature<th>PSI<th>status</tr>{rows}</table>
<h2>Performance (delayed labels)</h2><pre>{perf}</pre>
<h2>Human-in-the-loop</h2><pre>{hitl}</pre></body>"""
