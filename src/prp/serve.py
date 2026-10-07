"""FastAPI model service with human-in-the-loop review endpoints.

Run:  uvicorn prp.serve:app --port 8000
"""
from __future__ import annotations
import os, time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from . import store
from .config import champion_dir, load_config
from .explain import explain_rows
from .monitoring import run_monitoring
from .triage import route

DISCLAIMER = ("Decision-support only. Not a diagnosis. A licensed clinician must make all "
              "clinical decisions. Scores are estimates and may be wrong.")
STATE: dict = {"bundle": None, "counters": {"predictions": 0, "errors": 0, "latency_sum": 0.0}}


def load_bundle():
    path = champion_dir() / "bundle.joblib"
    if not path.exists():
        raise RuntimeError(f"No champion model at {path}. Run `make train` first.")
    STATE["bundle"] = joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_bundle()
    yield


app = FastAPI(title="Patient Risk Prediction API", version="1.0.0", lifespan=lifespan,
              description="Heart-disease risk triage. " + DISCLAIMER)


def require_key(x_api_key: str = Header(default="")):
    if x_api_key != os.environ.get("PRP_REVIEWER_KEY", "dev-reviewer-key"):
        raise HTTPException(401, "Invalid or missing X-API-Key")


class Patient(BaseModel):
    """Pseudonymous clinical features only -- never send names, MRNs or other direct identifiers."""
    age: float = Field(ge=18, le=110)
    sex: Literal["M", "F"]
    resting_bp: float = Field(ge=60, le=260)
    cholesterol: float | None = Field(default=None, ge=80, le=600)
    fasting_glucose: float | None = Field(default=None, ge=40, le=600)
    max_heart_rate: float = Field(ge=40, le=230)
    bmi: float = Field(ge=10, le=80)
    st_depression: float = Field(ge=-1, le=10)
    smoker: Literal[0, 1]
    diabetes: Literal[0, 1]
    family_history: Literal[0, 1]
    exercise_angina: Literal[0, 1]
    chest_pain_type: Literal["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"]
    external_id: str | None = Field(default=None, max_length=64, description="Pseudonymous reference, not an MRN")
    race_ethnicity: str | None = Field(default=None, description="AUDIT ONLY. Stored for fairness monitoring; never used for scoring.")


class Review(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64)
    decision: Literal["confirm", "override_higher", "override_lower", "defer"]
    clinician_risk: Literal["low", "moderate", "high"] | None = None
    notes: str | None = Field(default=None, max_length=2000)


class Outcome(BaseModel):
    label: Literal[0, 1]


@app.get("/health")
def health():
    ok = STATE["bundle"] is not None
    return {"status": "ok" if ok else "no_model", "model_version": ok and STATE["bundle"]["meta"]["version"]}


@app.get("/model-info")
def model_info():
    m = STATE["bundle"]["meta"]
    return {k: m[k] for k in ("version", "model_type", "trained_at", "thresholds", "test_metrics",
                              "promoted", "mlflow_run_id")} | {"fairness": m["fairness_summary"]}


@app.post("/predict")
def predict(p: Patient):
    t0 = time.perf_counter()
    b, cfg = STATE["bundle"], load_config()
    meta, cols = b["meta"], b["meta"]["features"]
    data = p.model_dump()
    X = pd.DataFrame([{c: data[c] for c in cols}])
    try:
        score = float(b["model"].predict_proba(X)[0, 1])
        expl = explain_rows(b["model"], X, meta["baseline"])[0]
    except Exception:
        STATE["counters"]["errors"] += 1
        raise HTTPException(500, "Scoring failed")
    dq = [c for c in ("cholesterol", "fasting_glucose") if data[c] is None]
    pid = os.urandom(8).hex()
    r = route(pid, score, meta["thresholds"], cfg["human_in_the_loop"]["auto_clear_audit_rate"], dq)
    feats = {c: data[c] for c in cols}
    store.save_prediction(feats, score, r, meta["version"], p.external_id, p.race_ethnicity, pid)
    ms = (time.perf_counter() - t0) * 1000
    STATE["counters"]["predictions"] += 1
    STATE["counters"]["latency_sum"] += ms
    return {"prediction_id": pid, "risk_score": round(score, 4), "risk_tier": r["tier"],
            "recommended_action": r["action"], "requires_clinician_review": r["requires_review"],
            "review_queue": r["queue"], "routing_reasons": r["reasons"],
            "top_factors": expl, "data_quality_flags": dq,
            "model_version": meta["version"], "latency_ms": round(ms, 1), "disclaimer": DISCLAIMER}


@app.get("/reviews/pending", dependencies=[Depends(require_key)])
def reviews_pending(limit: int = 50):
    return {"count": len(items := store.pending_reviews(min(limit, 500))), "items": items}


@app.post("/reviews/{prediction_id}", dependencies=[Depends(require_key)])
def submit_review(prediction_id: str, r: Review):
    if store.get_prediction(prediction_id) is None:
        raise HTTPException(404, "Unknown prediction")
    if r.decision.startswith("override") and not (r.notes and r.clinician_risk):
        raise HTTPException(422, "Overrides require clinician_risk and a written rationale in notes")
    store.save_review(prediction_id, r.reviewer, r.decision, r.clinician_risk, r.notes)
    return {"status": "recorded", "prediction_id": prediction_id}


@app.post("/outcomes/{prediction_id}", dependencies=[Depends(require_key)])
def submit_outcome(prediction_id: str, o: Outcome):
    if store.get_prediction(prediction_id) is None:
        raise HTTPException(404, "Unknown prediction")
    store.save_outcome(prediction_id, o.label)
    return {"status": "recorded"}


@app.get("/monitoring/summary", dependencies=[Depends(require_key)])
def monitoring_summary():
    return run_monitoring(STATE["bundle"], write=False)


@app.post("/admin/reload", dependencies=[Depends(require_key)])
def reload_model():
    load_bundle()
    return {"status": "reloaded", "model_version": STATE["bundle"]["meta"]["version"]}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    c = STATE["counters"]
    avg = c["latency_sum"] / c["predictions"] if c["predictions"] else 0
    return (f"prp_predictions_total {c['predictions']}\nprp_errors_total {c['errors']}\n"
            f"prp_latency_ms_avg {avg:.2f}\nprp_pending_reviews {len(store.pending_reviews(10000))}\n")
