# Patient Risk Prediction Pipeline (Heart-Disease Triage)

End-to-end, production-style ML system: **tracked** experiments, **gated** model promotion, **orchestrated** workflows, a **served** API,
**monitoring** for drift/performance, **fairness** auditing, and a **human-in-the-loop** review workflow.
Runs fully offline on synthetic data (swap in real data via `config.yaml`). See [MODEL_CARD.md](MODEL_CARD.md) for results and caveats.

```
 ingest ─► validate ─► train (CV, 2 candidates) ─► calibrate ─► test + fairness audit ─► GATES ─► promote
 (Prefect)   schema/range/    MLflow nested runs     sigmoid        subgroup TPR/FPR/AUC      AUC, ECE,   champion bundle +
             leakage checks                                         calibration               fairness,   MLflow registry alias
                                                                                              vs champion
 ┌───────────────────── serving (FastAPI) ─────────────────────┐        ┌──── monitoring flow (daily) ────┐
 │ /predict → score+tier+top factors → route → SQLite store   │ ─────► │ PSI drift · missingness · score  │
 │ /reviews (clinician confirm/override+rationale) /outcomes   │        │ AUC/sens vs baseline · override  │
 └──────────────────────────────────────────────────────────────┘        │ rate · backlog → alert + HTML    │
```

## Quick start
```bash
make install
make train            # Prefect flow: generates data, validates, trains, gates, promotes (artifacts/champion)
make test             # 16 tests incl. a gate that must BLOCK a bad model
make serve            # http://localhost:8000/docs
make simulate         # (2nd terminal) synthetic traffic + clinician reviews + outcomes
make monitor          # -> reports/monitoring_latest.{json,html}
make simulate-drift && make monitor   # watch the alerts fire
make mlflow-ui        # http://localhost:5000  experiments + registry
```
Docker: `PRP_REVIEWER_KEY=<secret> make docker`.

## Requirement → implementation
| Requirement | Where | Detail |
|---|---|---|
| Experiment tracking | `train.py` | MLflow (SQLite backend): params, CV/valid/test metrics, fairness metrics, thresholds, release artifacts, model registry w/ `champion` alias, `promotion=approved/blocked` tag |
| Model validation | `validation.py`, `train.py` | Data gate (schema, ranges, missingness, leakage); temporal split; CV; calibration (ECE/Brier); thresholds picked on validation only; champion-vs-challenger check; **all gates in `config.yaml`** |
| Workflow orchestration | `flows.py` | Prefect: `training_flow`, `monitoring_flow`, retries, cron deployments (`python -m prp.flows serve`) |
| Model serving | `serve.py` | FastAPI, typed/ranged input validation, explanations, `/health` `/model-info` `/metrics`, hot reload, one-step rollback (`make rollback`) |
| Performance monitoring | `monitoring.py` | PSI per feature + score, missing-lab drift, delayed-label AUC/sensitivity/ECE vs. baseline, subgroup performance |
| Bias/fairness | `fairness.py`, `MODEL_CARD.md` | Per-group TPR/FPR/PPV/AUC/calibration; demographic-parity / equal-opportunity / equalized-odds gaps; per-attribute gate policy; sensitive attrs excluded from features; oracle audit on synthetic truth |
| Human-in-the-loop | `triage.py`, `serve.py` | Model only routes; high/moderate → clinician queue; low-with-missing-labs escalated; 5% audit sample; overrides need rationale; override rate monitored; auth on reviewer endpoints |

## Example
```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{"age":67,"sex":"M","resting_bp":158,
 "cholesterol":290,"fasting_glucose":140,"max_heart_rate":118,"bmi":33,"st_depression":2.4,"smoker":1,
 "diabetes":1,"family_history":1,"exercise_angina":1,"chest_pain_type":"typical_angina"}'
# -> risk_score, risk_tier "high", recommended_action, top_factors[], requires_clinician_review:true
curl -H 'X-API-Key: dev-reviewer-key' localhost:8000/reviews/pending
```

## Before this touches real patients (not done here)
Real data + DUA/de-identification; prospective silent-mode trial; clinical safety case & regulatory review; replace the shared API key with
OIDC/RBAC; Postgres instead of SQLite; TLS, audit logging, PHI retention policy; alert routing (Slack/PagerDuty hooks are marked in `flows.py`);
intersectional fairness; scheduled retraining with human approval of the promotion.
