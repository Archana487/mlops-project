"""Simulate production: send patients to the API, have 'clinicians' review the queue,
then feed back real outcomes. Use --drift to inject covariate + concept drift.

  python scripts/simulate_traffic.py --n 800 [--drift]
"""
import argparse, math, sys
from pathlib import Path
import httpx, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prp.data import generate_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--drift", action="store_true")
ap.add_argument("--key", default="dev-reviewer-key")
a = ap.parse_args()

df = generate_dataset(a.n, seed=777 if a.drift else 123, drift=a.drift)
H, rng = {"X-API-Key": a.key}, np.random.default_rng(0)
cli = httpx.Client(base_url=a.url, timeout=30)
feat = ["age","sex","resting_bp","cholesterol","fasting_glucose","max_heart_rate","bmi","st_depression",
        "smoker","diabetes","family_history","exercise_angina","chest_pain_type","race_ethnicity"]
ids = []
for _, r in df.iterrows():
    body = {k: (None if isinstance(r[k], float) and math.isnan(r[k]) else (r[k].item() if hasattr(r[k], "item") else r[k]))
            for k in feat}
    body["external_id"] = r["patient_id"]
    resp = cli.post("/predict", json=body); resp.raise_for_status()
    ids.append((resp.json(), int(r["heart_disease"])))
print(f"scored {len(ids)} patients")

# clinicians work the queue: usually confirm; sometimes override when the model is clearly off
n_rev = 0
for pr, label in ids:
    if not pr["requires_clinician_review"]:
        continue
    if rng.random() < 0.9:                       # 10% of cases not yet reviewed (backlog)
        wrong_high = pr["risk_tier"] == "high" and label == 0 and rng.random() < 0.15
        wrong_low = pr["risk_tier"] != "high" and label == 1 and rng.random() < 0.25
        if wrong_high or wrong_low:
            body = {"reviewer": "dr_demo", "decision": "override_lower" if wrong_high else "override_higher",
                    "clinician_risk": "low" if wrong_high else "high",
                    "notes": "Clinical picture inconsistent with model score."}
        else:
            body = {"reviewer": "dr_demo", "decision": "confirm"}
        cli.post(f"/reviews/{pr['prediction_id']}", json=body, headers=H).raise_for_status(); n_rev += 1
for pr, label in ids:                             # outcomes arrive later (e.g. confirmed diagnosis)
    cli.post(f"/outcomes/{pr['prediction_id']}", json={"label": label}, headers=H).raise_for_status()
print(f"reviews submitted: {n_rev}; outcomes recorded: {len(ids)}")
