import joblib, pytest
from fastapi.testclient import TestClient

KEY = {"X-API-Key": "dev-reviewer-key"}
PATIENT = dict(age=67, sex="M", resting_bp=158, cholesterol=290, fasting_glucose=140, max_heart_rate=118,
               bmi=33, st_depression=2.4, smoker=1, diabetes=1, family_history=1, exercise_angina=1,
               chest_pain_type="typical_angina", race_ethnicity="Group C")


def test_training_meets_quality_gates_and_is_promoted(trained):
    res, _ = trained
    assert res["promoted"], res["gates_failed"]
    assert res["test_metrics"]["auc"] > 0.78 and res["test_metrics"]["ece"] < 0.05


def test_sensitive_attribute_not_a_model_feature(trained):
    from prp.config import champion_dir
    meta = joblib.load(champion_dir() / "bundle.joblib")["meta"]
    assert "race_ethnicity" not in meta["features"] and "heart_disease" not in meta["features"]


def test_failing_gate_blocks_promotion(trained, monkeypatch):
    from prp.config import champion_dir, load_config
    from prp.train import run_training
    cfg = load_config(); before = (champion_dir() / "meta.json").read_text()
    strict = {**cfg, "gates": {**cfg["gates"], "min_auc": 0.99}}
    _, df = trained
    res = run_training(df, strict)
    assert not res["promoted"] and any("AUC" in g for g in res["gates_failed"])
    assert (champion_dir() / "meta.json").read_text() == before     # champion untouched


@pytest.fixture()
def client(trained):
    from prp.serve import app
    with TestClient(app) as c:
        yield c


def test_predict_returns_decision_support_payload(client):
    r = client.post("/predict", json=PATIENT); assert r.status_code == 200
    j = r.json()
    assert 0 <= j["risk_score"] <= 1 and j["risk_tier"] in {"low", "moderate", "high"}
    assert "Decision-support" in j["disclaimer"] and j["top_factors"]
    assert j["risk_tier"] == "high" and j["requires_clinician_review"]


def test_predict_rejects_invalid_input(client):
    assert client.post("/predict", json={**PATIENT, "age": 250}).status_code == 422
    assert client.post("/predict", json={**PATIENT, "sex": "X"}).status_code == 422


def test_missing_lab_is_flagged_and_escalated(client):
    low = {**PATIENT, "age": 30, "sex": "F", "resting_bp": 110, "cholesterol": None, "bmi": 22, "st_depression": 0,
           "smoker": 0, "diabetes": 0, "family_history": 0, "exercise_angina": 0, "chest_pain_type": "non_anginal",
           "max_heart_rate": 185, "fasting_glucose": 90}
    j = client.post("/predict", json=low).json()
    assert "cholesterol" in j["data_quality_flags"] and j["requires_clinician_review"]


def test_review_workflow_requires_auth_and_rationale_for_override(client):
    pid = client.post("/predict", json=PATIENT).json()["prediction_id"]
    assert client.get("/reviews/pending").status_code == 401
    assert pid in [i["id"] for i in client.get("/reviews/pending", headers=KEY).json()["items"]]
    bad = client.post(f"/reviews/{pid}", headers=KEY, json={"reviewer": "dr", "decision": "override_lower"})
    assert bad.status_code == 422
    ok = client.post(f"/reviews/{pid}", headers=KEY, json={"reviewer": "dr", "decision": "override_lower",
                     "clinician_risk": "moderate", "notes": "Recent normal stress test."})
    assert ok.status_code == 200
    assert pid not in [i["id"] for i in client.get("/reviews/pending", headers=KEY).json()["items"]]


def test_monitoring_summary_and_metrics(client):
    for _ in range(3):
        client.post("/predict", json=PATIENT)
    s = client.get("/monitoring/summary", headers=KEY).json()
    assert "drift" in s and "human_in_the_loop" in s and "overall_status" in s
    assert "prp_predictions_total" in client.get("/metrics").text
