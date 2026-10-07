import numpy as np, pandas as pd, pytest
from prp.config import load_config
from prp.data import generate_dataset
from prp.fairness import expected_calibration_error, fairness_gate, group_metrics
from prp.monitoring import psi
from prp.triage import assign_tier, route
from prp.validation import DataValidationError, validate_dataframe

CFG = load_config()


def test_validation_passes_on_good_data():
    assert validate_dataframe(generate_dataset(3000, 0), CFG)["passed"]


def test_validation_rejects_out_of_range_and_missing_cols():
    df = generate_dataset(3000, 0)
    bad = df.copy(); bad.loc[:50, "age"] = 400
    with pytest.raises(DataValidationError):
        validate_dataframe(bad, CFG)
    with pytest.raises(DataValidationError):
        validate_dataframe(df.drop(columns=["bmi"]), CFG)


def test_validation_detects_leakage():
    df = generate_dataset(3000, 0); df["st_depression"] = df["heart_disease"] * 1.0
    with pytest.raises(DataValidationError, match="leakage"):
        validate_dataframe(df, CFG)


def test_group_metrics_known_values():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0]); p = np.array([.9, .8, .1, .2, .6, .3, .4, .1])
    g = np.array(["a"] * 4 + ["b"] * 4)
    gm = group_metrics(y, p, 0.5, g).set_index("group")
    assert gm.loc["a", "tpr"] == 1.0 and gm.loc["b", "tpr"] == 0.5
    ok, fails, d = fairness_gate(gm.reset_index(), {**CFG["gates"], "min_group_size": 1})
    assert not ok and d["equal_opportunity_gap"] == pytest.approx(0.5)


def test_calibration_error_perfect_and_bad():
    rng = np.random.default_rng(0); p = rng.random(20000); y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.15


def test_psi_detects_shift():
    rng = np.random.default_rng(0); a = pd.Series(rng.normal(0, 1, 5000))
    assert psi(a, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi(a, pd.Series(rng.normal(1.5, 1, 5000))) > 0.25


def test_triage_tiers_and_incomplete_data_escalation():
    thr = {"high": 0.4, "low": 0.1}
    assert assign_tier(0.5, thr) == "high" and assign_tier(0.2, thr) == "moderate" and assign_tier(0.05, thr) == "low"
    r = route("abc", 0.05, thr, 0.0, ["cholesterol"])        # low score but missing lab => human looks
    assert r["requires_review"] and r["queue"] == "standard"
    assert not route("abc", 0.05, thr, 0.0)["requires_review"]


def test_audit_sampling_rate_close_to_target():
    thr = {"high": 0.4, "low": 0.1}
    n = sum(route(f"id{i}", 0.01, thr, 0.05)["requires_review"] for i in range(5000))
    assert 0.03 < n / 5000 < 0.07
