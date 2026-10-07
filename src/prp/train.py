"""Training + evaluation + gated promotion, with full MLflow tracking."""
from __future__ import annotations
import json, tempfile, time
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import __version__
from .config import champion_dir, feature_columns, load_config, mlflow_uri, artifacts_dir
from .data import add_age_band, temporal_split
from .fairness import disparities, expected_calibration_error, fairness_gate, group_metrics
from .features import build_preprocessor
from .validation import validate_dataframe


def candidates(cfg: dict) -> dict:
    def wrap(clf):
        return Pipeline([("prep", build_preprocessor(cfg)), ("clf", clf)])
    return {
        "logistic_regression": wrap(LogisticRegression(C=0.5, max_iter=2000)),
        "hist_gradient_boosting": wrap(HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0, random_state=0)),
    }


def threshold_for_sensitivity(y, p, target: float) -> float:
    """Highest threshold that still achieves >= target sensitivity."""
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k])


def operating_metrics(y, p, thr) -> dict:
    y, pred = np.asarray(y), np.asarray(p) >= thr
    tp, fn = (pred & (y == 1)).sum(), (~pred & (y == 1)).sum()
    tn, fp = (~pred & (y == 0)).sum(), (pred & (y == 0)).sum()
    return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1), "flag_rate": pred.mean()}


def evaluate(model, df: pd.DataFrame, cfg: dict, thr: float) -> dict:
    cols, y = feature_columns(cfg), df[cfg["target"]].to_numpy()
    p = model.predict_proba(df[cols])[:, 1]
    m = {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
         "brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p),
         **operating_metrics(y, p, thr)}
    return {k: float(v) for k, v in m.items()}, p


def run_training(df: pd.DataFrame, cfg: dict | None = None, register: bool = True) -> dict:
    cfg = cfg or load_config()
    validate_dataframe(df, cfg)
    df = add_age_band(df)
    cols, target = feature_columns(cfg), cfg["target"]
    tr, va, te = temporal_split(df, cfg["split"]["train"], cfg["split"]["valid"])
    tcfg, gates = cfg["training"], cfg["gates"]

    mlflow.set_tracking_uri(mlflow_uri())
    exp = cfg["project"]["experiment_name"]
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=(artifacts_dir() / "mlartifacts").as_uri())
    mlflow.set_experiment(exp)

    with mlflow.start_run(run_name=f"pipeline-{time.strftime('%Y%m%d-%H%M%S')}") as parent:
        mlflow.log_params({"n_train": len(tr), "n_valid": len(va), "n_test": len(te),
                           "split": "temporal", "code_version": __version__,
                           "train_window": f"{tr.admission_date.min().date()}..{tr.admission_date.max().date()}",
                           "test_window": f"{te.admission_date.min().date()}..{te.admission_date.max().date()}"})
        # ---- 1. compare candidates (CV on train, then valid) -------------------------
        scores, fitted = {}, {}
        for name, pipe in candidates(cfg).items():
            with mlflow.start_run(run_name=name, nested=True):
                cv = StratifiedKFold(tcfg["cv_folds"], shuffle=True, random_state=0)
                cv_auc = cross_val_score(pipe, tr[cols], tr[target], cv=cv, scoring="roc_auc")
                model = CalibratedClassifierCV(pipe, method="sigmoid", cv=3).fit(tr[cols], tr[target])
                pv = model.predict_proba(va[cols])[:, 1]
                v_auc = roc_auc_score(va[target], pv)
                mlflow.log_params({"model": name, **{k: v for k, v in pipe.named_steps["clf"].get_params().items()
                                                     if isinstance(v, (int, float, str))}})
                mlflow.log_metrics({"cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
                                    "valid_auc": v_auc, "valid_brier": brier_score_loss(va[target], pv),
                                    "valid_ece": expected_calibration_error(va[target], pv)})
                scores[name], fitted[name] = v_auc, model
        best = max(scores, key=scores.get)
        model = fitted[best]
        mlflow.log_param("selected_model", best)

        # ---- 2. clinical operating points (chosen on VALID, never on test) -----------
        pv = model.predict_proba(va[cols])[:, 1]
        thr_high = threshold_for_sensitivity(va[target], pv, tcfg["target_sensitivity"])
        thr_low = threshold_for_sensitivity(va[target], pv, tcfg["safe_sensitivity"])
        mlflow.log_params({"threshold_high": round(thr_high, 4), "threshold_low": round(thr_low, 4)})

        # ---- 3. untouched test set: overall + subgroup audit -------------------------
        test_metrics, pt = evaluate(model, te, cfg, thr_high)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        fairness, gates_failed = {}, []
        primary = cfg["primary_audit_attribute"]
        for attr in cfg["audit_attributes"]:
            gm = group_metrics(te[target], pt, thr_high, te[attr])
            ok, fails, d = fairness_gate(gm, gates, cfg["fairness_gate_policy"].get(attr, "full"))
            fairness[attr] = {"table": gm.round(4).to_dict("records"), "summary": d, "passed": ok, "failures": fails}
            mlflow.log_metrics({f"fair_{attr}_{k}": v for k, v in d.items() if isinstance(v, float)})
            if True:                                # every audited attribute gates under its policy
                gates_failed += [f"[{attr}] {x}" for x in fails]
        # oracle audit (synthetic data only): how does the model do against ground truth?
        if "true_disease" in te:
            gm = group_metrics(te["true_disease"], pt, thr_high, te[primary])
            fairness["oracle_" + primary] = {"table": gm.round(4).to_dict("records"),
                                             "summary": disparities(gm)}
        if test_metrics["auc"] < gates["min_auc"]:
            gates_failed.append(f"AUC {test_metrics['auc']:.3f} < {gates['min_auc']}")
        if test_metrics["ece"] > gates["max_ece"]:
            gates_failed.append(f"ECE {test_metrics['ece']:.3f} > {gates['max_ece']}")

        # ---- 4. champion/challenger -------------------------------------------------------
        champ_auc = None
        bundle_path = champion_dir() / "bundle.joblib"
        if bundle_path.exists():
            champ = joblib.load(bundle_path)["model"]
            champ_auc, _ = evaluate(champ, te, cfg, thr_high)
            champ_auc = champ_auc["auc"]
            if test_metrics["auc"] < champ_auc - gates["max_champion_auc_regression"]:
                gates_failed.append(f"challenger AUC {test_metrics['auc']:.3f} regresses vs champion {champ_auc:.3f}")

        # ---- 5. package ---------------------------------------------------------------------
        X = tr[cols]
        baseline = {c: (X[c].median() if c in cfg["features"]["numeric"] else X[c].mode().iloc[0])
                    for c in cols}
        baseline = {k: (v.item() if hasattr(v, "item") else v) for k, v in baseline.items()}
        ref = tr.sample(min(3000, len(tr)), random_state=0)
        ref = ref[cols + ["race_ethnicity"]].assign(score=model.predict_proba(ref[cols])[:, 1])
        version = f"{time.strftime('%Y%m%d%H%M')}-{parent.info.run_id[:6]}"
        meta = {"version": version, "model_type": best, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mlflow_run_id": parent.info.run_id, "features": cols, "baseline": baseline,
                "thresholds": {"high": thr_high, "low": thr_low}, "test_metrics": test_metrics,
                "valid_auc_by_model": scores, "gates_failed": gates_failed, "promoted": False,
                "fairness_summary": {k: v.get("summary") for k, v in fairness.items()}}
        outdir = Path(tempfile.mkdtemp())
        joblib.dump({"model": model, "meta": meta, "reference": ref}, outdir / "bundle.joblib")
        (outdir / "fairness_report.json").write_text(json.dumps(fairness, indent=2, default=float))
        (outdir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
        mlflow.log_artifacts(str(outdir), artifact_path="release_candidate")
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        mlflow.log_metric("gates_passed", int(not gates_failed))
        mlflow.set_tag("promotion", "approved" if not gates_failed else "blocked")

        result = {"version": version, "model_type": best, "test_metrics": test_metrics,
                  "gates_failed": gates_failed, "champion_auc": champ_auc,
                  "run_id": parent.info.run_id, "bundle_dir": str(outdir), "fairness": fairness}
        # ---- 6. gated promotion ---------------------------------------------------------
        if not gates_failed and register:
            promote(outdir, meta, parent.info.run_id, cfg)
            result["promoted"] = True
        else:
            result["promoted"] = False
        return result


def promote(bundle_dir: Path, meta: dict, run_id: str, cfg: dict) -> None:
    """Atomic swap of the serving bundle + MLflow registry alias 'champion'."""
    import shutil
    dest = champion_dir()
    tmp = dest.with_name("champion.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(bundle_dir, tmp)
    meta = {**meta, "promoted": True}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    b = joblib.load(tmp / "bundle.joblib"); b["meta"] = meta; joblib.dump(b, tmp / "bundle.joblib")
    if dest.exists():
        shutil.rmtree(dest.with_name("previous"), ignore_errors=True)
        dest.rename(dest.with_name("previous"))     # keep one-step rollback
    tmp.rename(dest)
    client = MlflowClient(mlflow_uri())
    name = cfg["project"]["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.set_registered_model_alias(name, "champion", mv.version)
    client.set_model_version_tag(name, mv.version, "serving_version", meta["version"])


def rollback() -> str:
    import shutil
    prev, cur = champion_dir().with_name("previous"), champion_dir()
    if not prev.exists():
        raise RuntimeError("No previous champion to roll back to")
    shutil.rmtree(cur, ignore_errors=True)
    prev.rename(cur)
    return json.loads((cur / "meta.json").read_text())["version"]
