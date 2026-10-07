"""Prefect orchestration.

training_flow   : ingest -> validate -> train/evaluate/gate -> promote -> notify
monitoring_flow : load champion -> drift + performance + HITL health -> alert
Run locally:      python -m prp.flows train | monitor
Schedule (prod):  python -m prp.flows serve   (weekly retrain, daily monitoring)
"""
from __future__ import annotations
import sys
import joblib
import pandas as pd
from prefect import flow, get_run_logger, task

from .config import ROOT, champion_dir, load_config
from .data import generate_dataset
from .monitoring import run_monitoring
from .train import run_training
from .validation import validate_dataframe


@task(retries=2, retry_delay_seconds=5)
def ingest(path: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    p = ROOT / (path or cfg["data"]["path"])
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset(cfg["data"]["n_samples"], cfg["data"]["seed"]).to_csv(p, index=False)
        get_run_logger().info("Generated synthetic dataset at %s", p)
    return pd.read_csv(p, parse_dates=["admission_date"])


@task
def validate(df: pd.DataFrame) -> dict:
    rep = validate_dataframe(df, load_config())       # raises => flow fails, nothing trained
    get_run_logger().info("Data validation passed (%d rows)", rep["n_rows"])
    return rep


@task
def train_and_gate(df: pd.DataFrame) -> dict:
    res = run_training(df, load_config())
    log = get_run_logger()
    log.info("model=%s AUC=%.3f promoted=%s", res["model_type"], res["test_metrics"]["auc"], res["promoted"])
    for f in res["gates_failed"]:
        log.warning("GATE FAILED: %s", f)
    return {k: v for k, v in res.items() if k != "fairness"}


@flow(name="patient-risk-training", log_prints=True)
def training_flow(data_path: str | None = None) -> dict:
    df = ingest(data_path)
    validate(df)
    res = train_and_gate(df)
    if not res["promoted"]:
        print("NOT PROMOTED -> champion unchanged. Reasons:", res["gates_failed"])   # hook: Slack/PagerDuty
    return res


@task
def monitor(bundle_path: str) -> dict:
    return run_monitoring(joblib.load(bundle_path), load_config())


@flow(name="patient-risk-monitoring", log_prints=True)
def monitoring_flow() -> dict:
    rep = monitor(str(champion_dir() / "bundle.joblib"))
    print(f"status={rep['overall_status']} :: {rep['recommendation']}")
    if rep["overall_status"] == "alert":
        print("ALERT -> page on-call ML engineer + clinical safety officer")   # hook
    return rep


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        training_flow()
    elif cmd == "monitor":
        monitoring_flow()
    elif cmd == "serve":
        from prefect import serve
        serve(training_flow.to_deployment("weekly-retrain", cron="0 3 * * 0"),
              monitoring_flow.to_deployment("daily-monitoring", cron="0 6 * * *"))
