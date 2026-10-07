"""Config + path resolution. Paths can be overridden by env vars (used by tests/containers)."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = Path(os.environ.get("PRP_CONFIG", ROOT / "config.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


def artifacts_dir() -> Path:
    p = Path(os.environ.get("PRP_ARTIFACTS_DIR", ROOT / "artifacts"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def champion_dir() -> Path:
    return artifacts_dir() / "champion"


def db_path() -> Path:
    return Path(os.environ.get("PRP_DB", artifacts_dir() / "serving.db"))


def reports_dir() -> Path:
    p = Path(os.environ.get("PRP_REPORTS_DIR", ROOT / "reports"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{artifacts_dir() / 'mlflow.db'}")


def feature_columns(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    f = cfg["features"]
    return f["numeric"] + f["binary"] + f["categorical"]
