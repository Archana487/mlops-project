import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train a real (small) model into an isolated sandbox once per test session."""
    import os
    root = tmp_path_factory.mktemp("sandbox")
    os.environ.update(PRP_ARTIFACTS_DIR=str(root / "art"), PRP_DB=str(root / "serving.db"),
                      PRP_REPORTS_DIR=str(root / "rep"))
    from prp.config import load_config
    from prp.data import generate_dataset
    from prp.train import run_training
    cfg = load_config()
    df = generate_dataset(12000, seed=1)
    res = run_training(df, cfg)
    return res, df
