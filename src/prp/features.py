from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    f = cfg["features"]
    # add_indicator=True keeps "lab not measured" as a signal (it is informative, but is also
    # a potential proxy for access to care -- flagged in MODEL_CARD.md and audited by fairness gate).
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", drop="if_binary")
    return ColumnTransformer([
        ("num", num, f["numeric"]),
        ("bin", "passthrough", f["binary"]),
        ("cat", cat, f["categorical"]),
    ])
