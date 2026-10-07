"""SQLite store for predictions, clinician reviews and observed outcomes.
(Swap for Postgres in production; SQL is deliberately vanilla.)"""
from __future__ import annotations
import json, sqlite3, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
import pandas as pd
from .config import db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions(
  id TEXT PRIMARY KEY, ts TEXT, model_version TEXT, external_id TEXT,
  features TEXT, score REAL, tier TEXT, queue TEXT, requires_review INTEGER,
  priority INTEGER, reasons TEXT, race_ethnicity TEXT);
CREATE TABLE IF NOT EXISTS reviews(
  prediction_id TEXT PRIMARY KEY, ts TEXT, reviewer TEXT, decision TEXT,
  clinician_risk TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS outcomes(
  prediction_id TEXT PRIMARY KEY, ts TEXT, label INTEGER);
CREATE INDEX IF NOT EXISTS ix_pred_queue ON predictions(requires_review, queue, priority, ts);
"""
DECISIONS = {"confirm", "override_higher", "override_lower", "defer"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn():
    c = sqlite3.connect(db_path(), timeout=30)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def save_prediction(features: dict, score: float, route: dict, model_version: str,
                    external_id: str | None, race_ethnicity: str | None, pid: str | None = None) -> str:
    pid = pid or uuid.uuid4().hex
    with conn() as c:
        c.execute("INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                  (pid, now(), model_version, external_id, json.dumps(features), score,
                   route["tier"], route["queue"], int(route["requires_review"]), route["priority"],
                   json.dumps(route["reasons"]), race_ethnicity))
    return pid


def pending_reviews(limit: int = 50) -> list[dict]:
    with conn() as c:
        rows = c.execute("""SELECT p.* FROM predictions p LEFT JOIN reviews r ON r.prediction_id=p.id
                            WHERE p.requires_review=1 AND r.prediction_id IS NULL
                            ORDER BY p.priority, p.ts LIMIT ?""", (limit,)).fetchall()
    return [dict(r) | {"features": json.loads(r["features"]), "reasons": json.loads(r["reasons"])} for r in rows]


def get_prediction(pid: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone()
    return None if r is None else dict(r) | {"features": json.loads(r["features"])}


def save_review(pid: str, reviewer: str, decision: str, clinician_risk: str | None, notes: str | None) -> None:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO reviews VALUES(?,?,?,?,?,?)",
                  (pid, now(), reviewer, decision, clinician_risk, notes))


def save_outcome(pid: str, label: int) -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO outcomes VALUES(?,?,?)", (pid, now(), int(label)))


def predictions_df(since: str | None = None) -> pd.DataFrame:
    with conn() as c:
        q = "SELECT * FROM predictions" + (" WHERE ts>=?" if since else "")
        df = pd.read_sql_query(q, c, params=(since,) if since else ())
    if df.empty:
        return df
    feats = pd.json_normalize(df["features"].map(json.loads))
    return pd.concat([df.drop(columns="features").reset_index(drop=True), feats], axis=1)


def labeled_df() -> pd.DataFrame:
    df = predictions_df()
    if df.empty:
        return df
    with conn() as c:
        o = pd.read_sql_query("SELECT prediction_id AS id, label FROM outcomes", c)
    return df.merge(o, on="id")


def reviews_df() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query("""SELECT r.*, p.tier, p.score, p.ts AS pred_ts FROM reviews r
                                    JOIN predictions p ON p.id=r.prediction_id""", c)
