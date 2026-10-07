"""Human-in-the-loop policy. The model NEVER takes a clinical action on its own: it only
prioritises and routes work to clinicians. Every route below is a recommendation to a human."""
from __future__ import annotations
import hashlib

TIERS = {
    "high": {"queue": "priority", "action": "Urgent clinician review recommended",
             "requires_review": True, "priority": 1},
    "moderate": {"queue": "standard", "action": "Clinician review recommended",
                 "requires_review": True, "priority": 2},
    "low": {"queue": None, "action": "Routine care pathway (no model-triggered action)",
            "requires_review": False, "priority": 3},
}


def assign_tier(score: float, thresholds: dict) -> str:
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["low"]:
        return "moderate"
    return "low"


def _audit_hash(prediction_id: str) -> float:
    return int(hashlib.sha256(prediction_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def route(prediction_id: str, score: float, thresholds: dict, audit_rate: float,
          data_quality_flags: list[str] | None = None) -> dict:
    """Returns tier, queue and why. Deterministic audit sampling => reproducible/auditable."""
    tier = assign_tier(score, thresholds)
    info = dict(TIERS[tier], tier=tier, reasons=[])
    flags = data_quality_flags or []
    if tier == "low":
        if flags:   # low score based on missing key labs is not trustworthy -> human looks
            info.update(requires_review=True, queue="standard", priority=2,
                        action="Clinician review recommended (score based on incomplete data)")
            info["reasons"].append("incomplete_data: " + ",".join(flags))
        elif _audit_hash(prediction_id) < audit_rate:
            info.update(requires_review=True, queue="audit", priority=4,
                        action="Randomly sampled low-risk case for quality audit")
            info["reasons"].append("random_audit_sample")
    else:
        info["reasons"].append(f"score_{tier}")
    return info
