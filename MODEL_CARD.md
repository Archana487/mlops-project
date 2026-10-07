# Model Card — Heart-Disease Risk Triage (v1)

> **Status: demonstration on SYNTHETIC data. Not validated for clinical use.** All numbers below describe
> the synthetic benchmark. Before any real deployment: retrain on real, governed data, run a prospective
> silent-mode evaluation, and obtain clinical-safety + regulatory sign-off (e.g. SaMD classification, DCB0129/0160, FDA CDS guidance).

## Intended use
Prioritise adult patients for **clinician review** (cardiac work-up queue). Decision *support* only:
the system never orders tests, changes treatment, or closes a case without a human.
**Out of scope:** diagnosis, patients < 18, pregnancy, acute chest pain (use emergency pathway), any fully automated decision.

## Data & split
20,000 synthetic admissions (2023–24); ~19% recorded prevalence. **Temporal** split 70/15/15 (train/valid/test) — the test set is the most
recent period and untouched until final evaluation. Thresholds are chosen on *validation*, never on test.
Race/ethnicity, and age-band are used **only for auditing**; the model uses 13 clinical features (see `config.yaml`).

## Model
Calibrated (sigmoid, 3-fold) pipeline; candidates compared by CV + validation AUC:
logistic regression (selected, valid AUC 0.843) vs. gradient boosting (0.838). Simpler model won, so it ships (easier to explain and audit).

## Performance (held-out test, n=3,000)
| AUC | PR-AUC | Brier | ECE | Sensitivity @high | Specificity | PPV | NPV |
|---|---|---|---|---|---|---|---|
| 0.844 | 0.600 | 0.111 | 0.015 | 0.841 | 0.675 | 0.382 | 0.947 |

Operating points (calibrated probability): **high ≥ 0.144** (≈85% sensitivity on validation), **moderate 0.060–0.144**, **low < 0.060** (≈95% sensitivity retained).
Low PPV (0.38) is inherent to the prevalence; the model is a *prioritiser*, not a rule-in test.

## Fairness audit (race_ethnicity; test set)
| Group | n | TPR | FPR | AUC | Calibration gap (pred − observed) |
|---|---|---|---|---|---|
| A | 1649 | 0.844 | 0.321 | 0.850 | −0.003 |
| B | 635 | 0.806 | 0.315 | 0.834 | −0.011 |
| C | 433 | 0.836 | 0.344 | 0.819 | +0.030 |
| D | 283 | 0.903 | 0.344 | 0.875 | −0.009 |

Equal-opportunity gap 0.097 (gate ≤ 0.12) · min group AUC 0.819 · max calibration gap 0.030 (gate ≤ 0.05) → **passes**.

**Findings a reader should not miss**
1. **Label bias, not model bias, drives Group C's gap.** The EHR under-records disease in Group C (recorded prevalence 16.9% vs. true 20.8% in the synthetic oracle).
   Against *recorded* labels Group C looks over-predicted (+0.030); against ground truth it is well-calibrated (−0.010). Auditing against EHR labels alone can mislead in either direction. Real deployments must seek an independent gold standard for at least a sample.
2. **Sex:** women are flagged less often (35% vs 50%) with lower TPR (0.80 vs 0.87). Gap passes the gate (0.067) but is clinically important — heart disease is historically under-detected in women. Needs clinical review of whether the operating point should differ.
3. **Missing-lab indicator is a proxy** for access to care (Group C has 25% missing cholesterol vs. 5%). We keep the signal but (a) force human review of low scores built on missing labs and (b) audit subgroup performance every release.
4. Groups B and D are small (n=635 / 283); TPR estimates have wide confidence intervals (±~4–6 pts). A gap near the gate is not distinguishable from noise — collect more data before tightening.
5. **Age** legitimately drives risk, so TPR/flag-rate parity across age bands is *not* gated; calibration is (policy `calibration_only`).

## Human-in-the-loop
High & moderate → clinician queues (priority-ordered). Low + missing key lab → standard queue. 5% of other low-risk cases are deterministically sampled for audit.
Overrides require a clinical rationale; override rate > 30% triggers an alert. Overrides and outcomes feed the next retrain/evaluation.

## Known limitations
Synthetic data (real-world shift will be larger) · explanations are occlusion-based associations, not causal · single-site assumptions · no intersectional (e.g. race × sex) gating yet · outcomes arrive delayed so performance monitoring lags (drift monitoring is the leading indicator).
