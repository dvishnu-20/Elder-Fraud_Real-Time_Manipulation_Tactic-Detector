# V2 Phase 5: Temporal Feature Extraction & Statistical Audit Report

**Feature Extraction Status:** **SUCCESSFULLY EXTRACTED & AUDITED**  
**Frozen Classifier:** `ml/models/tactic_classifier/v2_temporal_validity/`  
**Extracted Feature Directory:** `data/processed/temporal_v2_features/`  
**Total Conversations Processed:** 2,000 (Train: 1,400, Val: 300, Test: 300)  
**Total Turns Processed:** 16,312 turns  
**Feature Dimension:** Exactly 5 continuous DistilBERT probabilities ($x_t \in [0, 1]^5$)  
**Extraction Seed:** 42 (Reproducible)  
**Current Gate Status:** **V2 TEMPORAL FEATURE EXTRACTION COMPLETE — AWAITING PHASE 5 MODELING APPROVAL**

---

## 1. Split Integrity & Provenance Verification

| Split | Conversation Count | Fraud Dialogues | Benign Dialogues | Total Turn Count | File Output Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 1,400 (70.0%) | 707 (50.5%) | 693 (49.5%) | 11,454 turns | `data/processed/temporal_v2_features/train_features.jsonl` |
| **Validation** | 300 (15.0%) | 160 (53.3%) | 140 (46.7%) | 2,458 turns | `data/processed/temporal_v2_features/validation_features.jsonl` |
| **Test** | 300 (15.0%) | 133 (44.3%) | 167 (55.7%) | 2,400 turns | `data/processed/temporal_v2_features/test_features.jsonl` |
| **Total** | **2,000** | **1,000** | **1,000** | **16,312 turns** | `data/processed/temporal_v2_features/` |

- **Conversation ID Overlap:**
  - Train $\cap$ Validation: **0**
  - Train $\cap$ Test: **0**
  - Validation $\cap$ Test: **0**
- **Legacy V1 Isolation:** All V1 models (`ml/models/tactic_classifier/retrained_v2/`, `ml/models/temporal_risk/lstm_v1/`) and sequence directories (`data/processed/temporal/`, `data/processed/temporal_v1/`) remain strictly frozen and untouched.

---

## 2. Sequence Quality & Dimension Audit

| Quality Metric | Train Split | Validation Split | Test Split | Aggregate Dataset |
| :--- | :---: | :---: | :---: | :---: |
| **Total Sequences ($N$)** | 1,400 | 300 | 300 | **2,000** |
| **Minimum Sequence Length** | 4 turns | 4 turns | 4 turns | **4 turns** |
| **Maximum Sequence Length** | 12 turns | 12 turns | 12 turns | **12 turns** |
| **Mean Sequence Length** | 8.18 turns | 8.19 turns | 8.00 turns | **8.16 turns** |
| **Median Sequence Length** | 8.0 turns | 8.0 turns | 8.0 turns | **8.0 turns** |
| **% Sequences $\ge 6$ Turns** | **84.43%** | **87.67%** | **83.33%** | **84.75%** |
| **% Sequences $\ge 10$ Turns** | 45.93% | 46.00% | 39.67% | **45.00%** |
| **Empty / Malformed Sequences** | 0 | 0 | 0 | **0** |
| **NaN / Null Probabilities** | 0 | 0 | 0 | **0** |
| **Probabilities Outside $[0, 1]$** | 0 | 0 | 0 | **0** |
| **Missing Turns** | 0 | 0 | 0 | **0** |

---

## 3. Continuous 5-Tactic Probability Distribution Audit

For every timestep, the frozen V2 DistilBERT classifier outputs 5 continuous sigmoid probabilities:
$$x_t = \begin{bmatrix} p_{\text{urgency}}, & p_{\text{isolation}}, & p_{\text{authority}}, & p_{\text{emotional}}, & p_{\text{channel}} \end{bmatrix}$$

### Summary Statistics by Split

#### Train Split ($N = 11,454$ turns)
| Tactic Dimension | Mean | Std Dev | Min | Median | 95th Percentile | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Urgency** | 0.0830 | 0.2708 | 0.0001 | 0.0007 | 0.9988 | 0.9989 |
| **Isolation** | 0.0767 | 0.2577 | 0.0001 | 0.0007 | 0.9839 | 0.9971 |
| **Authority** | 0.0605 | 0.2328 | 0.0002 | 0.0009 | 0.9984 | 0.9993 |
| **Emotional Destabilization** | 0.0261 | 0.1462 | 0.0003 | 0.0014 | 0.0205 | 0.9987 |
| **Channel Control** | 0.0610 | 0.2340 | 0.0001 | 0.0007 | 0.9985 | 0.9989 |

#### Validation Split ($N = 2,458$ turns)
| Tactic Dimension | Mean | Std Dev | Min | Median | 95th Percentile | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Urgency** | 0.0839 | 0.2723 | 0.0001 | 0.0007 | 0.9988 | 0.9989 |
| **Isolation** | 0.0793 | 0.2619 | 0.0001 | 0.0007 | 0.9839 | 0.9971 |
| **Authority** | 0.0619 | 0.2355 | 0.0002 | 0.0009 | 0.9984 | 0.9993 |
| **Emotional Destabilization** | 0.0244 | 0.1405 | 0.0003 | 0.0012 | 0.0205 | 0.9987 |
| **Channel Control** | 0.0623 | 0.2362 | 0.0001 | 0.0007 | 0.9986 | 0.9989 |

#### Test Split ($N = 2,400$ turns)
| Tactic Dimension | Mean | Std Dev | Min | Median | 95th Percentile | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Urgency** | 0.0827 | 0.2701 | 0.0001 | 0.0007 | 0.9988 | 0.9989 |
| **Isolation** | 0.0711 | 0.2482 | 0.0001 | 0.0007 | 0.9839 | 0.9971 |
| **Authority** | 0.0649 | 0.2412 | 0.0002 | 0.0010 | 0.9984 | 0.9993 |
| **Emotional Destabilization** | 0.0255 | 0.1449 | 0.0003 | 0.0012 | 0.0205 | 0.9987 |
| **Channel Control** | 0.0641 | 0.2396 | 0.0001 | 0.0007 | 0.9985 | 0.9989 |

*Finding:* The feature distributions across Train, Validation, and Test splits are highly consistent, ensuring zero distribution shift.

---

## 4. Temporal Dynamics & Pacing Audit

| Dynamic Metric | Train (Fraud / Benign) | Validation (Fraud / Benign) | Test (Fraud / Benign) |
| :--- | :---: | :---: | :---: |
| **Mean Tactic-Bearing Turns / Dialogue** | 3.70 / 1.07 | 3.65 / 1.06 | 3.80 / 1.20 |
| **Mean Turn of First Tactic (Fraud)** | 2.64 (Turns 1–7) | 2.55 (Turns 1–7) | 2.47 (Turns 1–7) |
| **Fraud with $\ge 2$ Normal Turns Before First Tactic** | **35.8%** (253 / 707) | **32.5%** (52 / 160) | **33.8%** (45 / 133) |
| **Fraud with Multi-Turn Normal Gaps ($\text{gap} \ge 2$)** | **30.6%** (216 / 707) | **27.5%** (44 / 160) | **30.1%** (40 / 133) |
| **Benign with $\ge 1$ Tactic Activation** | **64.5%** (447 / 693) | **61.4%** (86 / 140) | **70.1%** (117 / 167) |
| **Benign with $\ge 2$ Tactic Activations** | **21.2%** (147 / 693) | **22.1%** (31 / 140) | **25.1%** (42 / 167) |

---

## 5. Descriptive Baseline & Shortcut Resistance Check

Descriptive evaluation on the Test Split ($N = 300$ conversations, 133 Fraud, 167 Benign):

| Simple Tactic Metric | Fraud Group Mean | Benign Group Mean | Descriptive Predictive Power | Shortcut Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **A. Any Tactic Detected ($\ge 1$)** | 100.0% (133/133) | **70.1% (117/167)** | $P(\text{Fraud} \mid \text{Tactic}) = \mathbf{0.5320}$ | **Shortcut ELIMINATED** (triggers 70.1% false alarms on benign) |
| **B. Total Tactic Activations** | 3.80 turns | 1.20 turns | Significant overlap in count distributions | Count threshold alone insufficient |
| **C. Max Tactic Probability** | 0.9990 | 0.7012 | High max probabilities present in benign | Max-pooling alone insufficient |
| **D. Mean Tactic Probability** | 0.0780 | 0.0406 | Overlapping mean density | Mean-pooling alone insufficient |

**Conclusion:** The synthetic shortcut has been eliminated. Tactic presence alone only achieves $53.2\%$ precision. Detecting fraud on Dataset V2 requires modeling **temporal escalation, multi-turn persistence, and conversational context**.

---

## 6. Manifest File Created

Saved to: [`data/processed/temporal_v2_features/feature_manifest.json`](file:///c:/Users/duddu/Downloads/ELDERY%20LIVE%20TATIC%20DETECTOR/elder-fraud-detector/data/processed/temporal_v2_features/feature_manifest.json)

---

## 7. Final Status & Mandatory Stop Condition Observed

```
================================================================================
FINAL STATUS:
V2 TEMPORAL FEATURE EXTRACTION COMPLETE — AWAITING PHASE 5 MODELING APPROVAL
================================================================================
```
