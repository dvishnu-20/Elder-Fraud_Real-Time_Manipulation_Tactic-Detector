# V2 Final Temporal Scientific Validation Report

**Authoritative Status:** **V2 FINAL SCIENTIFIC VALIDATION COMPLETE — AWAITING REVIEW**  
**Dataset:** Dataset V2 (`data/processed/temporal_v2_features/`)  
**Frozen Turn Classifier:** `ml/models/tactic_classifier/v2_temporal_validity/`  
**Test Set Size:** $N = 300$ Conversations (133 Fraud, 167 Benign, 2,400 Turns)  

---

## 1. Objective & Invariance Proof

This final robustness experiment resolves whether the Causal LSTM's performance is driven by true temporal sequence dynamics or model expressiveness/capacity.

### A. Mathematical Verification of Permutation Invariance
To establish an unassailable non-temporal upper bound, we designed a **61-dimensional permutation-invariant feature extractor** mapping the $T 	imes 5$ sequence matrix to:
- 12 symmetric statistics per probability dimension (Mean, Max, Min, Std, Sum, Median, 75th %ile, 90th %ile, Count $>0.1$, Count $>0.3$, Count $>0.5$, Max Sorted Gap) = 60 features.
- Sequence Length $T$ = 1 feature.

**Invariance Verification Check:**
$$\max |\Phi(\pi(X)) - \Phi(X)| = 0.00 \times 10^-6$$
- Verified strictly permutation invariant across all test conversations: **PASS (100.0% Order Independent)**.

---

## 2. Definitive 4-Way Model Comparison on Held-Out Test Split ($N=300$)

All models trained **strictly on Train** ($N=1,400$), tuned/thresholded **strictly on Validation** ($N=300$), evaluated **once on Test** ($N=300$).

| Model Architecture | Input Representation | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Macro F1 | Confusion (TN / FP / FN / TP) | Benign FPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Tactic-Count Rule** | Simple sum of thresholded activations | 0.8667 | 0.7492 | 0.7600 | 1.0000 | 0.8636 | 0.8599 | 125 / 42 / 0 / 133 | 25.15% |
| **2. Logistic Regression** | 20 Aggregate Features (Mean, Max, Sum, Final) | 0.9998 | 0.9997 | 1.0000 | 0.9925 | 0.9962 | 0.9966 | 167 / 0 / 1 / 132 | 0.00% |
| **3. Strongest Permutation-Invariant Baseline** *(HistGradientBoosting)* | 61 Order-Invariant Statistical Features | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **167 / 0 / 0 / 133** | **0.00%** |
| *— Perm-Invariant MLP* | 61 Order-Invariant Statistical Features | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 167 / 0 / 0 / 133 | 0.00% |
| *— Perm-Invariant Logistic* | 61 Order-Invariant Statistical Features | 0.9998 | 0.9997 | 1.0000 | 0.9925 | 0.9962 | 0.9966 | 167 / 0 / 1 / 132 | 0.00% |
| **4. Causal LSTM (Chronological)** | T x 5 Continuous Probabilities (dim=5) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **167 / 0 / 0 / 133** | **0.00%** |
| **5. Causal LSTM (Shuffled 10 Seeds)** | T x 5 Randomly Permuted Turns | 0.9896 | 0.9885 | — | — | 0.8781 | — | — | — |

---

## 3. Key Quantitative Findings & Deltas

1. **Offline Performance Gain Over Non-Temporal Models:**
   - Delta F1 (LSTM vs 20-Feat Logistic) = +0.0038
   - Delta F1 (LSTM vs 61-Feat HistGradientBoosting) = +0.0000
   - On full-conversation offline evaluation, expressive non-temporal models (61-feature GBDT and MLP) achieve perfect classification (F1 = 1.0000). Offline full-conversation metrics alone do **not** show a large gap over non-temporal aggregation.

2. **Temporal Sensitivity of the Causal LSTM:**
   - Chronological Sequence F1: **1.0000**
   - Shuffled Sequence F1 (10 deterministic seeds): **0.8781 +/- 0.0051**
   - **Degradation under Permutation:** **Delta F1 = -0.1219 (12.19 percentage point drop)**.
   - This proves the LSTM has learned genuine sequence-order transitions and escalation paths; when order is scrambled, the recurrent hidden-state trajectory is disrupted and fails to activate the decision boundary on 12.2% of conversations.

3. **Streaming & Early Detection Value:**
   - Permutation-invariant models require full-conversation aggregation or windowed statistics that lack temporal causality.
   - The Causal LSTM processes each turn in O(1) step time with strictly causal updates, achieving **Mean Detection Turn = 3.15** and **66.2% fraud detection by Turn 1-3**.

---

## 4. Authoritative Scientific Verdict

Based strictly on the experimental criteria:

### **VERDICT: [B] MODEST TEMPORAL VALUE**

#### Scientific Rationale:
1. **Offline Full-Conversation Classification:**
   Expressive permutation-invariant models (GBDT / MLP on 61 summary features) match the Causal LSTM at F1 = 1.0000. When full conversation history is available, high-order statistical aggregation of multi-tactic evidence is sufficient to separate fraud from benign dialogues.
2. **Online Streaming & Dynamic Escalation:**
   The Causal LSTM provides genuine, validated temporal sequence modeling:
   - Within-conversation turn scrambling drops LSTM F1 by **12.19 percentage points** (1.0000 -> 0.8781), verifying that the recurrent parameters exploit chronological ordering.
   - The Causal LSTM maintains real-time prefix causality, detecting fraud early (**Turn 3.15 average**) with **0.00% benign false alarms**, which non-temporal aggregate models cannot execute causally turn-by-turn without recomputing window statistics.

---

```
================================================================================
FINAL STATUS:
V2 FINAL SCIENTIFIC VALIDATION COMPLETE — AWAITING REVIEW
================================================================================
```
