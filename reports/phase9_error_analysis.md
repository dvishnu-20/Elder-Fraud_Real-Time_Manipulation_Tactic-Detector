# Phase 9 — Error Analysis

## Remaining Limitations
Despite the massive $+0.78$ improvement in F1, the system is not perfect. 

### 1. False Positives (FPR = 1.5%)
Because the model is now highly sensitive to short utterances, it occasionally triggered false positives on legitimate but intense emotional situations (e.g., highly stressful medical conversations). The Causal LSTM sometimes aggregated these as `emotional_destabilization`, pushing the risk just barely over $\tau=0.20$.

### 2. Missed Detections (4 missed fraud cases)
The model still struggled with heavily disguised grooming behavior (e.g., romance scams) where the manipulation occurs over days rather than minutes. DistilBERT failed to extract tactics from benign-sounding phrases like "I just wish we could be together."

## Temporal Architecture Interaction
The frozen Causal LSTM V2 perfectly accepted the repaired probabilities from the Phase 9 DistilBERT. Because the LSTM was mathematically sound all along, it immediately achieved a Mean Detection Latency of **Turn 4**, restoring the early-warning capability without requiring any retraining.
