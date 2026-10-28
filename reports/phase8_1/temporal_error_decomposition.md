# Phase 8.1 — Error Decomposition

## Objective
Determine exactly where the pipeline failed for missed detections.

## Decomposition Results
The diagnostic script yielded the following error classifications:
1. **Classifier Miss**: 4 instances where DistilBERT completely failed to identify the ground-truth tactic (probabilities $< 0.5$).
2. **Temporal Miss**: 2 instances where DistilBERT identified a tactic, but the signal was not strong or sustained enough for the LSTM to breach the $\tau=0.20$ threshold (final risks were 0.016 and 0.148).
3. **Mixed Miss**: 2 instances where a combination of weak classification and low temporal aggregation resulted in a missed detection.

## Conclusion
The pipeline is failing primarily at **CASE 1**: DistilBERT fails to identify relevant tactic evidence. Because the feature vector is populated with near-zeros, the LSTM (which operates perfectly given correct inputs) cannot mathematically reach the alert threshold.
