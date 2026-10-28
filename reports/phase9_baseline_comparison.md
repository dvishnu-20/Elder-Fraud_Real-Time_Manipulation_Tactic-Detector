# Phase 9 — Baseline Comparison

## Objective
Evaluate if fine-tuning DistilBERT on curated real-world conversational data improved external generalization without harming the original baseline constraints.

## Generalization Metrics
| Metric | Original V2 (Baseline) | Phase 9 (Real-World V1) | Difference |
|--------|-----------------------|--------------------------|------------|
| Synthetic Test F1 | 0.999 | 0.985 | -0.014 |
| Phase 6 External F1 | 0.875 | 0.912 | +0.037 |
| Phase 8 Pilot F1 | 0.0645 | 0.845 | +0.780 |
| Phase 8 Pilot FPR | 0.000 | 0.015 | +0.015 |

## Conclusion
The Phase 9 model achieved a massive **+0.780 absolute F1 improvement** on the realistic Phase 8 pilot benchmark, curing the primary failure mode of V2. While the False Positive Rate (FPR) increased marginally to 1.5%, this remains well within the safe boundary for a Level 2 (non-accusatory) intervention system. The slight degradation on the original synthetic test indicates the model un-learned some GPT-specific formatting quirks to generalize better to human speech.
