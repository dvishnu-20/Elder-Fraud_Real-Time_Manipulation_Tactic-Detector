# Phase 10 — Final Paper-Ready Results Table

| Phase | Dataset | Model | F1 | Recall | Specificity | FPR | Detection Latency | Conclusion |
|-------|---------|-------|----|--------|-------------|-----|-------------------|------------|
| **Phase 5** | Synthetic V2 | Original V2 | 0.985 | 1.000 | 1.000 | 0.0% | Turn 5 | Baseline established. Temporal aggregation works on clean text. |
| **Phase 6** | External (Benchmark) | Original V2 | 0.875 | 0.937 | 0.857 | 14.3% | Turn 4.4 | Moderate generalization outside training set. |
| **Phase 7** | Real-Time Eng. | Original V2 | N/A | N/A | N/A | N/A | N/A | Engineering integration successful. |
| **Phase 8** | Controlled Pilot (Real) | Original V2 | 0.064 | 0.033 | 1.000 | 0.0% | Failed | Severe domain shift observed. Fails on short human utterances. |
| **Phase 8.1** | Root Cause Iso. | Original V2 | N/A | N/A | N/A | N/A | N/A | Domain gap isolated to Tactic Classifier (DistilBERT). LSTM verified sound. |
| **Phase 9** | Real-World Held-Out | Phase 9 + LSTM V2 | 0.845 | 0.820 | 0.985 | 1.5% | Turn 4 | Robustness improved. Real-world phrasing gap closed. |
| **Phase 10** | All Benchmarks | Final System | 0.845* | 0.820* | 0.985* | 1.5%* | Turn 4 | Results successfully reproduced. Research frozen. |

*\* Values refer to the primary Real-World Pilot evaluation constraint.*
