# Phase 9 — Ablation Study

## Objective
Determine exactly which data curation strategy provided the performance boost.

## F1 Performance on Realistic Pilot
| Model Configuration | F1 | Notes |
|---------------------|----|-------|
| A. Original V2 (Synthetic Only) | 0.0645 | Massive domain shift failure. |
| B. + Real-World Data | 0.720 | Solved the clean-transcript short-utterance issue. |
| C. + ASR Augmentation (No Real Data) | 0.150 | Taught the model to handle dropped words, but it still didn't understand human conversational phrasing. |
| D. Real-World + ASR Aug. (Phase 9) | **0.845** | Synergistic effect. The model learned human phrasing AND how to handle noisy transcripts. |

## Conclusion
Real-world conversational data is absolutely essential. ASR augmentation provides a secondary robustness boost but is useless without first fixing the underlying semantic domain shift.
