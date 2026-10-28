# Phase 9 — ASR Robustness

## Objective
Verify that the `word_deletion` and `punctuation_loss` augmentations applied during Phase 9 training actually made the classifier robust to speech recognition errors.

## Results
- **Original V2 ASR-Degraded F1**: 0.021
- **Phase 9 ASR-Degraded F1**: 0.760

## Conclusion
The targeted ASR data augmentation worked. While 0.760 is slightly lower than the clean-audio F1 (0.845), the model no longer catastrophically collapses when words are randomly dropped. It has learned to infer manipulation intent from fragmented context.
