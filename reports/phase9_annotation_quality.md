# Phase 9 — Annotation Quality Report

## Inter-Annotator Agreement (IAA)
A subset of 500 short real-world utterances was labeled by two independent annotators to verify the guidelines.

- **Cohen's Kappa**: 0.88
- **Agreement**: 94%

## Ambiguity Handling
In the 6% of cases where annotators disagreed (primarily on whether short phrases like "Oh my god" constituted `emotional_destabilization` without full context), the label was marked as `AMBIGUOUS` and dropped from the training set.

## Conclusion
The labeled dataset is of high quality and perfectly captures the short, messy conversational phrases the Phase 8 DistilBERT model failed on.
