# Phase 8 — Model Integrity Regression

## Objective
Verify that the scientifically validated Phase 5/6 models remain untouched and un-retrained.

## Verification
- DistilBERT SHA256 unchanged: **PASS** (`76d2d4e440f65cb1e9f275a6496a0dcd6751a35e1f85261882815ccdda59a684`)
- Causal LSTM SHA256 unchanged: **PASS** (`c6dd5b03ba91a20677ed4c7a148d23cba26ff08fa1528d5603a0cb1bd2ae64a1`)
- Threshold: tau = 0.20 **PASS**

## Conclusion
Model weights remain exactly as validated in previous phases. No unauthorized improvements were snuck into the codebase.
