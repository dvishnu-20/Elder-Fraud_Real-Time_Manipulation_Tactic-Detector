# Phase 8 — Causality Regression

## Objective
Repeat the Phase 7 causality tests to ensure future-turn leakage remains eradicated.

## Verification
- **Future Leakage**: `test_no_future_leakage` **PASS**
- **Prefix Consistency**: `test_prefix_consistency` **PASS**

## Conclusion
Risk at time $t$ strictly depends only on turns $\le t$. The pipeline is fully causal.
