# Phase 10 — Final System Limitations

| Limitation | Evidence / Impact | Addressed? | Remaining Gap |
|------------|-------------------|------------|---------------|
| **Synthetic-Data Dependence** | Models rely heavily on synthetic training distributions. | Partially (Phase 9 Real-world fine-tuning). | Still requires massive scale human-labeled data. |
| **Limited Real-World Pilot Scale** | Evaluated on 500 test samples. | No. | Insufficient for commercial deployment guarantees. |
| **Long-Term Grooming (Romance)** | Romance scams lack overt turn-level manipulation markers. | No. | System currently fails to detect ultra-slow grooming. |
| **False Positives in Intense Calls** | FPR = 1.5%. Legitimate medical/emotional crises trigger alerts. | Addressed via non-accusatory intervention UI (Level 2). | FPR > 0 remains a friction point. |
| **ASR Variability** | Degrades gracefully, but extreme noise still causes failures. | Partially (Phase 9 augmentation). | Extreme real-world acoustic noise untested. |
| **Language/Dialect Variation** | Tested exclusively on standard US English. | No. | Completely untested on AAVE, regional dialects, or non-English. |
| **Privacy Redaction Limits** | NER/Regex fails on heavily disguised PII. | Partially (Best-effort regex). | PII leakage risk is never zero. |
| **No Clinical Validation** | Not tested on actual elderly victims in real psychological distress. | No. | Efficacy of the UI intervention remains theoretically modeled. |
| **Adversarial Adaptation** | Scammers could change tactics if system becomes widespread. | No. | Requires continuous retraining in production. |
