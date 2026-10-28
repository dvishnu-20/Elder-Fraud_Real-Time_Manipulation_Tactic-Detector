# Phase 10 — Final Experimental Timeline

### Phase 3 → Dataset Generation
- **Objective**: Create 1,200 synthetic fraud/benign conversational transcripts.
- **Result**: Data generated and split (Train/Val/Test).

### Phase 4 → Tactic Classifier Training
- **Objective**: Train turn-level feature extractor.
- **Key Result**: DistilBERT V2 achieved 0.9413 Micro-F1.

### Phase 4.1 → Generalization Diagnosis
- **Objective**: Measure out-of-distribution performance on Phase 4 models.
- **Key Result**: Demonstrated synthetic-to-synthetic overfitting.

### Phase 5 → Temporal Validation
- **Objective**: Train Causal LSTM to aggregate Phase 4 tactic features.
- **Key Result**: LSTM achieved 0.985 F1 and early detection by Turn 5.
- **Frozen**: LSTM architecture and weights frozen. Threshold $\tau=0.20$ frozen.

### Phase 6 → External Validation
- **Objective**: Test frozen V2 system on independent external benchmark.
- **Key Result**: Achieved 0.875 F1. Confirmed moderate generalization.

### Phase 7 → Real-Time Integration
- **Objective**: Build engineering pipeline (FastAPI, WebSockets).
- **Key Result**: 32ms inference latency. Real-time streaming successful.

### Phase 8 → Controlled Pilot
- **Objective**: Test system on real-world noisy pilot data (ASR/human speech).
- **Key Result**: Severe performance collapse (F1 = 0.0645).
- **Limitation**: The system could not extract tactics from short, colloquial human speech.

### Phase 8.1 → Root-Cause Isolation
- **Objective**: Determine why Phase 8 failed.
- **Key Result**: Identified domain shift in the DistilBERT feature extractor. LSTM was mathematically sound.

### Phase 9 → Targeted Robustness Improvement
- **Objective**: Fix the Phase 8 limitation without altering the LSTM.
- **What Changed**: Retrained DistilBERT on 5,000 real-world samples + ASR augmentation.
- **Key Result**: Absolute F1 improved from 0.0645 to 0.845.
- **Scientific Conclusion**: Real-world data and noise augmentation successfully bridged the domain gap.

### Phase 10 → Final Validation & Freeze
- **Objective**: Audit reproducibility and integrity.
- **Key Result**: All metrics reproduced. System mathematically verified.
- **Frozen**: Entire architecture, datasets, and reports frozen for publication.
