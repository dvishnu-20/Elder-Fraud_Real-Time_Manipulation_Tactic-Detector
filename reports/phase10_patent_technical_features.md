# Phase 10 — Patent-Relevant Technical Features

*DISCLAIMER: Novelty and patentability require a formal prior-art and patent search and cannot be concluded from this experiment.*

## 1. Modular Classifier → Temporal Risk Engine Architecture
- **Technical Mechanism**: Decoupling turn-level tactic feature extraction (Transformer) from stateful sequence risk aggregation (Causal LSTM).
- **Experimental Evidence**: Proved by freezing the LSTM in Phase 5 and successfully swapping the DistilBERT model in Phase 9 without breaking risk calibration.
- **Novelty Angle**: Most scam detection systems use monolithic classifiers. This decoupled approach allows rapid domain adaptation without losing temporal calibration.

## 2. 5-Dimensional Manipulation Tactic Representation
- **Technical Mechanism**: Mapping raw conversational text into a continuous 5D probability space representing psychological pressure (Urgency, Isolation, Authority, Emotion, Channel Control).
- **Novelty Angle**: Moves away from keyword blocking towards semantic tactic estimation.

## 3. Causal Prefix Risk Estimation
- **Technical Mechanism**: The LSTM calculates $Risk(T_n)$ using exclusively information from $T_1 ... T_n$, mathematically mirroring real-time streaming constraints without future leakage.
- **Experimental Evidence**: Phase 5 and 10 tests confirm strict causality and low latency (~2ms).

## 4. Deterministic Risk Explanation Generation
- **Technical Mechanism**: Alert explanations are synthesized dynamically based on the highest-activating dimensions in the 5D tactic vector combined with the temporal trend (e.g., "Increasing Urgency").
- **Novelty Angle**: Provides interpretable security alerts without requiring a massive generative LLM at the edge.
