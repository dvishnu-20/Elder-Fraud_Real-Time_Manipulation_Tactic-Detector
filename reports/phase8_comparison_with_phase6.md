# Phase 8 vs Phase 6 Generalization Comparison

## Analysis
In Phase 6 (External Generalization), the Causal LSTM achieved:
- **F1**: 0.8750
- **FPR**: 14.3%

In Phase 8 (Controlled Pilot), the system achieved:
- **F1**: 0.0645
- **FPR**: 0.0%

## Why the Discrepancy?
Phase 6 evaluated the Causal LSTM using the **extracted feature arrays** from the Phase 6 dataset. It did not fully test DistilBERT's ability to extract those features from highly diverse, unstructured, non-GPT generated short phrases often found in noisy real-world ASR transcripts. 

When forced to parse raw ASR-degraded text in the Phase 8 pilot simulator, DistilBERT failed to assign high manipulation probabilities to obvious fraud statements (e.g., "I'm calling from the FBI. Send money now."). 

Because DistilBERT (the feature extractor) output near-zero tactic probabilities, the Causal LSTM never accumulated enough risk to cross the threshold, resulting in massive False Negatives (Recall = 3.3%).

## Conclusion
The system fails safely (0% FPR), but its sensitivity is fundamentally bottlenecked by the DistilBERT turn-classifier's inability to generalize beyond its synthetic training distribution to noisy, short, real-world conversational turns.
