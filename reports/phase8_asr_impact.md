# Phase 8 — ASR Impact and Degradation Analysis

## Objective
Measure how automatic speech recognition (ASR) degradation—caused by background noise or poor channel quality—affects tactic classification and temporal risk.

## Methodology
In the Phase 8 Controlled Pilot Simulator, 20% of conversations (and all noise-category conversations) were subjected to a 15% random word-deletion rate.

## Results
- **Clean Audio Accuracy**: The DistilBERT model severely struggled to generalize to the clean human-written phrases in the pilot, leading to a recall of 3.3%.
- **Degraded Audio Accuracy**: The word-deletion significantly worsened the classifier's confidence. Because the model was trained on pristine, synthetic GPT-generated text (Phase 2), it proved highly brittle to missing words.
- **False Positives**: Despite the degradation, ASR errors **did not** cause false positives. The FPR remained 0.0%.

## Conclusion
The current DistilBERT tactic classifier is highly brittle to real-world phrasing and is further crippled by ASR omissions. 
**Crucially, ASR errors cause false negatives (missed fraud), not false positives.** The system fails safe.
