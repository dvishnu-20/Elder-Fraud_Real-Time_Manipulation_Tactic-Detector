# Phase 8.1 — Short Utterance Analysis

## Objective
Determine if DistilBERT performance degrades heavily on short conversational utterances compared to longer ones.

## Results
Based on clean transcripts:
- **Short Utterances ($\le 6$ words)**: TP = 0, FN = 1
- **Long Utterances ($> 6$ words)**: TP = 3, FN = 4

## Conclusion
While long utterances had a higher absolute number of True Positives, the overall recall remained extremely poor across both lengths. DistilBERT failed to properly classify 11 and 13-word utterances just as it failed on shorter ones. This indicates a broader linguistic domain shift problem rather than a strict "character limit" issue.
