# Phase 8.1 — Root Cause Diagnostic Report

## Scientific Question
Why did sensitivity (recall) severely degrade during the Phase 8 Real-World Pilot while maintaining 0% false positives?

## Findings
The diagnostic experiment isolated the failure to the **DistilBERT V2 Tactic Classifier**. 

Even on **clean, perfectly transcribed reference text**, DistilBERT failed to assign probabilities $>0.5$ to obvious manipulation tactics. For example, the sentence *"I'm calling from your bank's fraud department. Your account is compromised."* generated tactic probabilities of $<0.01$ across all five categories. 

Because DistilBERT (the feature extractor) output near-zero tactic probabilities, the downstream Causal LSTM was starved of input signal and correctly refused to elevate the temporal risk, resulting in massive False Negatives.

## Primary Cause
**[C] PRIMARILY TURN-LEVEL CLASSIFIER LIMITATION**

## Evidence
1. **Clean vs Degraded Performance**: Both clean and degraded transcripts failed equally (TP=0, FN=2 in the diagnostic set). ASR degradation was not the root cause because the model failed on clean audio first.
2. **Short Utterance Impact**: DistilBERT successfully identified 3 tactics on long utterances but failed heavily on shorter, punchy sentences characteristic of real-world dialogue.
3. **Probability Failure**: Raw probabilities on clear fraud indicators (e.g., "federal investigation") barely reached ~0.37, falling short of the detection threshold.
4. **Temporal State Starvation**: The `temporal_miss` errors logged indicate that even when DistilBERT detected *some* signal, it wasn't strong or sustained enough for the LSTM to breach the $\tau=0.20$ threshold.

## Conclusion
The root cause is a **Domain Shift** between the synthetic training dataset (Phase 2) and real-world conversational phrasing. DistilBERT overfit to the linguistic patterns of the LLM that generated its training data and fails to generalize to natural human dialogue.
