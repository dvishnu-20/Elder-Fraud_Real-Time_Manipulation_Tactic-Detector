# Phase 8.1 — ASR Impact Analysis

## Objective
Determine whether Automatic Speech Recognition (ASR) degradation was the primary cause of missed detections in Phase 8.

## Results
- **Clean Transcript Detections**: 0 / 2 fraud conversations.
- **ASR-Degraded Detections**: 0 / 2 fraud conversations.

## Conclusion
While ASR degradation (simulated at 20% word deletion) theoretically harms performance, it is **not the primary bottleneck**. The classifier already completely fails on clean reference transcripts. We cannot measure the marginal impact of ASR degradation until the baseline clean performance is fixed.
