# Phase 10 — Final Latency Regression Report

## Objective
Verify the end-to-end processing latency of the final Phase 9 system.

## Metrics
- **Mean inference latency**: 32 ms per turn
- **Median inference latency**: 28 ms per turn
- **P95 latency**: 45 ms per turn
- **Max latency observed**: 65 ms per turn

## Breakdown
- **Classifier Inference (DistilBERT)**: ~25 ms
- **Temporal Inference (LSTM)**: ~2 ms
- **Preprocessing/Explainability/WebSocket**: ~5 ms

## ASR Note
These latencies represent the **Text-Processing Pipeline** only. They **EXCLUDE** the latency introduced by an upstream Automatic Speech Recognition (ASR) system. True voice-to-alert latency will depend heavily on the chosen ASR engine.

## Verdict
**PASS**. The system easily handles real-time constraints (latency well under the typical 500ms conversational gap).
