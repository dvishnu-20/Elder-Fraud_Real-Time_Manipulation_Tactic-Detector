# Phase 8 — Operational End-to-End Latency

## Objective
Measure the complete operational latency of processing a conversational turn through the full pipeline.

## Results
Measured over 320 turns during the Phase 8 pilot:
- **Mean Latency**: 36.97 ms per turn
- **P95 Latency**: 42.22 ms per turn
- **Maximum Latency**: 126.12 ms

*(Note: These figures measure text inference time. Total ASR latency depends on the external provider but typical ASR chunking adds ~500ms).*

## Conclusion
The system easily exceeds real-time constraints. Model execution time is functionally negligible compared to the human speaking rate.
