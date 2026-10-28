# Phase 8 — Alert Timing Study

## Objective
Measure when alerts occur during a simulated fraud event and whether they precede simulated harm.

## Results
- **True Positives**: 1 / 30 fraud conversations.
- **Alert Timing**: For the single detected conversation, the alert was triggered at **Turn 3**.
- **Before Harm?**: Yes, Turn 3 occurred before the simulated wire transfer request, proving that the temporal model structure *can* trigger an early warning.

## Conclusion
While the *rate* of detection (recall) was severely crippled by generalization failure, the *timing* of the detection in successful cases remains strong, occurring early in the conversational prefix.
