# Phase 8.1 — Tactic-Level Analysis

## Objective
Identify which specific tactics DistilBERT failed to extract from real-world speech.

## Breakdown of Misses
From the diagnostic script, the following specific ground-truth tactics were missed (probability remained $< 0.5$):
- **Authority**: Missed twice on clear statements ("bank's fraud department", "federal investigation").
- **Urgency**: Missed twice ("act immediately", "compromised").
- **Isolation**: Missed twice ("Do not tell anyone", "Don't call anyone or hang up").
- **Channel Control**: Missed once ("Don't call anyone or hang up").

## Conclusion
The failure is systemic across the tactics. The `authority` and `isolation` categories experienced catastrophic drops, as the model failed to connect real-world phrases to the synthetic representations it learned during Phase 2.
