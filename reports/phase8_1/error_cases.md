# Phase 8.1 — Error Cases Log

Representative examples of Classifier and Temporal misses from the diagnostic run.

### Case 1: Classifier Miss (Authority & Urgency)
- **Text**: "I'm calling from your bank's fraud department. Your account is compromised."
- **Ground Truth**: `authority`, `urgency`
- **DistilBERT Probabilities**: 
  - Urgency: 0.0002
  - Isolation: 0.0013
  - Authority: 0.0055
  - Emotional: 0.0009
  - Channel Control: 0.0002
- **Failure Mode**: Catastrophic classifier failure on clean text.

### Case 2: Classifier Miss (Isolation & Channel Control)
- **Text**: "Don't call anyone or hang up, or I will dispatch the officers now."
- **Ground Truth**: `isolation`, `channel_control`
- **DistilBERT Probabilities**: 
  - Urgency: 0.0004
  - Isolation: 0.0224
  - Authority: 0.0002
  - Emotional: 0.0006
  - Channel Control: 0.0012
- **Failure Mode**: Complete failure to detect explicit isolation and control commands.

### Case 3: Temporal Miss
- **Situation**: DistilBERT successfully breached the $0.50$ threshold on an individual tactic in a turn, feeding a positive signal to the Causal LSTM.
- **Result**: The final temporal risk reached $0.148$, failing to cross the $\tau=0.20$ threshold.
- **Failure Mode**: The upstream signals were so weak and sparse throughout the conversation that the LSTM correctly calculated the aggregate risk as low. The LSTM behaved mathematically correctly given the starved input vector.
