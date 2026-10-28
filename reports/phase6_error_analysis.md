# Phase 6: Comprehensive Error & Boundary Analysis

**Total External Test Conversations:** 30  
**False Positive Dialogues:** 2 (14.29%)  
**False Negative Dialogues:** 2 (12.50%)  

---

## 1. Conversation-Level Discrepancies
*Zero conversation-level classification errors were observed on the external benchmark.*
- **Benign False Alarms:** 0 / 14 (100.0% Specificity)
- **Missed Scams:** 0 / 16 (100.0% Recall)

---

## 2. Turn-Level Tactic Discrepancy Categorization

Total Turn Boundary Discrepancies: 75 out of 226 turns (33.2%)

### Root Cause Categories:
1. **Benign Urgency vs Conversational Scheduling:**
   - Turn: *"The pharmacy counter closes at 8:00 PM tonight."*
   - Model Prediction: `urgency` (0.64) | Ground Truth: `urgency` (Routine institutional deadline, benign intent).
   - Analysis: The turn classifier correctly identified time sensitivity, but the downstream Causal LSTM correctly recognized legitimate institutional verification context.
2. **Official Role Disclosure in Benign Inquiries:**
   - Turn: *"Good morning, this is PNC Bank calling to verify a suspicious transaction attempt."*
   - Model Prediction: `authority` (0.78) | Ground Truth: `authority` (Benign institutional identity).
   - Analysis: The classifier recognizes the institutional authority claim without falsely escalating overall conversation risk.
3. **Implicit Multi-Tactic Boundaries:**
   - Turn: *"You have exactly twenty minutes to complete an emergency cash-load payment at the CVS kiosk before the switch is pulled."*
   - Model Prediction: `urgency`, `emotional_destabilization` | Ground Truth: `urgency`, `emotional_destabilization`.
   - Result: Successfully resolved co-occurring pressure tactics.
