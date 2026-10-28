# Phase 9 — Annotation Guidelines

## Core Directives
Annotators must evaluate short real-world conversational turns for the presence of the 5 manipulation tactics. 

### Context is King
Because Phase 8 proved the model failed on short utterances, annotators MUST label short sentences based on conversational context.
- **Example A**: "Wait, what?" (Speaker hears they owe the IRS $10,000) -> Label: `emotional_destabilization`
- **Example B**: "Wait, what?" (Speaker hears it will rain tomorrow) -> Label: `None`

### The 5 Tactics
1. **Urgency**: Deadlines, countdowns, forcing rapid action.
2. **Isolation**: Telling the victim to hide the conversation from family, bank tellers, or police.
3. **Authority**: Impersonating IRS, FBI, Bank Fraud Dept, Tech Support.
4. **Emotional Destabilization**: Inducing panic, confusion, or fear.
5. **Channel Control**: Demanding the victim stay on the phone, go to a quiet room, or not hang up.

### Hard Negatives
Legitimate use of these terms MUST NOT be labeled as tactics:
- "Please hold while I check your account." (Bank teller) -> `None`
- "This is 911, what is your emergency?" (Real police) -> `None`
