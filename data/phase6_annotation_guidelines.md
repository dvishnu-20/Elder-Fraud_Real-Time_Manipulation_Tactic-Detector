# Phase 6: Authoritative Tactic Annotation Guidelines

These guidelines define the ground-truth annotation protocol for turn-level psychological manipulation tactics in elder-fraud detection.

---

## 1. Guiding Principles

1. **Behavior Over Keywords:**
   Tactic presence is determined by *manipulative conversational intent and behavioral coercion*, NOT by the presence of isolated keywords (e.g., "bank", "police", "urgent", "payment", "account" do NOT automatically imply a tactic).
2. **Institutional Legitimacy Disambiguation:**
   Legitimate deadlines, official titles, and routine security protocols that encourage independent verification are **NOT** manipulative tactics.
3. **Multi-Label Rule:**
   A single turn may contain multiple tactics if distinct clauses exhibit separate manipulative behaviors (e.g., *"You have 15 minutes before the federal marshal arrives, and you must not tell your spouse"* $\to$ `["urgency", "authority", "isolation"]`).

---

## 2. Tactic Definitions & Annotation Criteria

### A. `urgency` (Time-Pressure & Coercive Deadlines)
- **Positive Criteria:** Artificial deadlines designed to induce panic, bypass cognitive evaluation, and prevent deliberation (e.g., *"You have 15 minutes to pay or your power will be cut"*, *"Stay on the line and complete this wire immediately"*).
- **Negative Criteria:** Routine schedule reminders or standard business hours (e.g., *"Our office closes at 5:00 PM today"*, *"Flight check-in opens 24 hours in advance"*).
- **Borderline Cases:** A reminder with a date (e.g., *"Payment is due by the 1st of the month"*) is `none` unless accompanied by punitive immediate threats.

### B. `isolation` (Verification Suppression & Secrecy)
- **Positive Criteria:** Explicit or implicit instructions forbidding the victim from consulting family, bank tellers, police, attorneys, or third parties (e.g., *"Do not tell the bank teller why you are withdrawing cash"*, *"This is a gagged federal investigation; speaking to anyone is a felony"*).
- **Negative Criteria:** Standard privacy or HIPAA statements (e.g., *"For your privacy, please ensure no one else is listening while we verify your medical records"*, *"We recommend checking your statement privately"*).
- **Borderline Cases:** Advising someone not to share their one-time passcode (OTP) with anyone (including the caller) is legitimate security practice (`none`).

### C. `authority` (Institutional Power Coercion & Intimidation)
- **Positive Criteria:** Invoking legal, governmental, law enforcement, or executive status to command obedience or threaten punitive legal consequences (e.g., *"I am Special Agent Miller badge #4092, and a federal warrant has been issued for your arrest"*, *"You are in direct contempt of court"*).
- **Negative Criteria:** Routine identification of role in legitimate contexts (e.g., *"Hello, I'm Dr. Smith's receptionist calling to confirm your appointment"*, *"This is Detective Adams with a community outreach follow-up"*).
- **Borderline Cases:** Stating a company department (e.g., *"I am calling from the fraud prevention unit"*) without punitive threats or demands for immediate funds is `none`.

### D. `emotional_destabilization` (Guilt, Fear, Shame & Catastrophizing)
- **Positive Criteria:** Deliberately exploiting familial love, intense fear, shame, or catastrophe to shatter emotional equilibrium (e.g., *"Your grandson was in a terrible accident and is crying in a jail cell"*, *"How could you be so reckless with your family's life savings?"*).
- **Negative Criteria:** Expressions of routine empathy, clinical updates, or customer service apologies (e.g., *"I'm sorry to hear that your flight was delayed"*, *"We apologize for any inconvenience this maintenance caused"*).
- **Borderline Cases:** Expressing concern about billing arrears (e.g., *"We want to help you avoid service interruption"*) is `none`.

### E. `channel_control` (Communication Hijacking & Surveillance)
- **Positive Criteria:** Demanding persistent phone connectivity, forcing the victim to switch to unmonitored channels, or commanding software installation to gain remote device takeover (e.g., *"Do not hang up this call while driving to the store"*, *"Download AnyDesk immediately and provide me the 9-digit access code"*, *"Switch to Telegram for encrypted communication"*).
- **Negative Criteria:** Offering standard support channels or callback options (e.g., *"You can view your bill online at our official portal or call us back on the number on your card"*, *"Feel free to reply to this email ticket"*).
- **Borderline Cases:** Asking if the recipient can hear clearly or requesting a callback number for dropped calls is `none`.

---

## 3. Class `none`
- **Definition:** Any conversational turn that contains zero manipulative behaviors. Includes greetings, clarifications, legitimate questions, benign explanations, and recipient responses.
