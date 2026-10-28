# Phase 6: External Dataset Requirements & Specification

## 1. Objective
Phase 6 evaluates whether the Elder-Fraud Detector V2 pipeline (frozen DistilBERT turn classifier + frozen Causal LSTM risk model) generalizes beyond the synthetic training distribution to external, realistic, multi-domain dialogues.

---

## 2. Ingestion Criteria & Domain Coverage

The external evaluation dataset must satisfy the following strict requirements:
1. **Diverse Realistic Scam Typologies (Fraud Class):**
   - **IRS / Tax Impersonation:** Demands for immediate payment under threat of arrest, asset forfeiture, or warrant issuance.
   - **Bank / Financial Security Impersonation:** Urgent alerts claiming compromised accounts, unauthorized wires, or frozen assets requiring immediate credential/OTP disclosure or fund relocation.
   - **Tech Support / Remote Access Exploitation:** Alerts claiming malware infection, requiring AnyDesk/TeamViewer installation, remote access granting, and banking session hijacking.
   - **Grandchild / Family Emergency Bail Scam:** Urgent emotional distress calls claiming a family member is imprisoned/injured and requiring confidential bail transfer.
   - **Utility Cutoff Impersonation:** Threat of immediate power/water termination within 30–60 minutes unless payment is made via prepaid card or cryptocurrency.
   - **Social Security / Federal Legal Notice:** Demands to verify SSN and transfer funds to a "secure government safety locker".

2. **Diverse Legitimate Institutional Interactions (Benign Class):**
   - **Authentic Bank Customer Support & Fraud Alerts:** Verification of legitimate charges where the institution encourages the customer to call the number on the back of their physical card or visit a local branch.
   - **Healthcare & Clinic Appointments:** Appointment reminders, prescription refill notifications, and HIPAA privacy disclosures.
   - **Utility Billing & Scheduled Maintenance:** Standard service calls, meter reading confirmations, and payment plan options with online portal verification.
   - **Legitimate Technical Support:** Troubleshooting internet connectivity or software licensing with standard ticket reference numbers and official support domain verification.
   - **Community / Family Social Calls:** Non-institutional conversations with emotional phrasing, routine check-ins, and everyday scheduling.

---

## 3. Structural & Format Requirements

1. **Dialogue Format:**
   - Multi-turn conversational structure with alternating `caller` and `recipient` speakers.
   - Realistic conversation length: 4 to 12 turns.
2. **Annotation Granularity:**
   - Turn-level multi-label annotations across the 5 authoritative tactics (`urgency`, `isolation`, `authority`, `emotional_destabilization`, `channel_control`, or `none`).
   - Conversation-level ground truth binary label: `1` (Fraud) or `0` (Benign).
3. **Strict Non-Synthetic Provenance:**
   - Transcripts must not be generated from the synthetic slot-filling templates used in V1 or V2 training.
   - Lexical variety, authentic conversational pauses, realistic recipient skepticism, and diverse phrasing must be present.
4. **Data Isolation:**
   - Must be split into `EXTERNAL_DEV` (optional exploration) and `EXTERNAL_TEST` (held-out benchmark).
