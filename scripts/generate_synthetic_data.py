import json
import uuid
import os
import random
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

# ==============================================================================
# PHASE 3.5: TEMPORALLY VALID DATASET GENERATOR
# ==============================================================================
# In this generator:
# 1. Individual tactics (urgency, isolation, authority, emotional_destabilization,
#    channel_control) appear in BOTH fraud and benign conversations.
# 2. Benign conversations contain legitimate tactic-like language (routine deadlines,
#    official titles, privacy handling, channel guidance) accompanied by
#    INDEPENDENT VERIFICATION and NON-COERCIVE behavior.
# 3. Fraud conversations contain manipulative escalation, delayed coercion,
#    interspersed normal turns, and ACTIVE SUPPRESSION OF VERIFICATION.
# 4. Fraud != single tactic presence. Fraud = conversation-level behavioral progression.
# ==============================================================================

class TemporalVocabularyPool:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        self.topics = [
            "banking", "tech_support", "government_tax", "medical_health",
            "package_delivery", "family_emergency", "insurance",
            "utilities_bill", "retail_promotions", "legal_affairs",
            "account_security", "customer_service"
        ]

        # -------------------------------------------------------------
        # 1. BENIGN LEGITIMATE TACTIC-LIKE CALLER TURNS
        # (Contains tactic-like terms/themes, labeled with tactic category or none
        #  depending on whether legitimate tactic behavior is present, but ALWAYS
        #  paired with non-coercive framing and explicit verification safety)
        # -------------------------------------------------------------
        self.benign_urgency_turns = [
            "Our branch lobby closes at 5:00 PM today, but you can securely complete your deposit at the ATM or online anytime this weekend.",
            "The annual promotional rate on your certificate of deposit expires on Friday; please take your time to review the renewal options online.",
            "Your quarterly water bill grace period ends tomorrow; you can pay through the official municipal portal or mail a check at your convenience.",
            "The discount window for early property tax payment concludes at midnight, though standard payments are accepted throughout the month.",
            "Your prescription refill request will remain active at the pharmacy for five business days before being returned to stock.",
            "Please note that the insurance enrollment window ends next Tuesday; feel free to discuss your plan options with your family before choosing.",
            "Your domain registration has two days remaining in its grace period; you may renew whenever ready via the official registrar dashboard.",
            "Flight check-in opens 24 hours prior to departure, so you have plenty of time to select your seats on the airline app.",
            "The early-bird registration discount closes tonight, but standard registration remains open all through next week.",
            "Our customer service line is available until 8 PM this evening if you'd like to finalize your appointment time today."
        ]

        self.benign_authority_turns = [
            "This is Officer Martinez from the municipal police community liaison office calling with a reminder about tomorrow's neighborhood meeting; you can confirm this call with the precinct desk.",
            "I am Agent Walker with the County Property Assessor's Department; you can verify my employee ID and this inquiry on our county government website.",
            "Good morning, this is Dr. Chen's clinical coordinator reaching out regarding your lab schedule; please feel free to call our main clinic number on your appointment card.",
            "This is Sarah Jenkins from State Farm Insurance claims division; you can look up my adjuster license on the official state insurance registry.",
            "I am Inspector Ross from the City Water Works following up on your meter inspection request; our dispatch number is on your monthly statement.",
            "This is Attorney Miller's paralegal calling about your scheduled estate planning consultation; our office credentials are on the state bar directory.",
            "Good afternoon, I am Nurse Clara from the cardiology department; you can check into your patient portal to review all physician notes.",
            "This is Detective Reynolds with the property recovery bureau letting you know your lost keys were turned in; feel free to call our precinct front desk directly."
        ]

        self.benign_isolation_privacy_turns = [
            "To safeguard your confidential health records under HIPAA regulations, let's move to a private examination room where we can speak freely.",
            "When entering your debit PIN at the checkout counter, please shield the keypad from view to keep your code private.",
            "For your financial confidentiality, we recommend reviewing your investment statement in a quiet room or private browser tab.",
            "Attorney-client privilege protects our discussions, allowing you to share all relevant details with our legal team in total confidence.",
            "We strongly encourage you to discuss this pension distribution option with your spouse, trusted children, or financial advisor before deciding.",
            "Please keep your account recovery passphrase written down in a private, secure location at home rather than sharing it with anyone.",
            "Our private banking suite is reserved for clients who prefer a quiet environment to discuss their retirement planning."
        ]

        self.benign_emotional_empathy_turns = [
            "I understand how stressful unexpected billing issues can be, so please rest assured we are here to help resolve this step by step.",
            "We deeply apologize for the flight cancellation and want to make sure you and your family are comfortably accommodated on the next flight.",
            "Medical appointments can feel overwhelming, so please take a deep breath and take as much time as you need to ask questions.",
            "I realize hearing about suspicious activity on your card is alarming, but our fraud detection team has already secured your funds completely.",
            "We know how frustrating computer glitches are; let's work through this printer setup calmly until everything is working smoothly."
        ]

        self.benign_channel_turns = [
            "You can manage your utility payments either through our automated phone system, our official mobile app, or by mailing a standard check.",
            "If you prefer not to discuss this over the phone, you are welcome to visit any local branch or log into our secure online banking portal.",
            "Our technician can guide you over the phone, or if you prefer, we can schedule an in-person home service visit next Wednesday.",
            "To review the contract details, we have sent a copy via certified postal mail and also uploaded it to your secure customer account.",
            "Feel free to reply via our verified customer support portal or call the direct customer service number printed on your billing statement."
        ]

        # -------------------------------------------------------------
        # 2. BENIGN INDEPENDENT VERIFICATION PHRASES (Crucial Discriminator)
        # -------------------------------------------------------------
        self.benign_verification_offers = [
            "For your security, please hang up and dial the customer service number located on the back of your credit card to verify this notice.",
            "You can independently verify this communication by visiting the official .gov portal and checking your message inbox.",
            "Take all the time you need, and feel free to consult with your family or financial advisor before making any decisions.",
            "If you have any doubts, please visit your local branch in person with two forms of identification.",
            "You do not need to make any payment right now; you can review the formal written invoice arriving in your mailbox.",
            "Please verify my identity by calling our main office switchboard listed in the public directory."
        ]

        # -------------------------------------------------------------
        # 3. BENIGN ROUTINE / HARD NEGATIVE CALLER TURNS
        # -------------------------------------------------------------
        self.benign_routine_turns = [
            "Hello, I am following up on your inquiry regarding our seasonal lawn aeration and maintenance schedule.",
            "Good morning, I am calling to confirm your dental hygiene appointment for next Thursday at 10:30 AM.",
            "This is a courtesy reminder that your local library books are due for renewal early next week.",
            "We noticed your package from the hardware store has arrived at the local distribution facility for morning delivery.",
            "I am checking to see if the replacement air filters we sent for your HVAC unit arrived in good condition.",
            "Thank you for contacting customer service today; could you please provide your ticket reference number?",
            "Our community center is hosting a free digital literacy workshop for neighborhood seniors next Tuesday.",
            "Your monthly electric usage report is available for viewing on your utility account dashboard.",
            "We wanted to let you know that our branch lobby hours will be adjusted for the upcoming federal holiday.",
            "I am calling to confirm that your prescription refill has been processed and is ready for pickup."
        ]

        # -------------------------------------------------------------
        # 4. BENIGN RECIPIENT RESPONSES
        # -------------------------------------------------------------
        self.benign_recipient_responses = [
            "Thank you for the update; I will log into my official account online and check the details.",
            "Understood, I will discuss this with my daughter tonight and call back if we have any questions.",
            "I appreciate the reminder; I will stop by the neighborhood branch office tomorrow morning.",
            "Thanks for letting me know, I'll take a look at the paperwork arriving in the mail.",
            "Okay, I will verify this through the phone number printed on the back of my card.",
            "That sounds very reasonable; thank you for giving me the time to review this carefully.",
            "I have noted down the reference number and will check it on your website.",
            "Thank you for being so thorough and helpful; have a wonderful day."
        ]

        # -------------------------------------------------------------
        # 5. FRAUD MANIPULATIVE CALLER TURNS (Coercive, Deceptive)
        # -------------------------------------------------------------
        self.fraud_urgency_turns = [
            "You have exactly twenty minutes to authorize this emergency wire transfer before federal marshals execute an asset seizure.",
            "This enforcement window will permanently expire the moment you disconnect; you must execute the transaction right this second.",
            "If you do not complete the safety deposit within fifteen minutes, your social security benefits will be irrevocably terminated.",
            "Law enforcement is already preparing to dispatch officers to your residence; immediate payment is your only chance to cancel the warrant.",
            "The system clock is running down; every second you delay increases your personal liability and guarantees immediate account freezing."
        ]

        self.fraud_authority_turns = [
            "I am Senior Special Agent Vance from the Federal Criminal Investigation Directorate, and you are legally required to obey my direct orders.",
            "This is Chief Magistrate Sterling of the National Asset Oversight Commission; refusing my directive constitutes federal obstruction of justice.",
            "I hold direct enforcement authority from the Department of Justice, and your compliance with this emergency protocol is non-negotiable.",
            "This is Lead Investigator Thornton from the Financial Crimes Enforcement Bureau; I am commanding you to execute this protective measure.",
            "Under federal statutory authority, I am ordering you to comply immediately with our asset stabilization procedure."
        ]

        self.fraud_isolation_turns = [
            "Do not tell the bank teller or anyone in your family why you are withdrawing this cash; the entire local branch is under criminal investigation.",
            "You are bound by a strict federal non-disclosure order; speaking to your children, spouse, or attorney will result in immediate felony charges.",
            "Keep this operation strictly between us; if any third party finds out, our protective immunity agreement is instantly voided.",
            "Go into a locked room right now where no one can overhear you, and do not let anyone in your household know about this call.",
            "Do not contact your local branch or any outside advisor; they will give you bad advice and cause your accounts to be permanently lost."
        ]

        self.fraud_emotional_turns = [
            "Your entire life savings are being looted by international crime syndicates right now, and you are about to lose everything you own!",
            "Your grandson is currently locked in a detention cell crying in agony, and only your immediate payment can secure his release!",
            "You are facing total financial ruin and public humiliation; squad cars are already driving down your street to arrest you!",
            "How could you be so careless with your family's future? If you don't act right now, you will be left homeless and destitute!",
            "You are on the verge of catastrophic devastation because of your negligence; listen to me before it is too late!"
        ]

        self.fraud_channel_turns = [
            "Keep this mobile line open continuously and do not hang up while you drive to the store to purchase the prepaid voucher cards.",
            "Download the remote access utility immediately and give me the nine-digit control code so I can manipulate your screen directly.",
            "You must convert your cash into Bitcoin at the designated grocery store terminal and scan the QR barcode I send you.",
            "Do not use your standard banking application or visit your branch; execute the payment only through this direct link.",
            "Stay on this encrypted phone channel until the entire transaction is validated; disconnecting will trigger immediate arrest."
        ]

        # -------------------------------------------------------------
        # 6. FRAUD ACTIVE VERIFICATION SUPPRESSION (Crucial Discriminator)
        # -------------------------------------------------------------
        self.fraud_suppression_turns = [
            "Do NOT call the number on your card or attempt to verify this with your bank; our investigators have flagged internal employees as suspects.",
            "Do NOT hang up this phone to check any website; breaking this secure connection will forfeit your emergency protection immediately.",
            "You are legally prohibited from contacting outside parties or family members for verification during an active federal operation.",
            "Calling any public customer service number will tip off the syndicate and cause your accounts to be emptied instantly.",
            "Do not question my credentials or delay to seek second opinions; hesitation will result in immediate prosecution."
        ]

        # -------------------------------------------------------------
        # 7. FRAUD INNOCUOUS INTRODUCTORY / NORMAL TURNS
        # (Allows fraud dialogues to start with normal rapport or have normal gaps)
        # -------------------------------------------------------------
        self.fraud_normal_gaps = [
            "Good morning, I am calling regarding a routine administrative verification on your residential file.",
            "Thank you for answering, I have a few standard questions to update your contact preferences.",
            "Could you please confirm if you received our recent mailer regarding seasonal account maintenance?",
            "I see your account has been active for several years, which is why we are conducting this routine security check.",
            "Let me pull up your account summary on my system; please bear with me for just a moment.",
            "We want to ensure your profile information is completely up to date in our central database."
        ]

        # -------------------------------------------------------------
        # 8. FRAUD RECIPIENT RESPONSES (Distressed, Coerced, or Hesitant)
        # -------------------------------------------------------------
        self.fraud_recipient_responses = [
            "Oh goodness, I'm terrified! Please tell me what I need to do to protect my money!",
            "I don't understand, but I will do whatever you say officer, please don't send police to my home.",
            "I'm getting in my car right now, I have the phone with me and won't hang up.",
            "Please help me, I have my checkbook ready, how much do I need to withdraw?",
            "I won't say anything to the teller, I promise; just please make sure my grandson is safe.",
            "I am alone in my room with the door shut; what is the next step?"
        ]


class ProceduralTemporalDatasetGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.vocab = TemporalVocabularyPool(seed)

    def generate_benign_conversation(self, conv_id: str, archetype: str) -> Dict[str, Any]:
        """
        Generates a rich, realistic benign conversation.
        Archetypes:
        - 'routine_service': pure normal/hard-negative turns with normal resolution.
        - 'authority_with_verification': legitimate official caller + explicit verification offer.
        - 'urgency_with_safety': routine deadline/grace period + non-coercive timeline.
        - 'privacy_confidentiality': legitimate HIPAA/privilege/PIN advice + trusted family recommendation.
        - 'channel_guidance': legitimate omnichannel support options (online, branch, phone, mail).
        - 'multi_tactic_benign': combines 2+ tactic-like categories (e.g. authority + deadline + verification).
        - 'suspicious_looking_benign': customer has fraud alert on card, but bank rep tells them to hang up and call back on card number.
        """
        topic = random.choice(self.vocab.topics)
        turns = []
        turn_idx = 1

        # Determine sequence structure based on archetype
        if archetype == "routine_service":
            # 4 to 8 turns of routine, realistic customer interaction
            num_exchanges = random.choice([2, 3, 4])
            for _ in range(num_exchanges):
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "caller",
                    "text": random.choice(self.vocab.benign_routine_turns),
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "recipient",
                    "text": random.choice(self.vocab.benign_recipient_responses),
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1

        elif archetype == "authority_with_verification":
            # Legitimate authority intro -> recipient inquiry -> verification offer -> normal close
            # Turn 1: Caller (authority-like, but non-coercive)
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_authority_turns),
                "labels": ["authority"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "Thank you for letting me know. How can I confirm your department's phone number?",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            # Turn 3: Caller offers independent verification
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_verification_offers),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": random.choice(self.vocab.benign_recipient_responses),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1

            # Optional extension turns
            if random.random() < 0.5:
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "caller",
                    "text": "Have a great day, and feel free to reach out once you've reviewed the materials.",
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "recipient",
                    "text": "Thank you very much. Goodbye.",
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1

        elif archetype == "urgency_with_safety":
            # Normal opener -> legitimate deadline -> verification / safety advice -> normal close
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_routine_turns),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "Yes, I was wondering about the schedule and payment terms.",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            # Tactic-like urgency turn (legitimate deadline)
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_urgency_turns),
                "labels": ["urgency"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "Good to know. I'll make sure to review it online before the deadline.",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            # Verification / reassure
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_verification_offers),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": random.choice(self.vocab.benign_recipient_responses),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1

        elif archetype == "privacy_confidentiality":
            # Normal opener -> legitimate privacy advice -> recipient confirms -> verification / family consult
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_isolation_privacy_turns),
                "labels": ["isolation"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "I appreciate that. I always want to make sure my personal details stay protected.",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_verification_offers),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": random.choice(self.vocab.benign_recipient_responses),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1

            if random.random() < 0.6:
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "caller",
                    "text": "We will mail the formal paperwork to your home address on file.",
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "recipient",
                    "text": "Sounds wonderful, thank you.",
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1

        elif archetype == "channel_guidance":
            # Normal opener -> channel options -> recipient confirms -> verification
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_routine_turns),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "What is the best way for me to submit the documents?",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_channel_turns),
                "labels": ["channel_control"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": random.choice(self.vocab.benign_recipient_responses),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1

        elif archetype == "multi_tactic_benign":
            # 2+ distinct benign tactic-like turns (e.g. authority + deadline + channel + verification)
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_authority_turns),
                "labels": ["authority"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "I see, thank you for reaching out.",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_urgency_turns),
                "labels": ["urgency"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "Can I submit the forms through the online portal?",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_channel_turns),
                "labels": ["channel_control"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "Thank you. Is there an official contact number where I can double-check this?",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_verification_offers),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": random.choice(self.vocab.benign_recipient_responses),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1

        else: # suspicious_looking_benign
            # Emotional reassurance + verification advice
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_emotional_empathy_turns),
                "labels": ["emotional_destabilization"], # Empathy/emotion theme
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "I was really worried when I saw the alert on my phone.",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.benign_verification_offers),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": random.choice(self.vocab.benign_recipient_responses),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1

        return {
            "conversation_id": conv_id,
            "conversation_level_label": "benign",
            "topic": topic,
            "archetype": archetype,
            "turns": turns
        }

    def generate_fraud_conversation(self, conv_id: str, archetype: str) -> Dict[str, Any]:
        """
        Generates a rich, realistic fraud conversation.
        Archetypes:
        - 'immediate_coercion': fast authority/urgency escalation + suppression.
        - 'delayed_escalation': 2 to 4 normal turns first, then manipulation escalates.
        - 'multi_gap_progression': normal turn -> tactic -> normal turn -> normal turn -> tactic -> suppression.
        - 'late_tactic_burst': long normal discussion, then urgency/channel control at end.
        - 'diversified_sequence': combinatorial progression across 3+ tactics.
        """
        topic = random.choice(self.vocab.topics)
        turns = []
        turn_idx = 1

        if archetype == "immediate_coercion":
            # Authority -> Urgency -> Isolation -> Suppression -> Channel Control
            seq = [
                ("authority", random.choice(self.vocab.fraud_authority_turns)),
                ("urgency", random.choice(self.vocab.fraud_urgency_turns)),
                ("isolation", random.choice(self.vocab.fraud_isolation_turns)),
                ("none", random.choice(self.vocab.fraud_suppression_turns)), # Active verification suppression
                ("channel_control", random.choice(self.vocab.fraud_channel_turns))
            ]
            for tactic, text in seq:
                labels = [tactic] if tactic != "none" else ["none"]
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "caller",
                    "text": text,
                    "labels": labels,
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "recipient",
                    "text": random.choice(self.vocab.fraud_recipient_responses),
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1

        elif archetype == "delayed_escalation":
            # 2 to 4 normal turns before first manipulation
            # Normal exchange 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.fraud_normal_gaps),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "Yes, I am listening. What is this concerning?",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            # Normal exchange 2
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": "We are verifying an unapproved financial request flagged under your name.",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "An unapproved request? I didn't authorize anything!",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            # Escalation: Emotional -> Urgency -> Isolation -> Suppression
            seq = [
                ("emotional_destabilization", random.choice(self.vocab.fraud_emotional_turns)),
                ("urgency", random.choice(self.vocab.fraud_urgency_turns)),
                ("isolation", random.choice(self.vocab.fraud_isolation_turns)),
                ("channel_control", random.choice(self.vocab.fraud_channel_turns))
            ]
            for tactic, text in seq:
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "caller",
                    "text": text,
                    "labels": [tactic],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "recipient",
                    "text": random.choice(self.vocab.fraud_recipient_responses),
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1

        elif archetype == "multi_gap_progression":
            # Tactic -> normal gap -> normal gap -> tactic -> suppression -> channel control
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.fraud_authority_turns),
                "labels": ["authority"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "I see, let me get a pen to write this down.",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            # Normal gap turn 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.fraud_normal_gaps),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "Okay, I have my file ready.",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            # Normal gap turn 2
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": "I am reviewing your account balance history on my terminal right now.",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "Alright, what does it show?",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            # Tactic 2 (Urgency)
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.fraud_urgency_turns),
                "labels": ["urgency"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": "Can I call my son or the local police station to verify this?",
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            # Verification suppression & Isolation
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.fraud_suppression_turns),
                "labels": ["isolation"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": random.choice(self.vocab.fraud_recipient_responses),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1

        elif archetype == "late_tactic_burst":
            # 3 normal exchanges, then sudden late manipulation burst
            for _ in range(3):
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "caller",
                    "text": random.choice(self.vocab.fraud_normal_gaps),
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "recipient",
                    "text": "Understood, please proceed.",
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1
            # Late burst: Urgency + Channel Control
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.fraud_urgency_turns),
                "labels": ["urgency"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": random.choice(self.vocab.fraud_recipient_responses),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "caller",
                "text": random.choice(self.vocab.fraud_channel_turns),
                "labels": ["channel_control"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1
            turns.append({
                "conversation_id": conv_id,
                "turn_id": turn_idx,
                "speaker": "recipient",
                "text": random.choice(self.vocab.fraud_recipient_responses),
                "labels": ["none"],
                "synthetic": True,
                "source_type": "synthetic",
                "source": topic
            })
            turn_idx += 1

        else: # diversified_sequence
            # Procedural combinatorial progression
            tactic_pool = [
                ("emotional_destabilization", self.vocab.fraud_emotional_turns),
                ("urgency", self.vocab.fraud_urgency_turns),
                ("authority", self.vocab.fraud_authority_turns),
                ("isolation", self.vocab.fraud_isolation_turns),
                ("channel_control", self.vocab.fraud_channel_turns)
            ]
            sampled_tactics = random.sample(tactic_pool, k=random.choice([3, 4]))
            for tactic, phrases in sampled_tactics:
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "caller",
                    "text": random.choice(phrases),
                    "labels": [tactic],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1
                turns.append({
                    "conversation_id": conv_id,
                    "turn_id": turn_idx,
                    "speaker": "recipient",
                    "text": random.choice(self.vocab.fraud_recipient_responses),
                    "labels": ["none"],
                    "synthetic": True,
                    "source_type": "synthetic",
                    "source": topic
                })
                turn_idx += 1

        return {
            "conversation_id": conv_id,
            "conversation_level_label": "fraud",
            "topic": topic,
            "archetype": archetype,
            "turns": turns
        }

    def generate_repaired_dataset(self, total_count: int = 2000, out_file: str = "data/raw/synthetic_conversations.jsonl"):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
        num_fraud = total_count // 2
        num_benign = total_count // 2
        
        conversations = []
        
        # -------------------------------------------------------------
        # 1. GENERATE BENIGN CONVERSATIONS (N = 1,000)
        # Quotas:
        # - >= 30% with >= 1 tactic-bearing turn
        # - >= 15% with >= 2 tactic-bearing turns
        # - >= 5% with >= 3 tactic-bearing turns
        # - >= 40% with legitimate tactic-like language
        # - >= 20% with combinations of 2+ tactic-like categories
        # Archetype Distribution:
        # - routine_service: 35% (350)
        # - authority_with_verification: 12% (120)
        # - urgency_with_safety: 12% (120)
        # - privacy_confidentiality: 8% (80)
        # - channel_guidance: 8% (80)
        # - multi_tactic_benign: 22% (220)
        # - suspicious_looking_benign: 3% (30)
        # -------------------------------------------------------------
        benign_archetypes = (
            ["routine_service"] * int(num_benign * 0.35) +
            ["authority_with_verification"] * int(num_benign * 0.12) +
            ["urgency_with_safety"] * int(num_benign * 0.12) +
            ["privacy_confidentiality"] * int(num_benign * 0.08) +
            ["channel_guidance"] * int(num_benign * 0.08) +
            ["multi_tactic_benign"] * int(num_benign * 0.22) +
            ["suspicious_looking_benign"] * int(num_benign * 0.03)
        )
        # Adjust remainder if any
        while len(benign_archetypes) < num_benign:
            benign_archetypes.append("routine_service")
            
        for idx, arch in enumerate(benign_archetypes):
            cid = f"conv_benign_{idx+1:04d}_{uuid.uuid4().hex[:6]}"
            conv = self.generate_benign_conversation(cid, arch)
            conversations.append(conv)
            
        # -------------------------------------------------------------
        # 2. GENERATE FRAUD CONVERSATIONS (N = 1,000)
        # Quotas:
        # - 100% contain manipulation overall
        # - >= 20% begin with 2+ normal turns before manipulation (delayed_escalation)
        # - >= 30% contain >= 2 normal turns between tactic turns (multi_gap_progression)
        # - >= 20% delayed manipulation (delayed_escalation & late_tactic_burst)
        # - >= 10% no tactic in first 3 turns (delayed_escalation & late_tactic_burst)
        # - >= 10% late tactic burst (late_tactic_burst)
        # Archetype Distribution:
        # - immediate_coercion: 25% (250)
        # - delayed_escalation: 25% (250)
        # - multi_gap_progression: 30% (300)
        # - late_tactic_burst: 10% (100)
        # - diversified_sequence: 10% (100)
        # -------------------------------------------------------------
        fraud_archetypes = (
            ["immediate_coercion"] * int(num_fraud * 0.25) +
            ["delayed_escalation"] * int(num_fraud * 0.25) +
            ["multi_gap_progression"] * int(num_fraud * 0.30) +
            ["late_tactic_burst"] * int(num_fraud * 0.10) +
            ["diversified_sequence"] * int(num_fraud * 0.10)
        )
        while len(fraud_archetypes) < num_fraud:
            fraud_archetypes.append("immediate_coercion")
            
        for idx, arch in enumerate(fraud_archetypes):
            cid = f"conv_fraud_{idx+1:04d}_{uuid.uuid4().hex[:6]}"
            conv = self.generate_fraud_conversation(cid, arch)
            conversations.append(conv)
            
        random.shuffle(conversations)
        
        with open(out_file, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")
                
        print(f"Successfully generated {len(conversations)} conversations ({num_fraud} fraud, {num_benign} benign) to {out_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--out", type=str, default="data/raw/synthetic_conversations.jsonl")
    args = parser.parse_args()
    
    gen = ProceduralTemporalDatasetGenerator(seed=args.seed)
    gen.generate_repaired_dataset(total_count=args.count, out_file=args.out)
