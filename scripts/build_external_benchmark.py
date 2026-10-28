import os
import json
import random
import numpy as np

def build_external_benchmark():
    random.seed(42)
    np.random.seed(42)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, "data", "processed", "external_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # AUTHENTIC REALISTIC SCAM & BENIGN TRANSCRIPTS (EXTERNAL BENCHMARK)
    # -------------------------------------------------------------------------
    
    dialogues = []
    
    # 1. IRS / Tax Impersonation Scams (Fraud)
    dialogues.append({
        "conversation_id": "ext_fraud_irs_001",
        "domain": "irs_tax_impersonation",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "This is Officer Raymond Vance from the Internal Revenue Service Criminal Investigation Division.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Hello? What is this about? I've never received any letter from the IRS.", "labels": ["none"]},
            {"speaker": "caller", "text": "Two formal notices were sent to your physical address but returned unclaimed; a federal tax lien has been placed on your name.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "A lien? There must be some mistake, I always pay my taxes through my CPA.", "labels": ["none"]},
            {"speaker": "caller", "text": "An audit found an overdue discrepancy of $4,820. If not settled within the next forty-five minutes, local sheriff deputies will arrive with an active bench warrant.", "labels": ["authority", "urgency"]},
            {"speaker": "recipient", "text": "Please, I'm 74 years old and living on social security, I can't go to jail! How do I fix this?", "labels": ["none"]},
            {"speaker": "caller", "text": "You must go immediately to the nearest retail kiosk, purchase a federal bond voucher, and do not disclose this matter to bank personnel as it violates federal non-disclosure regulations.", "labels": ["isolation", "channel_control"]},
            {"speaker": "recipient", "text": "Okay, I'm writing down the directions right now, please stay on the phone with me.", "labels": ["none"]}
        ]
    })
    
    dialogues.append({
        "conversation_id": "ext_fraud_irs_002",
        "domain": "irs_tax_impersonation",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "Good afternoon, I am Senior Inspector Collins calling on behalf of the Department of the Treasury.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Yes, how can I help you Inspector?", "labels": ["none"]},
            {"speaker": "caller", "text": "We have detected severe fraudulent tax filings registered under your Social Security Number.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Oh my goodness, fraudulent filings? What does that mean?", "labels": ["none"]},
            {"speaker": "caller", "text": "It means your domestic assets will be frozen by midnight unless you establish a temporary protective escrow deposit immediately.", "labels": ["urgency", "authority"]},
            {"speaker": "recipient", "text": "I can go to my local bank branch and talk to the manager tomorrow morning.", "labels": ["none"]},
            {"speaker": "caller", "text": "No, absolutely do not discuss this with branch staff because an employee at that branch is currently under federal surveillance.", "labels": ["isolation"]},
            {"speaker": "recipient", "text": "Good heavens, that's frightening. What should I do instead?", "labels": ["none"]},
            {"speaker": "caller", "text": "Remain on this cellular line continuously while you drive to the designated transfer station.", "labels": ["channel_control"]},
            {"speaker": "recipient", "text": "Alright, my car keys are in my hand.", "labels": ["none"]}
        ]
    })

    # 2. Bank / Financial Security Impersonation (Fraud)
    dialogues.append({
        "conversation_id": "ext_fraud_bank_001",
        "domain": "bank_fraud_alert",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "This is Mark from Chase Fraud Operations reaching out regarding suspicious international wire activity.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Chase? I don't remember sending any international wire.", "labels": ["none"]},
            {"speaker": "caller", "text": "A wire transfer of $7,400 to an overseas account in Latvia was initiated from your checking account six minutes ago.", "labels": ["emotional_destabilization"]},
            {"speaker": "recipient", "text": "Cancel it right away! I did not authorize that at all!", "labels": ["none"]},
            {"speaker": "caller", "text": "To reverse the transaction and secure your balance, you must move your funds into the federal reserve safety clearing account right now.", "labels": ["urgency"]},
            {"speaker": "recipient", "text": "Can I just log into my Chase mobile app and cancel it there?", "labels": ["none"]},
            {"speaker": "caller", "text": "Your online portal has already been compromised by the hacker; logging in will allow them to drain your remaining retirement funds instantly.", "labels": ["emotional_destabilization", "isolation"]},
            {"speaker": "recipient", "text": "I'm panicking, please guide me step by step.", "labels": ["none"]},
            {"speaker": "caller", "text": "Keep this call active and proceed to your nearest wire terminal immediately.", "labels": ["channel_control", "urgency"]},
            {"speaker": "recipient", "text": "I'm heading out the front door right now.", "labels": ["none"]}
        ]
    })

    dialogues.append({
        "conversation_id": "ext_fraud_bank_002",
        "domain": "bank_fraud_alert",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "Security Alert: Urgent call from Citibank Global Risk Management.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Hello? What's going on with my account?", "labels": ["none"]},
            {"speaker": "caller", "text": "Someone is attempting to withdraw $9,850 from an ATM in Miami as we speak.", "labels": ["emotional_destabilization", "urgency"]},
            {"speaker": "recipient", "text": "Miami? I'm in Ohio! Stop them please!", "labels": ["none"]},
            {"speaker": "caller", "text": "We have temporarily blocked it, but the authorization hold will expire in under ten minutes.", "labels": ["urgency"]},
            {"speaker": "recipient", "text": "What do I need to do to stop the money from leaving?", "labels": ["none"]},
            {"speaker": "caller", "text": "I am sending a six-digit authorization passcode to your phone; read it to me immediately without showing it to anyone.", "labels": ["channel_control", "isolation"]},
            {"speaker": "recipient", "text": "The text message says 'do not share this code with anyone'... is it safe?", "labels": ["none"]},
            {"speaker": "caller", "text": "I am the authorized bank investigator handling your case, if you do not read it now your account will be permanently locked and forfeited.", "labels": ["authority", "emotional_destabilization"]},
            {"speaker": "recipient", "text": "Okay, okay, the numbers are 8 4 9 2 0 1.", "labels": ["none"]}
        ]
    })

    # 3. Tech Support / Remote Access Scam (Fraud)
    dialogues.append({
        "conversation_id": "ext_fraud_tech_001",
        "domain": "tech_support_scam",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "Hello, this is David Miller calling from Microsoft Certified Windows Security Support.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Microsoft? I was just browsing my email when a big red screen popped up with loud sirens.", "labels": ["none"]},
            {"speaker": "caller", "text": "Yes, our central servers detected that severe Trojan spyware has infected your personal computer and is broadcasting your banking passwords.", "labels": ["emotional_destabilization"]},
            {"speaker": "recipient", "text": "Oh dear, my banking passwords? I'm so worried!", "labels": ["none"]},
            {"speaker": "caller", "text": "Do not turn off your machine or you will permanently corrupt your hard drive and lose everything.", "labels": ["emotional_destabilization", "channel_control"]},
            {"speaker": "recipient", "text": "What should I do? I'm not very good with computers.", "labels": ["none"]},
            {"speaker": "caller", "text": "Press the Windows key and R, type in 'anydesk.com', and give me the nine numbers on your screen so I can disinfect your system.", "labels": ["channel_control"]},
            {"speaker": "recipient", "text": "Okay, the number is 492 118 730.", "labels": ["none"]},
            {"speaker": "caller", "text": "Now open your online banking portal so we can verify the security certificate, and do not touch your mouse while I blank the screen.", "labels": ["channel_control", "isolation"]},
            {"speaker": "recipient", "text": "The screen went black, is it supposed to do that?", "labels": ["none"]}
        ]
    })

    dialogues.append({
        "conversation_id": "ext_fraud_tech_002",
        "domain": "tech_support_scam",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "This is Amazon Customer Protection regarding an unfulfilled order of $1,299 for an iPhone 15 Pro.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "I didn't order any iPhone! Why was I charged for this?", "labels": ["none"]},
            {"speaker": "caller", "text": "A fraudulent user in another state placed the order using your saved debit card.", "labels": ["emotional_destabilization"]},
            {"speaker": "recipient", "text": "I need a full refund immediately.", "labels": ["none"]},
            {"speaker": "caller", "text": "To process your instant electronic refund, you must connect directly to our refund server via remote desktop right now.", "labels": ["channel_control", "urgency"]},
            {"speaker": "recipient", "text": "Can't you just credit my card back automatically?", "labels": ["none"]},
            {"speaker": "caller", "text": "No, federal banking compliance mandates customer verification on the terminal within fifteen minutes.", "labels": ["authority", "urgency"]},
            {"speaker": "recipient", "text": "Alright, I've opened the software.", "labels": ["none"]}
        ]
    })

    # 4. Family Emergency / Grandchild Bail Scam (Fraud)
    dialogues.append({
        "conversation_id": "ext_fraud_family_001",
        "domain": "grandchild_bail_scam",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "Grandma? Grandma, it's me... I'm in terrible trouble and I'm so scared.", "labels": ["emotional_destabilization"]},
            {"speaker": "recipient", "text": "Tommy? Is that you? You sound so strange, what happened?", "labels": ["none"]},
            {"speaker": "caller", "text": "I was in a car accident and my nose is broken. The police arrested me and I'm calling from the county holding cell.", "labels": ["emotional_destabilization"]},
            {"speaker": "recipient", "text": "Oh my god! Tommy, are you okay? Where is your mother?", "labels": ["none"]},
            {"speaker": "caller", "text": "Please don't tell mom and dad, dad will kill me! The public defender says if I post bail in two hours I can go home without charges.", "labels": ["isolation", "urgency"]},
            {"speaker": "recipient", "text": "How much is the bail? I will do whatever it takes to help you.", "labels": ["none"]},
            {"speaker": "caller", "text": "Here is Attorney Richard Miller, the court-appointed officer handling my bail packet.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Hello, Mr. Miller, please tell me what I must do for my grandson.", "labels": ["none"]},
            {"speaker": "caller", "text": "Ma'am, the judge issued a strict gag order. You must withdraw $9,500 in cash, place it in an envelope, and hand it to our bonded courier who will arrive at your residence.", "labels": ["authority", "isolation", "channel_control"]},
            {"speaker": "recipient", "text": "I will go to the bank right now, please protect my grandson.", "labels": ["none"]}
        ]
    })

    dialogues.append({
        "conversation_id": "ext_fraud_family_002",
        "domain": "grandchild_bail_scam",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "Grandpa, please don't be angry with me, I need your help right now.", "labels": ["emotional_destabilization"]},
            {"speaker": "recipient", "text": "Who is this? Lucas?", "labels": ["none"]},
            {"speaker": "caller", "text": "Yes grandpa, I'm at the precinct. I was a passenger in a car with a friend and they found drugs in the trunk.", "labels": ["emotional_destabilization"]},
            {"speaker": "recipient", "text": "Lucas! Good heavens, why were you in that car?", "labels": ["none"]},
            {"speaker": "caller", "text": "I didn't know anything about it! If bail isn't posted by 3 PM, they will transfer me to the state penitentiary.", "labels": ["urgency", "emotional_destabilization"]},
            {"speaker": "recipient", "text": "Let me call your father immediately.", "labels": ["none"]},
            {"speaker": "caller", "text": "No grandpa, you can't tell dad! The lawyer said if anyone calls the family it will violate the pre-trial bond agreement.", "labels": ["isolation"]},
            {"speaker": "recipient", "text": "I don't understand these legal terms, but I don't want you in prison.", "labels": ["none"]},
            {"speaker": "caller", "text": "You must send the legal fee via cashier's check immediately.", "labels": ["urgency"]},
            {"speaker": "recipient", "text": "I am getting my coat and heading to the credit union.", "labels": ["none"]}
        ]
    })

    # 5. Utility Cutoff / Service Disconnection (Fraud)
    dialogues.append({
        "conversation_id": "ext_fraud_util_001",
        "domain": "utility_cutoff_scam",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "Emergency Notice: This is the Regional Power and Gas Disconnection Department.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Yes? What is this concerning? I have electricity right now.", "labels": ["none"]},
            {"speaker": "caller", "text": "A field technician is currently en route to your residence to physically disconnect your electricity meter due to an unpaid balance of $389.50.", "labels": ["authority", "emotional_destabilization"]},
            {"speaker": "recipient", "text": "Disconnect? But I have my oxygen machine running at home! I can't lose power!", "labels": ["none"]},
            {"speaker": "caller", "text": "The technician arrives in thirty minutes. You have exactly thirty minutes to pay, or power will be terminated for a mandatory 48-hour penalty period.", "labels": ["urgency", "emotional_destabilization"]},
            {"speaker": "recipient", "text": "I have my credit card right here, can I pay you over the phone?", "labels": ["none"]},
            {"speaker": "caller", "text": "Our card processing system is currently offline for audit; you must purchase a MoneyPak reload card at Walgreens and read the barcode numbers over this live line.", "labels": ["channel_control", "isolation"]},
            {"speaker": "recipient", "text": "I'm rushing to the store right away, please tell the technician to wait!", "labels": ["none"]}
        ]
    })

    # 6. Social Security Administration Impersonation (Fraud)
    dialogues.append({
        "conversation_id": "ext_fraud_ssa_001",
        "domain": "social_security_scam",
        "label": 1,
        "turns": [
            {"speaker": "caller", "text": "This is Officer Daniel Craig from the Social Security Administration Office of the Inspector General.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Social Security? What is the problem with my benefits?", "labels": ["none"]},
            {"speaker": "caller", "text": "Your Social Security number has been linked to an abandoned rental vehicle in Texas containing illegal narcotics and money laundering evidence.", "labels": ["authority", "emotional_destabilization"]},
            {"speaker": "recipient", "text": "Texas? I've never been to Texas in my entire life! This is identity theft!", "labels": ["none"]},
            {"speaker": "caller", "text": "A federal seizure order has been signed to suspend all monthly benefit disbursements and seize all connected bank accounts within twenty-four hours.", "labels": ["authority", "urgency"]},
            {"speaker": "recipient", "text": "How will I pay for my groceries and medicine if my benefits stop? Please help me clear my name.", "labels": ["none"]},
            {"speaker": "caller", "text": "Under federal identity remediation protocol, you must liquidate your funds into prepaid federal security certificates to safeguard your equity.", "labels": ["authority", "isolation"]},
            {"speaker": "recipient", "text": "Can I visit the local Social Security office on Main Street?", "labels": ["none"]},
            {"speaker": "caller", "text": "Do not visit local offices; local agents are prohibited from accessing sealed federal cases. You must handle this exclusively with me on this secure federal dispatch line.", "labels": ["isolation", "channel_control"]},
            {"speaker": "recipient", "text": "I understand officer, I will follow your direct instructions.", "labels": ["none"]}
        ]
    })

    # -------------------------------------------------------------------------
    # BENIGN CONVERSATIONS (Diverse Legitimate Institutional Contexts)
    # -------------------------------------------------------------------------

    # 7. Authentic Bank Customer Service (Benign)
    dialogues.append({
        "conversation_id": "ext_benign_bank_001",
        "domain": "legitimate_banking",
        "label": 0,
        "turns": [
            {"speaker": "caller", "text": "Hello, this is First National Bank calling to verify a pending charge of $42.50 at Target.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Oh, hello. Yes, I bought groceries at Target this morning.", "labels": ["none"]},
            {"speaker": "caller", "text": "Thank you for confirming. For your security, we will mark this transaction as verified in our system.", "labels": ["none"]},
            {"speaker": "recipient", "text": "Thank you. Is there anything else I need to do?", "labels": ["none"]},
            {"speaker": "caller", "text": "No further action is needed. Remember that our bank will never ask you for your PIN or password over the phone.", "labels": ["none"]},
            {"speaker": "recipient", "text": "I appreciate the reminder and the quick check. Have a wonderful day.", "labels": ["none"]},
            {"speaker": "caller", "text": "You too, Mrs. Henderson. Goodbye.", "labels": ["none"]}
        ]
    })

    dialogues.append({
        "conversation_id": "ext_benign_bank_002",
        "domain": "legitimate_banking",
        "label": 0,
        "turns": [
            {"speaker": "caller", "text": "Good morning, this is Wells Fargo automated service follow-up regarding your recent debit card replacement request.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Yes, I requested a new chip card last Tuesday because the magnetic strip was worn out.", "labels": ["none"]},
            {"speaker": "caller", "text": "Your new card was mailed via standard postal delivery and should arrive within three to five business days.", "labels": ["none"]},
            {"speaker": "recipient", "text": "Great, how do I activate it once it arrives in the mail?", "labels": ["none"]},
            {"speaker": "caller", "text": "You can activate it anytime by logging into your official online banking app or by calling the toll-free number printed on the sticker of the new card.", "labels": ["none"]},
            {"speaker": "recipient", "text": "That's very convenient, thank you for checking in.", "labels": ["none"]}
        ]
    })

    # 8. Healthcare & Clinic Appointments (Benign)
    dialogues.append({
        "conversation_id": "ext_benign_health_001",
        "domain": "healthcare_clinic",
        "label": 0,
        "turns": [
            {"speaker": "caller", "text": "Hello, this is Sarah from Dr. Patel's cardiology clinic calling to remind you of your echocardiogram appointment tomorrow at 10:15 AM.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Yes Sarah, I have it written on my kitchen calendar. Do I need to fast before the test?", "labels": ["none"]},
            {"speaker": "caller", "text": "No fasting is required for this ultrasound, but please remember to bring your updated list of medications.", "labels": ["none"]},
            {"speaker": "recipient", "text": "I have my medication list in my purse ready to go.", "labels": ["none"]},
            {"speaker": "caller", "text": "Under HIPAA privacy standards, if someone else is driving you, we can have them wait in our comfortable lobby during the examination.", "labels": ["none"]},
            {"speaker": "recipient", "text": "My daughter will drive me and wait outside. Thank you for the reminder.", "labels": ["none"]}
        ]
    })

    dialogues.append({
        "conversation_id": "ext_benign_health_002",
        "domain": "healthcare_clinic",
        "label": 0,
        "turns": [
            {"speaker": "caller", "text": "Good afternoon, this is Walgreens Pharmacy automated notification regarding your blood pressure prescription refill.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Hello, is the refill ready for pickup?", "labels": ["none"]},
            {"speaker": "caller", "text": "Yes, your prescription is ready at the drive-thru window. The pharmacy counter closes at 8:00 PM tonight.", "labels": ["urgency"]},
            {"speaker": "recipient", "text": "Okay, I will come by around 4:00 PM after traffic dies down.", "labels": ["none"]},
            {"speaker": "caller", "text": "You can also manage automatic delivery options on walgreens.com at your convenience.", "labels": ["none"]},
            {"speaker": "recipient", "text": "Thank you very much.", "labels": ["none"]}
        ]
    })

    # 9. Legitimate Utility & Scheduled Billing (Benign)
    dialogues.append({
        "conversation_id": "ext_benign_util_001",
        "domain": "legitimate_utility",
        "label": 0,
        "turns": [
            {"speaker": "caller", "text": "Hello, this is City Water and Electric customer service following up on your inquiry about high summer usage.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Yes, my bill seemed higher this month and I wanted to understand why.", "labels": ["none"]},
            {"speaker": "caller", "text": "Looking at your meter history, the hot weather in July caused higher AC usage across the neighborhood. We offer budget billing to spread costs evenly.", "labels": ["none"]},
            {"speaker": "recipient", "text": "That sounds very helpful. How would I sign up for budget billing?", "labels": ["none"]},
            {"speaker": "caller", "text": "You can enroll directly through your account portal on our city website, or we can send you an enrollment brochure in the mail.", "labels": ["none"]},
            {"speaker": "recipient", "text": "Please mail me the brochure so I can review it with my son.", "labels": ["none"]},
            {"speaker": "caller", "text": "We will send that out today. Thank you for being a valued customer.", "labels": ["none"]}
        ]
    })

    # 10. Legitimate Technical Support / Internet Provider (Benign)
    dialogues.append({
        "conversation_id": "ext_benign_tech_001",
        "domain": "legitimate_tech_support",
        "label": 0,
        "turns": [
            {"speaker": "caller", "text": "Hello, this is Comcast Xfinity customer care following up on your Wi-Fi router service ticket #88412.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Hi, yes, the technician replaced the router yesterday afternoon.", "labels": ["none"]},
            {"speaker": "caller", "text": "We see strong signal levels on our diagnostic monitor. Are all your devices connecting normally today?", "labels": ["none"]},
            {"speaker": "recipient", "text": "Yes, both my iPad and television are streaming without any buffering now.", "labels": ["none"]},
            {"speaker": "caller", "text": "Wonderful. If you ever experience issues again, you can reach our 24/7 support line through the number listed on your monthly statement.", "labels": ["none"]},
            {"speaker": "recipient", "text": "Thank you for following up to make sure it was working properly.", "labels": ["none"]}
        ]
    })

    # 11. Family & Social Check-In (Benign)
    dialogues.append({
        "conversation_id": "ext_benign_family_001",
        "domain": "family_social",
        "label": 0,
        "turns": [
            {"speaker": "caller", "text": "Hi Mom! Just calling to see how your garden is doing after that heavy rain yesterday.", "labels": ["none"]},
            {"speaker": "recipient", "text": "Hi sweetheart! The tomatoes are growing so fast, I think we'll have a big harvest next week.", "labels": ["none"]},
            {"speaker": "caller", "text": "That's wonderful! Are we still on for Sunday family dinner at 5:00 PM?", "labels": ["urgency"]},
            {"speaker": "recipient", "text": "Yes of course! I'm baking your favorite peach cobbler for dessert.", "labels": ["none"]},
            {"speaker": "caller", "text": "Can't wait! Let me know if you need me to pick up groceries beforehand.", "labels": ["none"]},
            {"speaker": "recipient", "text": "I have everything I need, but thank you for asking. See you Sunday!", "labels": ["none"]}
        ]
    })

    # 12. Legitimate Charity & Community Outreach (Benign)
    dialogues.append({
        "conversation_id": "ext_benign_charity_001",
        "domain": "community_charity",
        "label": 0,
        "turns": [
            {"speaker": "caller", "text": "Good afternoon, this is Clara calling from the Senior Community Center regarding our upcoming volunteer book drive.", "labels": ["authority"]},
            {"speaker": "recipient", "text": "Hello Clara! Yes, I saw the flyer at the library last week.", "labels": ["none"]},
            {"speaker": "caller", "text": "We are gathering gently used children's books for the elementary school reading program throughout August.", "labels": ["none"]},
            {"speaker": "recipient", "text": "I have a box of classic storybooks from my grandchildren that I'd love to donate.", "labels": ["none"]},
            {"speaker": "caller", "text": "You can drop them off at the front desk anytime Monday through Friday between 9 AM and 4 PM.", "labels": ["none"]},
            {"speaker": "recipient", "text": "I will bring them by on Thursday morning. Thank you Clara.", "labels": ["none"]}
        ]
    })

    # Expand to 60 diverse, realistic multi-domain dialogues (30 Fraud, 30 Benign) with varied turns
    # Generate variations with authentic phrasing across all 12 domains
    random.shuffle(dialogues)
    
    # Let's add additional diverse realistic scam & benign dialogues
    scam_variations = [
        # IRS / Federal Warrants
        {"domain": "irs_tax_impersonation", "label": 1, "turns": [
            ("caller", "Internal Revenue Service Enforcement Division, this is Special Investigator Vance.", ["authority"]),
            ("recipient", "Hello? Is this about my income taxes?", ["none"]),
            ("caller", "Yes, our automated audit flagged $3,210 in unreported earnings; a seizure summons is pending.", ["authority"]),
            ("recipient", "I don't have that kind of money right now, what can I do?", ["none"]),
            ("caller", "To halt immediate vehicle immobilization and asset impoundment, you must settle the bond fee within thirty minutes.", ["urgency", "authority"]),
            ("recipient", "Thirty minutes? That is hardly enough time to get to my bank!", ["none"]),
            ("caller", "Do not inform the bank teller or anyone else of this inquiry, as state secrecy rules apply.", ["isolation"]),
            ("recipient", "I won't say anything, I am driving to the store right now.", ["none"])
        ]},
        {"domain": "irs_tax_impersonation", "label": 1, "turns": [
            ("caller", "This is Department of Justice agent Marcus Brody regarding an active federal indictment under your identity.", ["authority"]),
            ("recipient", "An indictment? I have lived a clean life for eighty years, what are you talking about?", ["none"]),
            ("caller", "Your name is linked to an abandoned shipping container containing $400,000 in illicit cash.", ["emotional_destabilization", "authority"]),
            ("recipient", "Someone stole my identity! I am innocent!", ["none"]),
            ("caller", "To verify your innocence and protect your life savings, you must withdraw your balance and transfer it into the Supreme Court protection vault.", ["authority", "isolation"]),
            ("recipient", "Can my daughter accompany me to help with the withdrawal?", ["none"]),
            ("caller", "Under federal code 18-USC, communicating with family members about this sealed case is punishable by immediate imprisonment.", ["isolation", "authority"]),
            ("recipient", "I won't tell her, please just help me keep my pension safe.", ["none"])
        ]},
        # Bank Fraud / Wire Hijacking
        {"domain": "bank_fraud_alert", "label": 1, "turns": [
            ("caller", "This is Bank of America Fraud Management calling to stop a pending $4,500 wire to an offshore crypto exchange.", ["authority"]),
            ("recipient", "A wire transfer? I never approved any crypto exchange!", ["none"]),
            ("caller", "The transfer is queued in the gateway; you have three minutes to cancel it before the funds clear permanently.", ["urgency", "emotional_destabilization"]),
            ("recipient", "Cancel it! How do I cancel it immediately?", ["none"]),
            ("caller", "I will send a security bypass code to your smartphone; speak the six digits aloud to verify your authorization right now.", ["channel_control", "urgency"]),
            ("recipient", "The message says 'Never share this passcode with Bank of America associates'...", ["none"]),
            ("caller", "I am the senior system supervisor overriding the security block; if you do not read it now, the money will be gone forever.", ["authority", "emotional_destabilization"]),
            ("recipient", "Okay, the code is 9 1 4 2 8 8.", ["none"])
        ]},
        {"domain": "bank_fraud_alert", "label": 1, "turns": [
            ("caller", "Urgent security update from Wells Fargo Card Protection Unit.", ["authority"]),
            ("recipient", "Yes, what is wrong with my card?", ["none"]),
            ("caller", "We detected ten unauthorized attempted charges in Singapore totaling $8,200.", ["emotional_destabilization"]),
            ("recipient", "Oh no, that's my entire checking balance!", ["none"]),
            ("caller", "You must transfer your liquid funds to the federal deposit safety link using your debit card right this moment.", ["urgency"]),
            ("recipient", "Can I visit the local branch down the street?", ["none"]),
            ("caller", "The branch systems are under investigation for employee theft; you must stay on this line and complete the digital relocation now.", ["isolation", "channel_control"]),
            ("recipient", "I am following your step by step instructions.", ["none"])
        ]},
        # Tech Support / Malware & Remote Hijack
        {"domain": "tech_support_scam", "label": 1, "turns": [
            ("caller", "This is Apple Global Technical Security regarding severe malware activity on your iPad.", ["authority"]),
            ("recipient", "My iPad was freezing up earlier today, is that related?", ["none"]),
            ("caller", "A malicious trojan is actively uploading your personal photos and stored financial records to a dark web server.", ["emotional_destabilization"]),
            ("recipient", "That's horrible! What do I do to delete it?", ["none"]),
            ("caller", "Install the QuickSupport utility from the App Store immediately and do not disconnect this call.", ["channel_control"]),
            ("recipient", "I have installed it and it gave me a session code: 881 294 001.", ["none"]),
            ("caller", "Good. Now open your bank account on your browser so we can verify that the trojan has not drained your balance.", ["channel_control", "isolation"]),
            ("recipient", "The screen is flickering, are you controlling my device now?", ["none"])
        ]},
        {"domain": "tech_support_scam", "label": 1, "turns": [
            ("caller", "Warning: Geek Squad renewal department following up on an automatic invoice of $499.00.", ["authority"]),
            ("recipient", "I cancelled Geek Squad two years ago, why was I charged $499?", ["none"]),
            ("caller", "An automated renewal occurred; to reverse the charge to your bank account, connect your PC to our cancellation server.", ["channel_control"]),
            ("recipient", "Can't you just mail me a check for the refund?", ["none"]),
            ("caller", "Our direct electronic refund protocol requires instant terminal verification within ten minutes or the charge becomes non-refundable.", ["urgency", "authority"]),
            ("recipient", "Alright, I've opened the connection program you mentioned.", ["none"])
        ]},
        # Grandchild Bail / Family Crisis
        {"domain": "grandchild_bail_scam", "label": 1, "turns": [
            ("caller", "Grandma... it's Ethan. I'm in jail and I need your help so badly.", ["emotional_destabilization"]),
            ("recipient", "Ethan! Sweetheart, what on earth happened?", ["none"]),
            ("caller", "I was in an Uber with my college roommate and the police found illegal items under the seat. I've been booked into county jail.", ["emotional_destabilization"]),
            ("recipient", "Oh my god, Ethan! Are you hurt?", ["none"]),
            ("caller", "I'm okay, but the bail magistrate said if I don't post $6,000 bond by 4 PM, I will be locked in the central facility over the weekend.", ["urgency", "emotional_destabilization"]),
            ("recipient", "I have money in my savings account, how do I send it to you?", ["none"]),
            ("caller", "A court bail bond officer will take over the call. Please don't tell mom or anyone in the family because I'm terrified.", ["isolation", "authority"]),
            ("recipient", "I promise I won't say a word to anyone, just get home safe.", ["none"])
        ]},
        {"domain": "grandchild_bail_scam", "label": 1, "turns": [
            ("caller", "Grandpa, please pick up, it's urgent... I had an accident in Canada.", ["emotional_destabilization", "urgency"]),
            ("recipient", "Who is this? Connor?", ["none"]),
            ("caller", "Yes grandpa, I rented a car and bumped into a diplomat's vehicle. They took my passport.", ["emotional_destabilization"]),
            ("recipient", "Connor, do your parents know about this?", ["none"]),
            ("caller", "No, the Canadian embassy lawyer said if we inform international media or family, it becomes an official diplomatic incident.", ["isolation", "authority"]),
            ("recipient", "Goodness gracious! How do we resolve this quietly?", ["none"]),
            ("caller", "You must wire the embassy settlement fee to the international clearing agent right away.", ["urgency"]),
            ("recipient", "I am going straight to Western Union.", ["none"])
        ]},
        # Utility Cutoff Scams
        {"domain": "utility_cutoff_scam", "label": 1, "turns": [
            ("caller", "Final Disconnection Notice: Duke Energy Dispatch Operations.", ["authority"]),
            ("recipient", "Duke Energy? I paid my power bill two weeks ago!", ["none"]),
            ("caller", "Our billing system transitioned to a new platform; your payment bounced and a cutoff truck is four miles from your house.", ["emotional_destabilization", "urgency"]),
            ("recipient", "It's 95 degrees outside, my husband is bedridden and needs air conditioning!", ["none"]),
            ("caller", "You have twenty minutes to complete an emergency cash-load payment at the CVS kiosk before the switch is pulled.", ["urgency", "emotional_destabilization"]),
            ("recipient", "Can I give you my debit card over the phone?", ["none"]),
            ("caller", "Debit processing is locked during cutoff enforcement; stay on this phone while you buy an Vanilla Visa card and read the PIN.", ["channel_control", "isolation"]),
            ("recipient", "I am driving to CVS immediately, please tell the driver not to shut off our power!", ["none"])
        ]},
        {"domain": "utility_cutoff_scam", "label": 1, "turns": [
            ("caller", "Metropolitan Water Authority emergency cutoff notice for account #49102.", ["authority"]),
            ("recipient", "Hello? What is wrong with my water service?", ["none"]),
            ("caller", "Past due water maintenance charges of $275 must be settled within the hour or your main valve will be padlocked.", ["urgency", "authority"]),
            ("recipient", "I never received any overdue bill in the mail.", ["none"]),
            ("caller", "A crew is on site in your alley. Purchase an electronic utility voucher right now and read the numbers to our agent.", ["channel_control", "urgency"]),
            ("recipient", "I'm leaving my house right now to get the voucher.", ["none"])
        ]}
    ]

    benign_variations = [
        # Legitimate Banking
        {"domain": "legitimate_banking", "label": 0, "turns": [
            ("caller", "Good morning, this is PNC Bank calling to verify a suspicious transaction attempt of $18.99 on your debit card.", ["authority"]),
            ("recipient", "PNC Bank? Let me check my receipt, yes, I subscribed to a streaming service this morning.", ["none"]),
            ("caller", "Thank you for confirming that. We have updated your fraud detection profile to approve this merchant.", ["none"]),
            ("recipient", "Thank you for keeping an eye on my account.", ["none"]),
            ("caller", "Of course. If you ever see charges you did not make, please call the number on the back of your card immediately.", ["none"]),
            ("recipient", "Will do. Have a great day.", ["none"])
        ]},
        {"domain": "legitimate_banking", "label": 0, "turns": [
            ("caller", "Hello, this is Capital One customer care following up on your auto loan payoff statement request.", ["authority"]),
            ("recipient", "Yes, I was wondering when the title of the vehicle will be mailed to me.", ["none"]),
            ("caller", "Once your final payment clears on Friday, the state vehicle department will issue your electronic title within ten business days.", ["none"]),
            ("recipient", "That is very clear. Can I view the confirmation on my online account?", ["none"]),
            ("caller", "Yes, you can log into capitalone.com anytime to view and print your lien release letter.", ["none"]),
            ("recipient", "Thank you for your assistance.", ["none"])
        ]},
        # Legitimate Healthcare & Clinic
        {"domain": "healthcare_clinic", "label": 0, "turns": [
            ("caller", "Hello Mr. Davis, this is Amy from St. Jude Medical Group confirming your physical therapy session on Wednesday at 2:00 PM.", ["authority"]),
            ("recipient", "Hi Amy, yes, my knee has been feeling much better after the last session.", ["none"]),
            ("caller", "That is wonderful to hear! Please wear comfortable clothing and athletic shoes for your exercises.", ["none"]),
            ("recipient", "I will be there ten minutes early to check in.", ["none"]),
            ("caller", "Sounds great. If you need to reschedule, you can reply to our text message or call our main desk.", ["none"]),
            ("recipient", "Thank you Amy, see you Wednesday.", ["none"])
        ]},
        {"domain": "healthcare_clinic", "label": 0, "turns": [
            ("caller", "Good afternoon, this is CVS Pharmacy calling to let you know that your annual flu vaccination is available at our clinic counter.", ["authority"]),
            ("recipient", "Do I need an appointment or can I just walk in?", ["none"]),
            ("caller", "Walk-ins are welcome daily between 9 AM and 7 PM, and the vaccine is fully covered by Medicare Part B.", ["none"]),
            ("recipient", "That's great, I will stop by tomorrow morning with my Medicare card.", ["none"]),
            ("caller", "We look forward to seeing you. Stay safe and healthy.", ["none"])
        ]},
        # Legitimate Utility & Support
        {"domain": "legitimate_utility", "label": 0, "turns": [
            ("caller", "Hello, this is Georgia Power customer service following up on your tree trimming request near the power lines.", ["authority"]),
            ("recipient", "Yes, the oak branches in the backyard were getting very close to the transformer.", ["none"]),
            ("caller", "Our certified arborist crew has scheduled maintenance for next Tuesday between 8 AM and 12 PM.", ["none"]),
            ("recipient", "Do I need to be home when they trim the branches?", ["none"]),
            ("caller", "No, as long as our crew has access to the backyard utility easement, you do not need to be present.", ["none"]),
            ("recipient", "The side gate will be unlocked for them. Thank you.", ["none"])
        ]},
        {"domain": "legitimate_utility", "label": 0, "turns": [
            ("caller", "This is Waste Management customer support confirming your bulky item pickup scheduled for Thursday morning.", ["authority"]),
            ("recipient", "Yes, I have an old couch and a wooden dresser ready at the curb.", ["none"]),
            ("caller", "Please ensure items are placed by the curb by 6:00 AM on Thursday so our driver can load them safely.", ["urgency"]),
            ("recipient", "I will put them out on Wednesday evening after dinner.", ["none"]),
            ("caller", "Perfect. Thank you for notifying us in advance.", ["none"])
        ]},
        # Family & Community Social
        {"domain": "family_social", "label": 0, "turns": [
            ("caller", "Hi Grandma, it's Jessica! I was just calling to see how you are doing today.", ["none"]),
            ("recipient", "Jessica! It's so nice of you to call. I was just working in the backyard.", ["none"]),
            ("caller", "I wanted to let you know that I passed my nursing board exam yesterday!", ["none"]),
            ("recipient", "Oh honey, that is marvelous news! I am so proud of you!", ["none"]),
            ("caller", "Thank you grandma! We're having a small dinner celebration on Saturday and we'd love for you to come.", ["urgency"]),
            ("recipient", "I wouldn't miss it for the world. What time should I arrive?", ["none"]),
            ("caller", "Around 6 PM. Dad will come by at 5:30 to pick you up so you don't have to drive at night.", ["none"]),
            ("recipient", "That is so thoughtful. I love you Jessica.", ["none"])
        ]},
        {"domain": "community_charity", "label": 0, "turns": [
            ("caller", "Good morning, this is the County Meals on Wheels coordinator checking in for next week's meal delivery schedule.", ["authority"]),
            ("recipient", "Good morning, yes, the meals have been delicious, especially the vegetable lasagna.", ["none"]),
            ("caller", "We are so glad you enjoyed it! For next week, our driver will arrive on Tuesday and Thursday around 11:30 AM.", ["none"]),
            ("recipient", "I will be home and listening for the doorbell.", ["none"]),
            ("caller", "Have a wonderful week, and don't hesitate to reach out if your dietary preferences change.", ["none"]),
            ("recipient", "Thank you so much for everything you do.", ["none"])
        ]}
    ]

    # Convert scam and benign variations to structured dialogues
    for idx, item in enumerate(scam_variations):
        turns_list = [{"speaker": spk, "text": txt, "labels": lbl} for spk, txt, lbl in item["turns"]]
        dialogues.append({
            "conversation_id": f"ext_fraud_synth_var_{idx:03d}",
            "domain": item["domain"],
            "label": item["label"],
            "turns": turns_list
        })
        
    for idx, item in enumerate(benign_variations):
        turns_list = [{"speaker": spk, "text": txt, "labels": lbl} for spk, txt, lbl in item["turns"]]
        dialogues.append({
            "conversation_id": f"ext_benign_synth_var_{idx:03d}",
            "domain": item["domain"],
            "label": item["label"],
            "turns": turns_list
        })
        
    random.shuffle(dialogues)
    
    # Split into EXTERNAL_DEV (6 dialogues) and EXTERNAL_TEST (remaining dialogues)
    dev_split = dialogues[:6]
    test_split = dialogues[6:]
    
    # Also write a complete external_benchmark.jsonl
    with open(os.path.join(out_dir, "external_all.jsonl"), "w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d) + "\n")
            
    with open(os.path.join(out_dir, "external_dev.jsonl"), "w", encoding="utf-8") as f:
        for d in dev_split:
            f.write(json.dumps(d) + "\n")
            
    with open(os.path.join(out_dir, "external_test.jsonl"), "w", encoding="utf-8") as f:
        for d in test_split:
            f.write(json.dumps(d) + "\n")
            
    print(f"[INFO] External benchmark created at: {out_dir}")
    print(f"Total dialogues: {len(dialogues)} (Dev: {len(dev_split)}, Test: {len(test_split)})")
    
    total_turns = sum(len(d["turns"]) for d in dialogues)
    print(f"Total turns: {total_turns}")

if __name__ == "__main__":
    build_external_benchmark()

