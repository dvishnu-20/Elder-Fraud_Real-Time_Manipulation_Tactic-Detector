import json
import os
import random
import itertools

os.makedirs("data", exist_ok=True)

TACTIC_LABELS = [
    "urgency",
    "isolation",
    "authority",
    "emotional_destabilization",
    "channel_control"
]

def generate_comprehensive_challenge_set():
    random.seed(42)
    samples = []
    seen_texts = set()

    def add(text, labels, category, difficulty="standard", topic="general", tone="standard", pair_id=None, is_benign_pair_member=None):
        text = text.strip()
        if text in seen_texts:
            return
        seen_texts.add(text)
        for l in labels:
            if l != "none" and l not in TACTIC_LABELS:
                raise ValueError(f"Invalid label: {l}")
        
        sample = {
            "text": text,
            "labels": labels,
            "category": category,
            "difficulty": difficulty,
            "topic": topic,
            "tone": tone
        }
        
        if pair_id is not None:
            sample["pair_id"] = pair_id
            sample["is_benign_pair_member"] = is_benign_pair_member
            
        samples.append(sample)

    # =========================================================================
    # 1. RIGOROUS CONTRASTIVE PAIRS (55 pairs = 110 turns)
    # Distinct intent/behavior between benign context vs manipulative coercion
    # =========================================================================
    contrastive_pairs_data = [
        # --- URGENCY PAIRS (10) ---
        ("urgency", "The bank closes at 4 PM, so please submit the transfer before then.", ["none"],
         "You have until 4 PM to send the transfer, and you must do it right now without telling anyone.", ["urgency", "isolation"]),
        ("urgency", "Your prescription expires in 24 hours, so please pick it up from the pharmacy soon.", ["none"],
         "If you don't pick up and pay for this prescription in the next 24 hours, all your health benefits will be terminated.", ["urgency", "emotional_destabilization"]),
        ("urgency", "We recommend paying the utility invoice by Friday to avoid late fees.", ["none"],
         "Pay this invoice within the next fifteen minutes or law enforcement will be dispatched to your door.", ["urgency", "authority"]),
        ("urgency", "Your flight to Chicago boards in 15 minutes at Gate 12.", ["none"],
         "You must wire the money for the flight in 15 minutes or you'll be stranded and arrested at the terminal.", ["urgency", "emotional_destabilization"]),
        ("urgency", "The grocery store discount on gift cards ends tonight at 9 PM.", ["none"],
         "Go to the grocery store and purchase the gift cards before tonight or your arrest warrant will be activated.", ["urgency", "authority"]),
        ("urgency", "Please review the lease agreement and sign it by the end of the day.", ["none"],
         "Sign the document immediately before the end of the day, or your bank accounts will be frozen permanently.", ["urgency", "emotional_destabilization"]),
        ("urgency", "The municipal water company needs to inspect your meter tomorrow morning.", ["none"],
         "If you don't allow our water inspector inside within ten minutes, your supply will be permanently disconnected.", ["urgency"]),
        ("urgency", "The courier will drop off your package by 3 PM this afternoon.", ["none"],
         "The courier arrives at 3 PM; hand over the cash immediately or face immediate prosecution.", ["urgency", "authority"]),
        ("urgency", "Your streaming subscription will automatically renew next month unless canceled.", ["none"],
         "Your account is being charged thousands of dollars right now, act this second to reverse the charges!", ["urgency"]),
        ("urgency", "The post office closes in thirty minutes, let's mail the package today.", ["none"],
         "You have thirty minutes to mail the parcel before the cartel finds out where you live.", ["urgency", "emotional_destabilization"]),

        # --- ISOLATION PAIRS (10) ---
        ("isolation", "For patient privacy under HIPAA rules, please take this medical consultation in a quiet room.", ["none"],
         "To prevent the authorities from finding out, you must take this call in a quiet room and tell no one in your household.", ["isolation", "authority"]),
        ("isolation", "It's a surprise birthday party for Grandma, so please keep it secret from her.", ["none"],
         "This is an official covert investigation into your grandmother, and you are legally forbidden from telling her.", ["isolation", "authority"]),
        ("isolation", "Please enter your debit card PIN privately on the keypad so no one sees it.", ["none"],
         "Do not let the bank teller see you making this withdrawal; they are suspected of being part of the crime ring.", ["isolation"]),
        ("isolation", "Let's review your retirement savings in my private office so we aren't disturbed.", ["none"],
         "We must move your retirement funds alone because your children are plotting to take your estate.", ["isolation", "emotional_destabilization"]),
        ("isolation", "I prefer to keep my salary negotiations confidential from other employees.", ["none"],
         "You are strictly prohibited under federal gag orders from mentioning this transaction to anyone at your workplace.", ["isolation", "authority"]),
        ("isolation", "Please read this standard contract in the waiting room while we prepare your paperwork.", ["none"],
         "Read this contract alone and do not let your lawyer or family inspect it, or the settlement is void.", ["isolation"]),
        ("isolation", "It is best practice for HR to handle employee disputes discreetly without office gossip.", ["none"],
         "If you speak to anyone else about this dispute, you will be fired on the spot and blacklisted across the industry.", ["isolation", "emotional_destabilization"]),
        ("isolation", "I'm going for a short walk in the park to clear my head and get some fresh air.", ["none"],
         "Walk to your car alone right now so nobody in the house can overhear what I'm commanding you to do.", ["isolation", "channel_control"]),
        ("isolation", "Keep the ballot booth curtain closed so your vote remains completely private.", ["none"],
         "Do not speak to the election workers or bank staff, they are monitoring you illegally.", ["isolation"]),
        ("isolation", "We should discuss this private family decision together over dinner tonight.", ["none"],
         "Do not inform any other family members about this wire transfer; they cannot know.", ["isolation"]),

        # --- AUTHORITY PAIRS (10) ---
        ("authority", "I am Officer Davis calling to let you know we found your lost wallet at the bus stop.", ["none"],
         "I am Officer Davis and there is an outstanding federal arrest warrant in your name that requires immediate bail.", ["authority", "emotional_destabilization"]),
        ("authority", "This is the IRS sending your routine annual tax refund notification in the mail.", ["none"],
         "This is the IRS Special Enforcement Division demanding immediate payment or federal marshals will detain you.", ["authority", "urgency", "emotional_destabilization"]),
        ("authority", "I am the branch bank manager, returning your call regarding current mortgage interest rates.", ["none"],
         "I am the chief bank auditor, and I am seizing your account assets immediately due to suspicious criminal activity.", ["authority", "emotional_destabilization"]),
        ("authority", "This is the IT tech support desk, following up on your printer ticket from yesterday.", ["none"],
         "This is the Chief Security Officer at Microsoft; your computer has been flagged for illegal illicit materials.", ["authority"]),
        ("authority", "I am Judge Smith's court clerk, calling to confirm your rescheduled jury duty date.", ["none"],
         "I am Judge Smith, and you are in contempt of federal court for failing to appear.", ["authority"]),
        ("authority", "This is the neighborhood pharmacy letting you know your blood pressure medicine is ready.", ["none"],
         "This is the Drug Enforcement Administration; we are confiscating your pharmacy orders as evidence.", ["authority"]),
        ("authority", "Hello, this is customer service from Amazon. How may I help you with your order today?", ["none"],
         "This is the Amazon Fraud Taskforce; your identity has been flagged for money laundering.", ["authority"]),
        ("authority", "I'm the new regional director calling to introduce myself to the regional staff.", ["none"],
         "I'm the senior regional director, and my orders are absolute: wire the balance or be terminated immediately.", ["authority", "emotional_destabilization"]),
        ("authority", "This is the border customs office confirming that your international travel visa was approved.", ["none"],
         "This is the border patrol taskforce; we have seized a contraband shipment bearing your identity.", ["authority", "emotional_destabilization"]),
        ("authority", "I'm a city building inspector checking the street for code compliance.", ["none"],
         "I'm a federal code inspector, and I am condemning your property and ordering an immediate eviction today.", ["authority", "urgency"]),

        # --- EMOTIONAL DESTABILIZATION PAIRS (10) ---
        ("emotional_destabilization", "I was so heartbroken when my childhood pet passed away last year.", ["none"],
         "If you don't pay this veterinary bill in ten minutes, your pet will be put down immediately.", ["emotional_destabilization", "urgency"]),
        ("emotional_destabilization", "I'm feeling a bit anxious about the upcoming college final exams.", ["none"],
         "If you fail this verification, you will be expelled, sued for fraud, and ruin your life forever.", ["emotional_destabilization"]),
        ("emotional_destabilization", "Your grandson got a speeding ticket and had to pay a fifty dollar fine.", ["none"],
         "Your grandson was in a catastrophic accident and is weeping blood in a holding cell waiting for your bail!", ["emotional_destabilization"]),
        ("emotional_destabilization", "The financial markets experienced a slight dip during yesterday's trading session.", ["none"],
         "The financial system has collapsed and all your life savings have been wiped out forever!", ["emotional_destabilization"]),
        ("emotional_destabilization", "I accidentally scratched the fender of the rental vehicle in the parking garage.", ["none"],
         "You totaled our commercial vehicle and you owe forty thousand dollars right now or we take your home!", ["emotional_destabilization", "authority"]),
        ("emotional_destabilization", "There is a standard security update available for your online banking profile.", ["none"],
         "Criminal hackers have penetrated your life, stolen your identity, and are draining every cent you own!", ["emotional_destabilization", "urgency"]),
        ("emotional_destabilization", "I get a little nervous whenever severe thunderstorms are in the forecast.", ["none"],
         "A catastrophic tornado is directly targeting your exact address; you will perish if you don't act!", ["emotional_destabilization"]),
        ("emotional_destabilization", "The municipal property tax assessment went up slightly for our neighborhood.", ["none"],
         "The county is repossessing your house today and throwing you out onto the street with nothing!", ["emotional_destabilization", "urgency"]),
        ("emotional_destabilization", "I heard a news report about a burglary three towns over.", ["none"],
         "A violent syndicate knows where you sleep and will arrive at your home tonight if you don't wire the ransom.", ["emotional_destabilization", "urgency"]),
        ("emotional_destabilization", "Your annual Social Security earnings overview arrived in your postal mailbox.", ["none"],
         "Your Social Security number was found at a narcotics crime scene, making you a prime federal suspect.", ["emotional_destabilization"]),

        # --- CHANNEL CONTROL PAIRS (10) ---
        ("channel_control", "Please stay on the line for a moment while I transfer your call to the billing department.", ["none"],
         "Do not hang up this telephone line under any circumstances, or police cruisers will surround your house.", ["channel_control", "authority"]),
        ("channel_control", "Could you share your screen during our Zoom team meeting to show the presentation slides?", ["none"],
         "You must install AnyDesk remote software right now and give me full master access to your desktop.", ["channel_control"]),
        ("channel_control", "Keep your phone on speaker while we prepare dinner together so we can chat.", ["none"],
         "Keep this phone call on speaker in your front pocket while you enter the bank so I can listen to every word.", ["channel_control", "isolation"]),
        ("channel_control", "Please leave the browser tab open while your photo upload completes.", ["none"],
         "Do not close this browser window or touch your keyboard, or your computer system will be destroyed.", ["channel_control", "emotional_destabilization"]),
        ("channel_control", "Hold on for a second while I retrieve a pen from my desk drawer.", ["none"],
         "Maintain this phone connection continuously; if you disconnect for one second, the deal is terminated.", ["channel_control"]),
        ("channel_control", "I'll stay on the phone with you until you locate your misplaced car keys.", ["none"],
         "I am remaining on this line while you drive to the Bitcoin kiosk; keep the receiver active.", ["channel_control"]),
        ("channel_control", "Open your web browser and search for that new soup recipe we talked about.", ["none"],
         "Open your browser immediately, go to the URL I dictate, and install the remote control application.", ["channel_control"]),
        ("channel_control", "Let's turn on video mode so we can say hi to the grandchildren.", ["none"],
         "Activate your camera right now so I can monitor your physical movements during this transfer.", ["channel_control"]),
        ("channel_control", "Leave the backup application running in the background while your files sync.", ["none"],
         "Leave the remote desktop access tool running in the background and do not touch your mouse or keyboard.", ["channel_control"]),
        ("channel_control", "I'll send you a text message with the restaurant address while we're talking.", ["none"],
         "I just sent an authentication link; click and approve it right now while we are on the line.", ["channel_control", "urgency"]),

        # --- MULTI-LABEL PAIRS (5) ---
        ("multi_label", "The bank needs you to come to the local branch to update your mailing address with your ID.", ["none"],
         "This is federal law enforcement: do not inform your family, stay on this call, and drive to the bank to wire the bail right now or face arrest.", ["authority", "isolation", "channel_control", "urgency", "emotional_destabilization"]),
        ("multi_label", "Your tax return has a minor typographical error; please call the IRS support line to adjust it.", ["none"],
         "The IRS has issued an arrest warrant: you have fifteen minutes to stay on this line and purchase gift card vouchers or deputies will arrive.", ["authority", "urgency", "channel_control", "emotional_destabilization"]),
        ("multi_label", "Please keep this medical diagnosis private and consult with Dr. Patel tomorrow morning.", ["none"],
         "This is the state medical board: you are under formal investigation, do not speak to anyone, and wire the bond right now or lose your license.", ["authority", "isolation", "urgency"]),
        ("multi_label", "My computer is running slowly; I'll ask the local computer technician to look at it this weekend.", ["none"],
         "I am Microsoft Headquarters: do not turn off your PC, give me remote control of your screen immediately, or your files will be erased.", ["authority", "channel_control", "urgency", "emotional_destabilization"]),
        ("multi_label", "I plan to transfer money to my grandson next week for his college semester tuition.", ["none"],
         "Your grandson was arrested in a foreign country: wire three thousand dollars in twenty minutes and don't tell your husband or he won't be released.", ["emotional_destabilization", "urgency", "isolation"])
    ]

    for idx, (topic_tag, benign_text, benign_labels, manip_text, manip_labels) in enumerate(contrastive_pairs_data):
        pair_id = f"cp_{idx:03d}"
        add(benign_text, benign_labels, f"contrastive_pair_{topic_tag}", difficulty="contrastive", topic=topic_tag, tone="benign", pair_id=pair_id, is_benign_pair_member=True)
        add(manip_text, manip_labels, f"contrastive_pair_{topic_tag}", difficulty="contrastive", topic=topic_tag, tone="manipulative", pair_id=pair_id, is_benign_pair_member=False)

    # =========================================================================
    # 2. BENIGN HARD NEGATIVES (110 turns) - All labeled ['none']
    # Vocabulary-rich turns with scam keywords in benign/educational contexts
    # =========================================================================
    hard_negatives = [
        "My grandson warned me that scammers often ask victims to buy Apple gift cards at grocery stores.",
        "I saw a news report about fraudsters posing as IRS agents and threatening people with arrest warrants.",
        "The bank teller handed me a pamphlet explaining common wire transfer scams targeting seniors.",
        "Can you explain how a Bitcoin ATM works and why people use cryptocurrency for investments?",
        "The local police department is holding a community workshop next Tuesday on fraud prevention.",
        "We bought several Target gift cards as holiday presents for our nieces and nephews this year.",
        "I read an article about how criminals use remote desktop software like TeamViewer to deceive people.",
        "Are wire transfers faster than traditional bank checks when purchasing real estate?",
        "We learned in our senior center lecture that real federal agencies never demand wire transfers.",
        "Is it true that the IRS only contacts taxpayers through official postal mail?",
        "The newspaper editorial discussed how fraudulent call centers operate internationally.",
        "I asked the bank branch manager if there is a fee for outgoing international wire transfers.",
        "We received a legitimate IRS refund check in the mail for our overpaid estimated quarterly taxes.",
        "My grandson showed me how to set up two-factor authentication on my iPad for online banking.",
        "The police officer at the school gave a presentation on internet safety and avoiding phishing emails.",
        "I need to go to the bank to deposit my paycheck and ask about their wire transfer policies.",
        "For my birthday, my daughter sent me a hundred dollar Amazon gift card.",
        "The documentary on Netflix explained how international scam rings launder money using Bitcoin.",
        "I had to call customer support because my account was locked after too many incorrect password attempts.",
        "The city council passed a resolution to increase funding for the police department's cybercrime unit.",
        "It's important to report any suspicious emails claiming to be from the IRS to the authorities.",
        "I used a wire transfer to pay for the down payment on my new car last week.",
        "My nephew got arrested for a minor traffic violation, but he's out on bail now.",
        "The local news anchor warned citizens about a new scam involving fake arrest warrants.",
        "I purchased a prepaid Visa gift card to give to the mail carrier as a thank you gift.",
        "The IRS website has a whole section dedicated to identifying tax fraud and scams.",
        "Our bank uses advanced encryption and fraud detection algorithms to protect our accounts.",
        "I heard a podcast about the history of Bitcoin and how decentralized finance works.",
        "The detective is investigating the burglary, but they haven't made an arrest yet.",
        "Always be cautious when someone calls out of the blue asking for a wire transfer.",
        "I bought a Steam gift card for my grandson so he can buy video games online.",
        "The university offers a course on cybersecurity and how to prevent digital fraud.",
        "I'm reading a thriller novel about a detective trying to catch a prolific scam artist.",
        "The police cruiser was parked outside the bank, probably just for a routine patrol.",
        "I need to file my taxes soon, I hope I don't get audited by the IRS.",
        "Wire fraud is a serious federal offense that carries severe penalties under the law.",
        "My friend was a victim of identity theft, it took her months to sort out her bank accounts.",
        "The tech support representative helped me reset my router without asking for any remote access.",
        "I read that you should never give your social security number to someone who calls claiming to be police.",
        "The bank sent me a notification that my new debit card is in the mail.",
        "We watched a movie about a group of hackers who orchestrate a massive cryptocurrency heist.",
        "The IRS office downtown is closed on all federal holidays.",
        "I accidentally bought the wrong type of gift card, I hope the store lets me exchange it.",
        "The police chief gave a press conference regarding the recent increase in neighborhood burglaries.",
        "I'm considering investing a small amount in Bitcoin just to see how it performs.",
        "The bank teller recognized me and asked how my grandchildren were doing.",
        "I received a phishing email that looked exactly like it was from my bank.",
        "The arrest record is public information and can be requested at the county courthouse.",
        "I listened to a radio segment about protecting seniors from financial exploitation and scams.",
        "I need to authorize a wire transfer to pay my property taxes by the end of the month."
    ]
    
    subjects = ["My friend", "My daughter", "The news anchor", "The bank manager", "A police officer", "The consumer report"]
    actions = [
        "told me about a scam involving wire transfers.",
        "warned us that the IRS never calls directly.",
        "explained how gift card fraud works.",
        "suggested we use strong passwords to prevent account hacking.",
        "said we should hang up on suspicious callers claiming to be the police.",
        "mentioned that Bitcoin ATMs are often used by fraudsters.",
        "reminded me to never share my screen with unknown tech support.",
        "discussed the signs of a fake arrest warrant over the phone.",
        "gave a presentation on avoiding internet scams.",
        "handed out brochures about safe banking practices."
    ]
    for s in subjects:
        for a in actions:
            hard_negatives.append(f"{s} {a}")

    for text in hard_negatives:
        add(text, ["none"], "benign_hard_neg", "hard", "educational", "neutral")

    # =========================================================================
    # 3. DIRECT FRAUD & MANIPULATION (300 turns)
    # Both keyword-explicit and subtle non-keyword manipulation
    # =========================================================================
    institutions = [
        ("the Cyber Threat Intelligence Unit", "tech"),
        ("the County Sheriff's Eviction Taskforce", "legal"),
        ("the Federal Trade Commission Oversight Board", "gov"),
        ("the Municipal Property Tax Assessment Bureau", "tax"),
        ("the Global Bank Wire Interception Department", "bank"),
        ("the International Cryptography Taskforce", "crypto"),
        ("the National Security Communications Division", "gov"),
        ("the State Healthcare Fraud Investigations Unit", "medical"),
        ("the Federal Reserve Money Laundering Taskforce", "bank"),
        ("the United States Postal Inspection Service", "gov"),
        ("the Bureau of Consumer Financial Protection", "bank"),
        ("the European Cybercrime Centre Operations Command", "tech"),
        ("the Division of Securities and Investment Fraud", "gov"),
        ("the National Border Patrol Enforcement Agency", "legal"),
        ("the Social Security Administration Identity Unit", "gov")
    ]
    actions = [
        ("terminate your account access within thirty minutes", ["urgency"]),
        ("require you to maintain absolute confidentiality from family", ["isolation"]),
        ("keep this telephone channel active until clearance is issued", ["channel_control"]),
        ("issue immediate seizure warrants against your real estate", ["emotional_destabilization", "urgency"]),
        ("arrest you if you do not comply immediately", ["authority", "urgency", "emotional_destabilization"]),
        ("monitor your screen to prevent further data loss", ["channel_control", "emotional_destabilization"]),
        ("ask that you do not speak to the bank tellers", ["isolation"]),
        ("demand you wire the funds before the end of the day", ["urgency", "authority"]),
        ("confiscate all your physical assets by tomorrow morning", ["emotional_destabilization", "urgency"]),
        ("require you to download our secure remote viewer tool", ["channel_control", "authority"])
    ]
    for (inst, topic), (act_text, act_labels) in itertools.product(institutions, actions):
        add(f"This is Senior Officer Walker from {inst}; we {act_text}.", ["authority"] + act_labels, "direct_fraud_expanded", "hard", topic, "official")
        add(f"I represent {inst}, and you are instructed to {act_text}.", ["authority"] + act_labels, "direct_fraud_expanded", "hard", topic, "stern")

    # =========================================================================
    # 4. BENIGN FILLER & DAILY SCENARIOS (30 turns) - All labeled ['none']
    # =========================================================================
    benign_filler = [
        "Good morning, how is the weather over on your side of town today?",
        "I was thinking of making vegetable soup for dinner tonight with carrots and celery.",
        "Did you get a chance to read the Sunday newspaper article about the botanical gardens?",
        "The traffic on the interstate was surprisingly light this morning during commute hours.",
        "Let me check my paper calendar to see if next Tuesday afternoon works for our lunch.",
        "Thank you so much for returning my call, I appreciate you getting back to me.",
        "I planted some tomato seedlings in the backyard garden over the weekend.",
        "Could you repeat that last sentence? My hearing aid had a little static.",
        "Yes, I understand what you're saying, that makes total sense.",
        "Alright, let me write that down on my notepad with a blue pen.",
        "Flight 842 to Chicago is at final boarding call, all ticketed passengers must report to Gate B12 immediately.",
        "The pharmacy closes in ten minutes, so please pick up your blood pressure medication before they lock up.",
        "Turn off the stove right now, the soup pot on the back burner is boiling over!",
        "We have a flash flood warning in effect for our county; move to higher ground immediately.",
        "Your dinner reservation at the restaurant will be released in five minutes if you don't check in at the host stand.",
        "The school bus is pulling up to the corner in two minutes, grab your backpack and coat quickly.",
        "The flash sale on airline tickets ends at midnight, so let's book our vacation dates tonight.",
        "Quickly put the frozen groceries into the freezer before the ice cream begins to melt.",
        "The theatre doors close promptly at 8 PM and late arrivals will not be seated until intermission.",
        "Please submit your timesheet before 5:00 PM today so payroll can process your direct deposit on time.",
        "Good afternoon, I am Dr. Patel, your cardiologist, following up on your recent echocardiogram results.",
        "This is Mrs. Higgins from the public library, letting you know the book you placed on hold is ready for pickup.",
        "Hello, I am Robert from the county board of elections verifying voter registration addresses.",
        "This is Pastor David calling from Grace Community Church regarding the volunteer schedule for Sunday.",
        "I am Sarah, the notary public assigned to witness your signature on the property deed transfer.",
        "Hello, this is Officer Davis with the highway patrol; I'm calling to let you know we found your lost license plate.",
        "I'm calling from the municipal water authority to notify residents of routine pipe maintenance on Thursday.",
        "This is Dean Alvarez from the university alumni association welcoming you to the homecoming weekend.",
        "Good morning, this is the building superintendent letting you know the elevator inspection is completed.",
        "I am Nurse Clara from the orthopedic clinic calling to confirm your physical therapy appointment for tomorrow."
    ]
    for text in benign_filler:
        add(text, ["none"], "benign_neutral", "standard", "general", "casual")

    # Save challenge set
    out_path = "data/challenge_set.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Successfully generated {len(samples)} distinct challenge turns in {out_path}")
    
    hn_count = sum(1 for s in samples if s["category"] == "benign_hard_neg")
    cp_count = sum(1 for s in samples if "pair_id" in s)
    print(f"Hard Negative count: {hn_count}")
    print(f"Contrastive Pair turns count: {cp_count} ({cp_count//2} pairs)")
    
    return samples

if __name__ == "__main__":
    generate_comprehensive_challenge_set()
