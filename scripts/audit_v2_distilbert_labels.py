import os
import sys
import json
import torch
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TACTIC_LABELS = [
    "urgency",
    "isolation",
    "authority",
    "emotional_destabilization",
    "channel_control"
]

def run_label_quality_audit():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(project_root, "ml", "models", "tactic_classifier", "v2_temporal_validity")
    test_file = os.path.join(project_root, "data", "processed", "temporal_v2", "test.jsonl")
    
    with open(os.path.join(model_dir, "threshold_config.json"), "r") as f:
        t_cfg = json.load(f)
    thresholds = t_cfg["thresholds"]
    thresh_map = {TACTIC_LABELS[i]: thresholds[i] for i in range(len(TACTIC_LABELS))}
    
    print("=" * 80)
    print("V2 DISTILBERT LABEL QUALITY & ANNOTATION ERROR AUDIT")
    print("=" * 80)
    print(f"Model: {model_dir}")
    print(f"Test File: {test_file}")
    print(f"Calibrated Thresholds: {thresh_map}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    turns = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line.strip()))
                
    print(f"Total test turns: {len(turns)}")
    
    # Run batch inference
    all_texts = [t["text"] for t in turns]
    batch_size = 64
    all_probs = []
    
    for i in range(0, len(all_texts), batch_size):
        batch_txt = all_texts[i:i+batch_size]
        enc = tokenizer(batch_txt, truncation=True, padding=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).numpy()
            all_probs.extend(probs)
            
    all_probs = np.array(all_probs)
    
    # -------------------------------------------------------------
    # TASK 1: Audit all Fraud turns labeled ["none"]
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TASK 1: AUDIT OF FRAUD TURNS LABELED ['none']")
    print("=" * 50)
    
    fraud_none_turns = []
    fraud_none_overpreds = []
    
    for idx, (t, p) in enumerate(zip(turns, all_probs)):
        is_fraud = (t.get("conversation_level_label") == "fraud")
        raw_labels = t.get("labels", [])
        is_none_label = (raw_labels == ["none"] or len(raw_labels) == 0)
        
        if is_fraud and is_none_label:
            pred_tactics = [TACTIC_LABELS[j] for j in range(5) if p[j] > thresholds[j]]
            record = {
                "turn_index": idx,
                "conversation_id": t.get("conversation_id"),
                "turn_id": t.get("turn_id"),
                "speaker": t.get("speaker"),
                "text": t.get("text"),
                "true_labels": raw_labels,
                "predicted_tactics": pred_tactics,
                "probabilities": {TACTIC_LABELS[j]: round(float(p[j]), 4) for j in range(5)}
            }
            fraud_none_turns.append(record)
            if len(pred_tactics) > 0:
                fraud_none_overpreds.append(record)
                
    print(f"Total Fraud turns labeled ['none']: {len(fraud_none_turns)}")
    print(f"Fraud ['none'] turns where model predicted >= 1 tactic: {len(fraud_none_overpreds)} ({len(fraud_none_overpreds)/len(fraud_none_turns)*100:.2f}%)")
    
    tactic_overpred_counts = Counter()
    for rec in fraud_none_overpreds:
        for tac in rec["predicted_tactics"]:
            tactic_overpred_counts[tac] += 1
            
    print(f"Breakdown of predicted tactics on Fraud ['none'] turns: {dict(tactic_overpred_counts)}")
    
    print("\nSample Fraud ['none'] Over-predictions:")
    for rec in fraud_none_overpreds[:8]:
        print(f"  CID: {rec['conversation_id']} (Turn {rec['turn_id']}, {rec['speaker']})")
        print(f"  Text: \"{rec['text']}\"")
        print(f"  True: {rec['true_labels']} | Pred: {rec['predicted_tactics']} | Probs: {rec['probabilities']}\n")

    # -------------------------------------------------------------
    # TASK 2: Specifically Audit Isolation on True=0, Pred=1
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TASK 2: ISOLATION LABEL QUALITY CLASSIFICATION")
    print("=" * 50)
    
    # Analyze generation script generator logic
    # In generate_synthetic_data.py:
    # `seq = [..., ("none", random.choice(self.vocab.fraud_suppression_turns)), ...]`
    # In immediate_coercion: line 439 labeled fraud_suppression_turns as labels=["none"]!
    # But fraud_suppression_turns contains:
    # "Do NOT call the number on your card or attempt to verify this with your bank; our investigators have flagged internal employees as suspects."
    # "Do NOT hang up this phone to check any website; breaking this secure connection will forfeit your emergency protection immediately."
    # "You are legally prohibited from contacting outside parties or family members for verification during an active federal operation."
    # "Calling any public customer service number will tip off the syndicate and cause your accounts to be emptied instantly."
    
    isolation_fp_records = []
    iso_idx = TACTIC_LABELS.index("isolation")
    
    for idx, (t, p) in enumerate(zip(turns, all_probs)):
        is_fraud = (t.get("conversation_level_label") == "fraud")
        raw_labels = t.get("labels", [])
        true_iso = 1 if "isolation" in raw_labels else 0
        pred_iso = 1 if p[iso_idx] > thresholds[iso_idx] else 0
        
        if is_fraud and true_iso == 0 and pred_iso == 1:
            txt = t.get("text", "")
            # Classification
            is_suppression_text = any(phrase in txt.lower() for phrase in [
                "do not call", "do not hang up", "contacting outside parties", "family members", "public customer service", "tip off", "prohibited"
            ])
            
            if is_suppression_text:
                cat = "B. Mislabeled Isolation / Verification Suppression"
            elif any(w in txt.lower() for w in ["routine", "bear with me", "update your contact"]):
                cat = "A. Genuine Non-Isolation (Classifier Error)"
            else:
                cat = "C. Ambiguous"
                
            isolation_fp_records.append({
                "conversation_id": t.get("conversation_id"),
                "turn_id": t.get("turn_id"),
                "text": txt,
                "prob": round(float(p[iso_idx]), 4),
                "category": cat
            })
            
    cat_counts = Counter(r["category"] for r in isolation_fp_records)
    print(f"Total Fraud Turns with True Isolation=0, Pred Isolation=1: {len(isolation_fp_records)}")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} ({count/len(isolation_fp_records)*100:.1f}%)")
        
    print("\nExamples for each Category:")
    for cat in sorted(cat_counts.keys()):
        print(f"\n--- Category: {cat} ---")
        exs = [r for r in isolation_fp_records if r["category"] == cat][:3]
        for ex in exs:
            print(f"  CID: {ex['conversation_id']} (Turn {ex['turn_id']}) [p_iso={ex['prob']}]: \"{ex['text']}\"")

    # -------------------------------------------------------------
    # TASK 3: Systematic Sub-category Breakdown
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TASK 3: SYSTEMATIC ISOLATION SUB-CATEGORY BREAKDOWN")
    print("=" * 50)
    
    subcategories = {
        "1. Fraud Tactic-Bearing Turns": lambda t: t.get("conversation_level_label") == "fraud" and any(l in TACTIC_LABELS for l in t.get("labels", [])),
        "2. Fraud Normal Turns (Gaps/Openers)": lambda t: t.get("conversation_level_label") == "fraud" and t.get("labels", []) == ["none"] and not any(kw in t.get("text", "").lower() for kw in ["do not call", "prohibited", "tip off", "do not hang up"]),
        "3. Fraud Verification-Suppression Turns": lambda t: t.get("conversation_level_label") == "fraud" and any(kw in t.get("text", "").lower() for kw in ["do not call", "prohibited", "tip off", "do not hang up"]),
        "4. Benign Tactic-Bearing Turns": lambda t: t.get("conversation_level_label") == "benign" and any(l in TACTIC_LABELS for l in t.get("labels", [])),
        "5. Benign Hard Negatives (Caller None)": lambda t: t.get("conversation_level_label") == "benign" and t.get("labels", []) == ["none"] and t.get("speaker") == "caller",
        "6. Benign Recipient Turns": lambda t: t.get("conversation_level_label") == "benign" and t.get("speaker") == "recipient",
    }
    
    print(f"{'Subcategory':<42} | {'Total Turns':<11} | {'Pred Isolation +':<16} | {'Isolation Rate':<14}")
    print("-" * 90)
    
    task3_results = {}
    for name, fn in subcategories.items():
        subset_indices = [i for i, t in enumerate(turns) if fn(t)]
        if not subset_indices:
            continue
        sub_p = all_probs[subset_indices, iso_idx]
        pred_pos = int((sub_p > thresholds[iso_idx]).sum())
        tot = len(subset_indices)
        rate = pred_pos / tot if tot > 0 else 0.0
        print(f"{name:<42} | {tot:<11} | {pred_pos:<16} | {rate*100:<13.2f}%")
        task3_results[name] = {
            "total_turns": tot,
            "pred_isolation_positives": pred_pos,
            "rate": round(rate, 4)
        }

    # -------------------------------------------------------------
    # TASK 4: Check All Five Tactics for Plausible False Positives
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TASK 4: CHECK ALL FIVE TACTICS ON LABELED ['none'] TURNS")
    print("=" * 50)
    
    all_tactic_audit = {}
    print(f"{'Tactic':<25} | {'None Turns':<12} | {'Over-predictions':<18} | {'Over-pred Rate':<16}")
    print("-" * 75)
    
    for j, tactic in enumerate(TACTIC_LABELS):
        none_indices = [i for i, t in enumerate(turns) if tactic not in t.get("labels", [])]
        p_tac = all_probs[none_indices, j]
        overpreds = int((p_tac > thresholds[j]).sum())
        tot_none = len(none_indices)
        rate = overpreds / tot_none if tot_none > 0 else 0.0
        print(f"{tactic:<25} | {tot_none:<12} | {overpreds:<18} | {rate*100:<15.2f}%")
        all_tactic_audit[tactic] = {
            "total_negative_turns": tot_none,
            "over_predictions": overpreds,
            "rate": round(rate, 4)
        }

    # Save detailed JSON report
    audit_data = {
        "task1_fraud_none_overpredictions": {
            "total_fraud_none_turns": len(fraud_none_turns),
            "overprediction_count": len(fraud_none_overpreds),
            "overprediction_rate": round(len(fraud_none_overpreds) / len(fraud_none_turns), 4),
            "by_tactic": dict(tactic_overpred_counts),
            "examples": fraud_none_overpreds[:10]
        },
        "task2_isolation_breakdown": {
            "total_isolation_fps": len(isolation_fp_records),
            "category_counts": dict(cat_counts),
            "records": isolation_fp_records
        },
        "task3_systematic_subcategories": task3_results,
        "task4_all_tactics": all_tactic_audit
    }
    
    with open(os.path.join(project_root, "data", "v2_distilbert_label_quality_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        
    print(f"\nSaved raw JSON report to data/v2_distilbert_label_quality_audit.json")

if __name__ == "__main__":
    run_label_quality_audit()
