import json
import numpy as np

def audit_sequences():
    splits = ['train', 'val', 'test']
    cids = {}
    
    print("=" * 60)
    print("PHASE 5 FEATURE EXTRACTION AUDIT")
    print("=" * 60)
    
    for s in splits:
        path = f"data/processed/temporal/{s}_seq.jsonl"
        lines = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        cids[s] = set(l["conversation_id"] for l in lines)
        lens = [l["num_turns"] for l in lines]
        turns = sum(lens)
        fraud_cnt = sum(1 for l in lines if l["label"] == 1)
        benign_cnt = sum(1 for l in lines if l["label"] == 0)
        
        # Check dimensional consistency
        dim_check = all(len(tp) == 5 for l in lines for tp in l["turn_probabilities"])
        range_check = all(0.0 <= p <= 1.0 for l in lines for tp in l["turn_probabilities"] for p in tp)
        
        print(f"\n--- {s.upper()} SPLIT ---")
        print(f"File: {path}")
        print(f"Sequences / Conversations: {len(lines)}")
        print(f"Total Turns: {turns}")
        print(f"Class Distribution: Fraud = {fraud_cnt}, Benign = {benign_cnt}")
        print(f"Sequence Lengths: Min = {min(lens)}, Max = {max(lens)}, Mean = {np.mean(lens):.2f}, Median = {np.median(lens):.1f}")
        print(f"All Timesteps exactly 5D continuous probabilities: {dim_check}")
        print(f"All probabilities within [0.0, 1.0]: {range_check}")
        
    print("\n--- DISJOINT SPLIT INTEGRITY ---")
    print(f"Train & Validation Overlap: {len(cids['train'] & cids['val'])} (Must be 0)")
    print(f"Train & Test Overlap:       {len(cids['train'] & cids['test'])} (Must be 0)")
    print(f"Validation & Test Overlap:  {len(cids['val'] & cids['test'])} (Must be 0)")
    
    print("\n--- SAMPLE EXTRACTED SEQUENCES ---")
    val_lines = [json.loads(l) for l in open("data/processed/temporal/val_seq.jsonl", "r", encoding="utf-8")]
    for idx, sample in enumerate(val_lines[:3]):
        print(f"\nSample {idx+1}:")
        print(f"  Conversation ID: {sample['conversation_id']}")
        print(f"  Topic:           {sample['topic']}")
        print(f"  Label:           {sample['conversation_level_label']} ({sample['label']})")
        print(f"  Length:          {sample['num_turns']} turns")
        print(f"  First 3 Turn Probability Vectors (urgency, isolation, authority, emotional, channel):")
        for t_i, vec in enumerate(sample['turn_probabilities'][:3]):
            print(f"    Turn {t_i+1} ({sample['turn_speakers'][t_i]}): {vec} | Text: '{sample['turn_texts'][t_i][:70]}...'")

if __name__ == "__main__":
    audit_sequences()
