import json
import os
import difflib
from collections import defaultdict

def calculate_similarity(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def audit():
    print("================ DEEP TARGETED DATASET AUDIT ================\n")
    
    def load_turns(filepath):
        turns = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f: turns.append(json.loads(line))
        return turns
        
    train_turns = load_turns("data/processed/train.jsonl")
    val_turns = load_turns("data/processed/validation.jsonl")
    test_turns = load_turns("data/processed/test.jsonl")
    all_turns = train_turns + val_turns + test_turns
    
    if not all_turns:
        print("Dataset empty. Run generate and prepare first.")
        return

    # Tracking structures
    unique_raw = set()
    unique_norm = set()
    unique_caller = set()
    unique_recipient = set()
    unique_tactic = set()
    unique_benign_tactic = set()
    unique_fraud_tactic = set()
    
    tactic_stats = defaultdict(lambda: {"total": 0, "unique_texts": set(), "fraud": 0, "benign": 0, "unique_benign_texts": set(), "multi": 0, "benign_examples": []})
    
    topics = set()
    topic_convs = defaultdict(set)
    tactic_x_topic = defaultdict(int)
    fraud_x_topic = defaultdict(int)
    benign_x_topic = defaultdict(int)
    
    seqs_fraud = defaultdict(set)
    seqs_benign = defaultdict(set)
    tactic_orders = defaultdict(set)
    
    long_range = {"1+": 0, "2+": 0, "3+": 0, "max": 0, "examples": []}
    
    conv_groups = defaultdict(list)
    for t in all_turns: 
        conv_groups[t['conversation_id']].append(t)
        topics.add(t.get('source', 'unknown'))
        topic_convs[t.get('source', 'unknown')].add(t['conversation_id'])
        if t['conversation_level_label'] == 'fraud':
            fraud_x_topic[t.get('source', 'unknown')] += 1
        else:
            benign_x_topic[t.get('source', 'unknown')] += 1

    total_convs = len(conv_groups)
    fraud_convs = sum(1 for c, t in conv_groups.items() if t[0]['conversation_level_label'] == 'fraud')
    benign_convs = total_convs - fraud_convs
    
    conv_lengths = []
    tactic_counts_per_conv = defaultdict(int)
    
    # Analyze uniqueness and tactics
    for t in all_turns:
        text = t['text']
        norm = text.lower().strip()
        
        unique_raw.add(text)
        unique_norm.add(norm)
        
        if t['speaker'] == 'caller': unique_caller.add(norm)
        else: unique_recipient.add(norm)
            
        labels = t['labels']
        is_tactic = "none" not in labels
        is_fraud = t['conversation_level_label'] == 'fraud'
        is_multi = len(labels) > 1 and is_tactic
        
        if is_tactic:
            unique_tactic.add(norm)
            if is_fraud: unique_fraud_tactic.add(norm)
            else: unique_benign_tactic.add(norm)
                
            for l in labels:
                tactic_stats[l]["total"] += 1
                tactic_stats[l]["unique_texts"].add(norm)
                if is_fraud: tactic_stats[l]["fraud"] += 1
                else: 
                    tactic_stats[l]["benign"] += 1
                    tactic_stats[l]["unique_benign_texts"].add(norm)
                    if len(tactic_stats[l]["benign_examples"]) < 10 and norm not in [e[0] for e in tactic_stats[l]["benign_examples"]]:
                        tactic_stats[l]["benign_examples"].append((norm, t['source']))
                if is_multi: tactic_stats[l]["multi"] += 1
                
                tactic_x_topic[t.get('source', 'unknown')] += 1
                
    # Analyze sequences and long range dependencies
    for cid, turns in conv_groups.items():
        conv_lengths.append(len(turns))
        
        tactic_turns = [t for t in turns if "none" not in t['labels']]
        tactic_counts_per_conv[len(tactic_turns)] += 1
        
        seq = []
        order = []
        normal_gap = 0
        max_gap = 0
        has_tactic = False
        
        # for long range examples
        last_tactic_text = ""
        current_gap = 0
        
        for t in turns:
            if t['speaker'] != 'caller': continue
            
            if "none" not in t['labels']:
                l_str = "+".join(sorted(t['labels']))
                seq.append(l_str)
                order.append(l_str)
                
                if has_tactic:
                    max_gap = max(max_gap, normal_gap)
                    if normal_gap >= 1: 
                        if cid not in [e['cid'] for e in long_range['examples']] and len(long_range['examples']) < 5:
                            long_range['examples'].append({"cid": cid, "gap": normal_gap, "t1": last_tactic_text, "t2": t['text']})
                has_tactic = True
                last_tactic_text = t['text']
                normal_gap = 0
            else:
                if has_tactic:
                    normal_gap += 1
                    
        seq_str = " -> ".join(seq)
        if not seq_str: seq_str = "none"
        
        is_fraud = turns[0]['conversation_level_label'] == 'fraud'
        if is_fraud: seqs_fraud[seq_str].add(cid)
        else: seqs_benign[seq_str].add(cid)
        
        tactic_orders[" -> ".join(order)].add(cid)
        
        if max_gap >= 1: long_range["1+"] += 1
        if max_gap >= 2: long_range["2+"] += 1
        if max_gap >= 3: long_range["3+"] += 1
        long_range["max"] = max(long_range["max"], max_gap)

    # Leakage and Overlap Checks
    train_norms = set(t['text'].lower().strip() for t in train_turns)
    val_norms = set(t['text'].lower().strip() for t in val_turns)
    test_norms = set(t['text'].lower().strip() for t in test_turns)
    train_callers = set(t['text'].lower().strip() for t in train_turns if t['speaker'] == 'caller')
    test_callers = set(t['text'].lower().strip() for t in test_turns if t['speaker'] == 'caller')
    
    exact_leakage = len(train_norms.intersection(test_norms))
    caller_leakage = len(train_callers.intersection(test_callers))
    
    # Check against held-out challenge set
    challenge_norms = set()
    if os.path.exists("data/challenge_set.jsonl"):
        with open("data/challenge_set.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    challenge_norms.add(item["text"].lower().strip())
    challenge_overlap_train = len(train_norms.intersection(challenge_norms))
    challenge_overlap_val = len(val_norms.intersection(challenge_norms))
    challenge_overlap_test = len(test_norms.intersection(challenge_norms))

    hard_negative_count = sum(1 for t in all_turns if t.get("difficulty") == "hard_negative" or t.get("difficulty") == "contrastive_benign")
    contrastive_turn_count = sum(1 for t in all_turns if "contrastive" in t.get("difficulty", "") or t.get("source_type") == "contrastive_pair")

    # ------------------ PRINT REPORT ------------------
    print("### 1. Overall Dataset & Split Dimensions")
    print(f"- Total Conversations: {total_convs} (Fraud: {fraud_convs}, Benign: {benign_convs})")
    print(f"- Total Turns: {len(all_turns)} (Train: {len(train_turns)}, Validation: {len(val_turns)}, Test: {len(test_turns)})")
    print(f"- Unique raw sentences: {len(unique_raw)}")
    print(f"- Unique normalized sentences: {len(unique_norm)}")
    print(f"- Unique caller sentences: {len(unique_caller)}")
    print(f"- Unique recipient sentences: {len(unique_recipient)}")
    print(f"- Hard Negative Turns: {hard_negative_count}")
    print(f"- Contrastive Pair Turns in Dataset: {contrastive_turn_count}")

    print("\n### 2. Tactic Distribution (Multi-Hot)")
    tactics = ["urgency", "isolation", "authority", "emotional_destabilization", "channel_control"]
    for tac in tactics:
        s = tactic_stats[tac]
        print(f"[{tac.upper()}] Total: {s['total']} | Unique: {len(s['unique_texts'])} | Fraud: {s['fraud']} | Benign: {s['benign']} | Multi-label: {s['multi']}")

    print("\n### 3. Leakage & Challenge-Set Isolation Audit")
    print(f"- Exact duplicate normalized sentences across Train/Test: {exact_leakage} (Recipient/normal acknowledgments)")
    print(f"- Exact duplicate CALLER sentences across Train/Test: {caller_leakage}")
    print(f"- Overlap with held-out Challenge Set (Train): {challenge_overlap_train} (Must be 0)")
    print(f"- Overlap with held-out Challenge Set (Validation): {challenge_overlap_val} (Must be 0)")
    print(f"- Overlap with held-out Challenge Set (Test): {challenge_overlap_test} (Must be 0)")

    print("\n### 4. Sequence & Structure Diversity")
    print(f"- Total unique sequences across dataset: {len(seqs_fraud) + len(seqs_benign)}")
    print(f"- Avg turns per conv: {sum(conv_lengths)/total_convs:.1f} (Range: {min(conv_lengths)} to {max(conv_lengths)})")
    print(f"- Convs with 0 tactics: {tactic_counts_per_conv[0]}")
    print(f"- Convs with 1 tactic: {tactic_counts_per_conv[1]}")
    print(f"- Convs with 2 tactics: {tactic_counts_per_conv[2]}")
    print(f"- Convs with 3 tactics: {tactic_counts_per_conv[3]}")
    print(f"- Convs with 4+ tactics: {sum(v for k,v in tactic_counts_per_conv.items() if k >= 4)}")

    print("\n### 5. Long-Range Dependencies (Normal turns between tactics)")
    print(f"- Convs with 1+ normal turns between tactics: {long_range['1+']}")
    print(f"- Convs with 2+ normal turns between tactics: {long_range['2+']}")
    print(f"- Convs with 3+ normal turns between tactics: {long_range['3+']}")
    print(f"- Max gap: {long_range['max']} normal turns")

    print("\n### 6. Semantic/Topic Distribution")
    print(f"- Distinct topics: {len(topics)}")
    for topic in sorted(topics):
        print(f"  [{topic.upper()}] Convs: {len(topic_convs[topic])} | Fraud: {fraud_x_topic[topic]} | Benign: {benign_x_topic[topic]} | Tactic turns: {tactic_x_topic[topic]}")

if __name__ == "__main__":
    audit()

