import os
import sys
import json
import torch
import numpy as np
import random
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.temporal_dataset import ConversationSequenceDataset, get_aggregated_features_for_lr
from ml.temporal_model import CausalLSTMRiskModel
from ml.temporal_baselines import (
    MajorityClassBaseline,
    TacticCountRuleBaseline,
    AggregatedLogisticRegressionBaseline,
    calculate_full_classification_metrics
)

def run_validity_audit():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data", "processed", "temporal")
    model_dir = os.path.join(project_root, "ml", "models", "temporal_risk", "lstm_v1")
    
    train_path = os.path.join(data_dir, "train_seq.jsonl")
    val_path = os.path.join(data_dir, "val_seq.jsonl")
    test_path = os.path.join(data_dir, "test_seq.jsonl")
    
    train_ds = ConversationSequenceDataset(train_path)
    val_ds = ConversationSequenceDataset(val_path)
    test_ds = ConversationSequenceDataset(test_path)
    
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    threshold = config["calibrated_threshold"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lstm_model = CausalLSTMRiskModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pooling_head=config["pooling_head"]
    ).to(device)
    lstm_model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device))
    lstm_model.eval()

    print("=" * 80)
    print("PHASE 5 SCIENTIFIC VALIDITY AUDIT EXECUTION")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # AUDIT 1: TACTIC COUNT vs FRAUD LABEL
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("AUDIT 1 — TACTIC COUNT vs FRAUD LABEL (TEST SET: N=240)")
    print("=" * 40)
    
    tactic_thresh = 0.2
    
    # Analyze distributions
    contingency = {
        "zero": {"fraud": 0, "benign": 0},
        "ge1": {"fraud": 0, "benign": 0},
        "ge2": {"fraud": 0, "benign": 0},
        "ge3": {"fraud": 0, "benign": 0},
    }
    
    all_tactics_count = []
    all_max_probs = []
    all_sum_probs = []
    all_final_probs = []
    
    for item in test_ds.samples:
        is_fraud = (item["label"] == 1)
        probs = np.array(item["turn_probabilities"], dtype=np.float32)
        
        # Tactic bearing turns (turns where any tactic > 0.2)
        turn_has_tactic = (probs > tactic_thresh).any(axis=1) if len(probs) > 0 else np.array([])
        num_tactic_turns = int(turn_has_tactic.sum()) if len(probs) > 0 else 0
        
        max_p = float(probs.max()) if len(probs) > 0 else 0.0
        sum_p = float(probs.sum()) if len(probs) > 0 else 0.0
        final_p = float(probs[-1].max()) if len(probs) > 0 else 0.0
        
        # distinct tactics detected (across entire conv)
        distinct_tactics = int((probs.max(axis=0) > tactic_thresh).sum()) if len(probs) > 0 else 0
        
        target = "fraud" if is_fraud else "benign"
        if num_tactic_turns == 0:
            contingency["zero"][target] += 1
        if num_tactic_turns >= 1:
            contingency["ge1"][target] += 1
        if num_tactic_turns >= 2:
            contingency["ge2"][target] += 1
        if num_tactic_turns >= 3:
            contingency["ge3"][target] += 1
            
        all_tactics_count.append(num_tactic_turns)
        all_max_probs.append(max_p)
        all_sum_probs.append(sum_p)
        all_final_probs.append(final_p)
        
    print("\nContingency Table (Tactic-bearing turns at threshold > 0.20):")
    print(f"{'Condition':<15} | {'Fraud':<8} | {'Benign':<8} | {'P(Fraud | Condition)':<20}")
    print("-" * 60)
    for cond in ["zero", "ge1", "ge2", "ge3"]:
        f_count = contingency[cond]["fraud"]
        b_count = contingency[cond]["benign"]
        tot = f_count + b_count
        p_fraud = (f_count / tot) if tot > 0 else 0.0
        print(f"{cond:<15} | {f_count:<8} | {b_count:<8} | {p_fraud:<20.4f}")
        
    # Check max tactic prob distribution
    fraud_max_probs = [p for p, item in zip(all_max_probs, test_ds.samples) if item["label"] == 1]
    benign_max_probs = [p for p, item in zip(all_max_probs, test_ds.samples) if item["label"] == 0]
    
    print(f"\nMax Tactic Probability across turns:")
    print(f"  Fraud:  min={np.min(fraud_max_probs):.4f}, mean={np.mean(fraud_max_probs):.4f}, max={np.max(fraud_max_probs):.4f}")
    print(f"  Benign: min={np.min(benign_max_probs):.4f}, mean={np.mean(benign_max_probs):.4f}, max={np.max(benign_max_probs):.4f}")

    # -------------------------------------------------------------
    # AUDIT 3 & 4: BASELINES VERIFICATION
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("AUDIT 3 & 4 — BASELINE SPECIFICATION CHECK")
    print("=" * 40)
    
    # -------------------------------------------------------------
    # AUDIT 5: TRUE TEMPORAL VALUE TEST (Permutation Ablation)
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("AUDIT 5 — TRUE TEMPORAL VALUE TEST: PERMUTATION ABLATION")
    print("=" * 40)
    
    # 1. Original Chronological Evaluation
    orig_probs = []
    y_test = np.array([s["label"] for s in test_ds.samples])
    
    with torch.no_grad():
        for item in test_ds.samples:
            sub_probs = np.array(item["turn_probabilities"], dtype=np.float32)
            seq_len = len(sub_probs)
            sub_tensor = torch.tensor(sub_probs).unsqueeze(0).to(device)
            sub_mask = torch.ones((1, seq_len), dtype=torch.float32).to(device)
            sub_len = torch.tensor([seq_len], dtype=torch.long).to(device)
            r = float(torch.sigmoid(lstm_model(sub_tensor, sub_mask, sub_len)).item())
            orig_probs.append(r)
            
    m_orig = calculate_full_classification_metrics(y_test, np.array(orig_probs), threshold=threshold)
    print(f"Original Chronological Test Set:")
    print(f"  ROC-AUC={m_orig['roc_auc']:.4f}, PR-AUC={m_orig['pr_auc']:.4f}, F1={m_orig['f1']:.4f}, Acc={m_orig['accuracy']:.4f}, Prec={m_orig['precision']:.4f}, Rec={m_orig['recall']:.4f}")
    
    # 2. Permuted Sequences (Multiple Shuffles)
    shuffled_f1s = []
    shuffled_rocs = []
    shuffled_prs = []
    shuffled_accs = []
    
    num_shuffles = 20
    for seed in range(num_shuffles):
        np.random.seed(seed + 100)
        perm_probs = []
        for item in test_ds.samples:
            probs = np.array(item["turn_probabilities"], dtype=np.float32)
            seq_len = len(probs)
            # Permute rows (timesteps) within the conversation
            perm_indices = np.random.permutation(seq_len)
            perm_seq = probs[perm_indices]
            
            sub_tensor = torch.tensor(perm_seq).unsqueeze(0).to(device)
            sub_mask = torch.ones((1, seq_len), dtype=torch.float32).to(device)
            sub_len = torch.tensor([seq_len], dtype=torch.long).to(device)
            
            with torch.no_grad():
                r = float(torch.sigmoid(lstm_model(sub_tensor, sub_mask, sub_len)).item())
            perm_probs.append(r)
            
        m_perm = calculate_full_classification_metrics(y_test, np.array(perm_probs), threshold=threshold)
        shuffled_f1s.append(m_perm["f1"])
        shuffled_rocs.append(m_perm["roc_auc"])
        shuffled_prs.append(m_perm["pr_auc"])
        shuffled_accs.append(m_perm["accuracy"])
        
    print(f"\nPermuted/Shuffled Timesteps (Within Conversation, N={num_shuffles} random seeds):")
    print(f"  ROC-AUC: mean={np.mean(shuffled_rocs):.4f} +/- {np.std(shuffled_rocs):.4f} (range: {np.min(shuffled_rocs):.4f} - {np.max(shuffled_rocs):.4f})")
    print(f"  PR-AUC:  mean={np.mean(shuffled_prs):.4f} +/- {np.std(shuffled_prs):.4f} (range: {np.min(shuffled_prs):.4f} - {np.max(shuffled_prs):.4f})")
    print(f"  F1:      mean={np.mean(shuffled_f1s):.4f} +/- {np.std(shuffled_f1s):.4f} (range: {np.min(shuffled_f1s):.4f} - {np.max(shuffled_f1s):.4f})")
    print(f"  Acc:     mean={np.mean(shuffled_accs):.4f} +/- {np.std(shuffled_accs):.4f}")

    # -------------------------------------------------------------
    # AUDIT 6: PREFIX COMPARISON DEEP DIVE
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("AUDIT 6 — PREFIX COMPARISON & DENOMINATOR VERIFICATION")
    print("=" * 40)
    
    # Check length distribution of test set
    lengths = [len(s["turn_probabilities"]) for s in test_ds.samples]
    fraud_lens = [len(s["turn_probabilities"]) for s in test_ds.samples if s["label"] == 1]
    benign_lens = [len(s["turn_probabilities"]) for s in test_ds.samples if s["label"] == 0]
    print(f"Length distribution in test set:")
    print(f"  Total: min={min(lengths)}, max={max(lengths)}, counts={dict(Counter(lengths))}")
    print(f"  Fraud counts:  {dict(sorted(Counter(fraud_lens).items()))}")
    print(f"  Benign counts: {dict(sorted(Counter(benign_lens).items()))}")
    
    prefix_checkpoints = [1, 2, 3, 4, 5, 6, 7]
    print(f"\nDetailed Prefix Breakdown (Only evaluating conversations with length >= K):")
    print(f"{'Checkpoint':<10} | {'Total N':<8} | {'Fraud N':<8} | {'Benign N':<8} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'TN':<4} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 95)
    
    for k in prefix_checkpoints:
        sub_samples = [s for s in test_ds.samples if len(s["turn_probabilities"]) >= k]
        p_labels = []
        p_probs = []
        for s in sub_samples:
            sub_probs = np.array(s["turn_probabilities"][:k], dtype=np.float32)
            sub_tensor = torch.tensor(sub_probs).unsqueeze(0).to(device)
            sub_mask = torch.ones((1, k), dtype=torch.float32).to(device)
            sub_len = torch.tensor([k], dtype=torch.long).to(device)
            with torch.no_grad():
                r = float(torch.sigmoid(lstm_model(sub_tensor, sub_mask, sub_len)).item())
            p_probs.append(r)
            p_labels.append(s["label"])
            
        p_labels = np.array(p_labels)
        p_probs = np.array(p_probs)
        m = calculate_full_classification_metrics(p_labels, p_probs, threshold=threshold)
        cm = m["confusion_matrix"]
        fraud_n = int((p_labels == 1).sum())
        benign_n = int((p_labels == 0).sum())
        print(f"Turn {k:<5} | {len(sub_samples):<8} | {fraud_n:<8} | {benign_n:<8} | {cm['tp']:<4} | {cm['fp']:<4} | {cm['fn']:<4} | {cm['tn']:<4} | {m['precision']:<6.4f} | {m['recall']:<6.4f} | {m['f1']:<6.4f}")

    print(f"\nFixed-Denominator Prefix Breakdown (All 122 Fraud / 118 Benign; short conversations use their final turn):")
    print(f"{'Checkpoint':<10} | {'Total N':<8} | {'Fraud N':<8} | {'Benign N':<8} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'TN':<4} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 95)
    for k in prefix_checkpoints:
        p_labels = []
        p_probs = []
        for s in test_ds.samples:
            actual_k = min(k, len(s["turn_probabilities"]))
            sub_probs = np.array(s["turn_probabilities"][:actual_k], dtype=np.float32)
            sub_tensor = torch.tensor(sub_probs).unsqueeze(0).to(device)
            sub_mask = torch.ones((1, actual_k), dtype=torch.float32).to(device)
            sub_len = torch.tensor([actual_k], dtype=torch.long).to(device)
            with torch.no_grad():
                r = float(torch.sigmoid(lstm_model(sub_tensor, sub_mask, sub_len)).item())
            p_probs.append(r)
            p_labels.append(s["label"])
        p_labels = np.array(p_labels)
        p_probs = np.array(p_probs)
        m = calculate_full_classification_metrics(p_labels, p_probs, threshold=threshold)
        cm = m["confusion_matrix"]
        fraud_n = int((p_labels == 1).sum())
        benign_n = int((p_labels == 0).sum())
        print(f"Turn {k:<5} | {len(test_ds.samples):<8} | {fraud_n:<8} | {benign_n:<8} | {cm['tp']:<4} | {cm['fp']:<4} | {cm['fn']:<4} | {cm['tn']:<4} | {m['precision']:<6.4f} | {m['recall']:<6.4f} | {m['f1']:<6.4f}")

    # -------------------------------------------------------------
    # AUDIT 7: TIME-TO-DETECTION DISTRIBUTION
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("AUDIT 7 — TIME-TO-DETECTION EXACT DISTRIBUTION")
    print("=" * 40)
    
    fraud_samples = [s for s in test_ds.samples if s["label"] == 1]
    det_turns = []
    never_detected = []
    
    for item in fraud_samples:
        probs = item["turn_probabilities"]
        detected = False
        for t_idx in range(1, len(probs) + 1):
            sub_probs = np.array(probs[:t_idx], dtype=np.float32)
            sub_tensor = torch.tensor(sub_probs).unsqueeze(0).to(device)
            sub_mask = torch.ones((1, t_idx), dtype=torch.float32).to(device)
            sub_len = torch.tensor([t_idx], dtype=torch.long).to(device)
            with torch.no_grad():
                r = float(torch.sigmoid(lstm_model(sub_tensor, sub_mask, sub_len)).item())
            if r >= threshold:
                det_turns.append(t_idx)
                detected = True
                break
        if not detected:
            never_detected.append(item["conversation_id"])
            
    print(f"Total fraud conversations: {len(fraud_samples)}")
    print(f"Total detected: {len(det_turns)}")
    print(f"Never detected before end: {len(never_detected)}")
    
    counts = Counter(det_turns)
    total_f = len(fraud_samples)
    print(f"\n{'Detection Turn':<15} | {'Count':<8} | {'Percentage':<12} | {'Cumulative %':<15}")
    print("-" * 60)
    cum = 0
    for t in sorted(counts.keys()):
        c = counts[t]
        cum += c
        print(f"Turn {t:<10} | {c:<8} | {c/total_f*100:<11.2f}% | {cum/total_f*100:<14.2f}%")
    print(f"Mean detection turn:   {np.mean(det_turns):.4f}")
    print(f"Median detection turn: {np.median(det_turns):.4f}")

    # -------------------------------------------------------------
    # AUDIT 8: FALSE POSITIVES & FALSE NEGATIVES
    # -------------------------------------------------------------
    print("\n" + "=" * 40)
    print("AUDIT 8 — FALSE POSITIVE & FALSE NEGATIVE CASEREVIEW")
    print("=" * 40)
    
    fps = []
    fns = []
    
    for item in test_ds.samples:
        probs = np.array(item["turn_probabilities"], dtype=np.float32)
        seq_len = len(probs)
        sub_tensor = torch.tensor(probs).unsqueeze(0).to(device)
        sub_mask = torch.ones((1, seq_len), dtype=torch.float32).to(device)
        sub_len = torch.tensor([seq_len], dtype=torch.long).to(device)
        with torch.no_grad():
            r = float(torch.sigmoid(lstm_model(sub_tensor, sub_mask, sub_len)).item())
        pred = int(r >= threshold)
        actual = int(item["label"])
        
        tactic_bearing_turns = int((probs > 0.2).any(axis=1).sum()) if seq_len > 0 else 0
        max_tactic_p = float(probs.max()) if seq_len > 0 else 0.0
        
        info = {
            "conversation_id": item["conversation_id"],
            "label": actual,
            "topic": item.get("topic", "unknown"),
            "num_turns": seq_len,
            "tactic_bearing_turns": tactic_bearing_turns,
            "max_tactic_prob": max_tactic_p,
            "lstm_risk": float(round(r, 6)),
            "decision": pred
        }
        
        if actual == 0 and pred == 1:
            fps.append(info)
        elif actual == 1 and pred == 0:
            fns.append(info)
            
    print(f"False Positives count: {len(fps)}")
    for fp in fps:
        print(f"  FP: {fp}")
        
    print(f"False Negatives count: {len(fns)}")
    for fn in fns:
        print(f"  FN: {fn}")
        
    # Save full audit results
    audit_data = {
        "contingency_table": contingency,
        "permutation_ablation": {
            "orig_f1": m_orig["f1"],
            "orig_roc": m_orig["roc_auc"],
            "orig_pr": m_orig["pr_auc"],
            "shuffled_f1_mean": float(np.mean(shuffled_f1s)),
            "shuffled_f1_std": float(np.std(shuffled_f1s)),
            "shuffled_roc_mean": float(np.mean(shuffled_rocs)),
            "shuffled_pr_mean": float(np.mean(shuffled_prs)),
        },
        "fps": fps,
        "fns": fns,
        "time_to_detection_counts": dict(counts)
    }
    
    with open(os.path.join(project_root, "data", "scientific_validity_audit.json"), "w") as f:
        json.dump(audit_data, f, indent=2)

if __name__ == "__main__":
    run_validity_audit()
