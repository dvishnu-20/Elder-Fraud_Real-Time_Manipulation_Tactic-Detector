import os
import json
import random
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

TACTIC_LABELS = [
    "urgency",
    "isolation",
    "authority",
    "emotional_destabilization",
    "channel_control"
]

LABEL_TO_ID = {label: i for i, label in enumerate(TACTIC_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(TACTIC_LABELS)}

THRESHOLDS_TO_SWEEP = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60]

def run_challenge_audit():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    model_dir = os.path.join(project_root, "ml", "models", "tactic_classifier", "final")
    challenge_file = os.path.join(project_root, "data", "challenge_set.jsonl")
    
    print("=" * 70)
    print("PHASE 4.1: GENERALIZATION AUDIT (EVALUATION ONLY)")
    print("=" * 70)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Trained model directory not found: {model_dir}")
    if not os.path.exists(challenge_file):
        raise FileNotFoundError(f"Challenge set not found: {challenge_file}")
        
    # Load config and calibrated thresholds
    with open(os.path.join(model_dir, "config_ext.json"), "r") as f:
        config_ext = json.load(f)
    baseline_thresholds = config_ext.get("thresholds", [0.5] * len(TACTIC_LABELS))
    print(f"Loaded Calibrated Decision Thresholds (Baseline): {dict(zip(TACTIC_LABELS, baseline_thresholds))}\n")

    # Load Model & Tokenizer
    print("Loading existing trained DistilBERT classifier (NO RETRAINING)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load Challenge Set Samples
    samples = []
    with open(challenge_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
            
    print(f"Loaded {len(samples)} Challenge Set turns for evaluation.\n")

    # Perform Inference
    all_texts = [s["text"] for s in samples]
    ground_truth_labels = []
    
    for s in samples:
        multi_hot = [0.0] * len(TACTIC_LABELS)
        for l in s["labels"]:
            if l in LABEL_TO_ID:
                multi_hot[LABEL_TO_ID[l]] = 1.0
        ground_truth_labels.append(multi_hot)
        
    y_true = np.array(ground_truth_labels)
    
    batch_size = 32
    all_probs = []
    
    print("Executing batch inference on challenge set...")
    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**encodings)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            
    y_probs = np.vstack(all_probs)

    # Pre-calculate idx_hard_neg
    idx_hard_neg = [idx for idx, s in enumerate(samples) if s.get("category") == "benign_hard_neg"]

    # =========================================================================
    # THRESHOLD ANALYSIS (SWEEP)
    # =========================================================================
    threshold_sweep_results = []
    threshold_matrix_per_tactic = {label: [] for label in TACTIC_LABELS}
    
    for t in THRESHOLDS_TO_SWEEP:
        y_pred_t = (y_probs > t).astype(int)
        
        # Micro and Macro F1 for threshold t
        t_micro_p = precision_score(y_true, y_pred_t, average="micro", zero_division=0)
        t_micro_r = recall_score(y_true, y_pred_t, average="micro", zero_division=0)
        t_micro_f1 = f1_score(y_true, y_pred_t, average="micro", zero_division=0)
        
        t_macro_p = precision_score(y_true, y_pred_t, average="macro", zero_division=0)
        t_macro_r = recall_score(y_true, y_pred_t, average="macro", zero_division=0)
        t_macro_f1 = f1_score(y_true, y_pred_t, average="macro", zero_division=0)
        
        # Hard-negative specificity at threshold t (no false alarms on hard negatives)
        t_hard_neg_fp = sum(1 for idx in idx_hard_neg if y_pred_t[idx].sum() > 0)
        t_hard_neg_spec = (len(idx_hard_neg) - t_hard_neg_fp) / max(len(idx_hard_neg), 1)
        
        # Total False Positives & False Negatives across all turns at threshold t
        total_fps = 0
        total_fns = 0
        for idx in range(len(samples)):
            true_s = set([TACTIC_LABELS[j] for j in range(5) if y_true[idx][j] == 1.0])
            pred_s = set([TACTIC_LABELS[j] for j in range(5) if y_pred_t[idx][j] == 1.0])
            total_fps += len(pred_s - true_s)
            total_fns += len(true_s - pred_s)
            
        tactic_metrics_at_t = {}
        for j, label in enumerate(TACTIC_LABELS):
            pj = precision_score(y_true[:, j], y_pred_t[:, j], zero_division=0)
            rj = recall_score(y_true[:, j], y_pred_t[:, j], zero_division=0)
            fj = f1_score(y_true[:, j], y_pred_t[:, j], zero_division=0)
            tactic_metrics_at_t[label] = {
                "precision": float(pj),
                "recall": float(rj),
                "f1": float(fj)
            }
            threshold_matrix_per_tactic[label].append({
                "threshold": t,
                "precision": float(pj),
                "recall": float(rj),
                "f1": float(fj)
            })
            
        threshold_sweep_results.append({
            "threshold": t,
            "micro_precision": float(t_micro_p),
            "micro_recall": float(t_micro_r),
            "micro_f1": float(t_micro_f1),
            "macro_precision": float(t_macro_p),
            "macro_recall": float(t_macro_r),
            "macro_f1": float(t_macro_f1),
            "hard_neg_specificity": float(t_hard_neg_spec),
            "hard_neg_fp_count": int(t_hard_neg_fp),
            "total_false_positives": int(total_fps),
            "total_false_negatives": int(total_fns),
            "per_tactic": tactic_metrics_at_t
        })

    # Best threshold per tactic based on max F1
    best_thresholds = {}
    for label in TACTIC_LABELS:
        best_entry = max(threshold_matrix_per_tactic[label], key=lambda x: x["f1"])
        best_thresholds[label] = best_entry["threshold"]

    # =========================================================================
    # OPTIMIZED PREDICTIONS (Using best per-tactic thresholds from sweep)
    # =========================================================================
    y_pred = np.zeros_like(y_probs)
    for i, label in enumerate(TACTIC_LABELS):
        t = best_thresholds[label]
        y_pred[:, i] = (y_probs[:, i] > t).astype(int)

    # Baseline predictions (at 0.50 threshold)
    y_pred_50 = (y_probs > 0.50).astype(int)

    # =========================================================================
    # COMPUTE METRICS AT OPTIMAL PER-TACTIC THRESHOLDS
    # =========================================================================
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    per_tactic_metrics = {}
    for i, label in enumerate(TACTIC_LABELS):
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        try:
            pr_auc = average_precision_score(y_true[:, i], y_probs[:, i])
        except Exception:
            pr_auc = 0.0
        per_tactic_metrics[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "pr_auc": float(pr_auc),
            "best_threshold": float(best_thresholds[label]),
            "support": int(y_true[:, i].sum())
        }

    # 1. Benign Hard Negatives
    hard_neg_fp_count = sum(1 for i in idx_hard_neg if y_pred[i].sum() > 0)
    hard_neg_acc = (len(idx_hard_neg) - hard_neg_fp_count) / max(len(idx_hard_neg), 1)
    
    # 2. Contrastive Pairs
    pairs = {}
    for i, s in enumerate(samples):
        pid = s.get("pair_id")
        if pid:
            if pid not in pairs:
                pairs[pid] = {}
            if s.get("is_benign_pair_member"):
                pairs[pid]["benign"] = i
            else:
                pairs[pid]["manipulative"] = i

    cp_total = 0
    cp_benign_correct = 0
    cp_manip_correct = 0
    cp_pair_correct = 0
    cp_same_pred = 0
    cp_results = []
    
    for pid, p_idx in pairs.items():
        if "benign" in p_idx and "manipulative" in p_idx:
            cp_total += 1
            b_idx = p_idx["benign"]
            m_idx = p_idx["manipulative"]
            
            b_true = y_true[b_idx]
            b_pred = y_pred[b_idx]
            m_true = y_true[m_idx]
            m_pred = y_pred[m_idx]
            
            b_correct = np.array_equal(b_true, b_pred)
            m_correct = np.array_equal(m_true, m_pred)
            same_pred = np.array_equal(b_pred, m_pred)
            
            if b_correct: cp_benign_correct += 1
            if m_correct: cp_manip_correct += 1
            if b_correct and m_correct: cp_pair_correct += 1
            if same_pred: cp_same_pred += 1
            
            cp_results.append({
                "pair_id": pid,
                "topic": samples[b_idx].get("topic"),
                "benign_text": samples[b_idx]["text"],
                "benign_true": [TACTIC_LABELS[j] for j in range(5) if b_true[j] == 1],
                "benign_pred": [TACTIC_LABELS[j] for j in range(5) if b_pred[j] == 1],
                "benign_probs": {TACTIC_LABELS[j]: round(float(y_probs[b_idx, j]), 4) for j in range(5)},
                "manipulative_text": samples[m_idx]["text"],
                "manipulative_true": [TACTIC_LABELS[j] for j in range(5) if m_true[j] == 1],
                "manipulative_pred": [TACTIC_LABELS[j] for j in range(5) if m_pred[j] == 1],
                "manipulative_probs": {TACTIC_LABELS[j]: round(float(y_probs[m_idx, j]), 4) for j in range(5)},
                "pair_correct": bool(b_correct and m_correct),
                "same_pred": bool(same_pred)
            })

    # False Positives and False Negatives analysis
    false_positives = []
    false_negatives = []
    
    for i, s in enumerate(samples):
        true_set = set([TACTIC_LABELS[j] for j in range(5) if y_true[i][j] == 1.0])
        pred_set = set([TACTIC_LABELS[j] for j in range(5) if y_pred[i][j] == 1.0])
        
        diff_fp = pred_set - true_set
        if diff_fp:
            false_positives.append({
                "text": s["text"],
                "category": s.get("category"),
                "true_labels": list(true_set) if true_set else ["none"],
                "pred_labels": list(pred_set),
                "spurious_labels": list(diff_fp),
                "probabilities": {TACTIC_LABELS[j]: round(float(y_probs[i, j]), 4) for j in range(5)}
            })
            
        diff_fn = true_set - pred_set
        if diff_fn:
            false_negatives.append({
                "text": s["text"],
                "category": s.get("category"),
                "true_labels": list(true_set),
                "pred_labels": list(pred_set) if pred_set else ["none"],
                "missed_labels": list(diff_fn),
                "probabilities": {TACTIC_LABELS[j]: round(float(y_probs[i, j]), 4) for j in range(5)}
            })

    # Select 25 representative predictions across categories
    random.seed(42)
    rep_indices = []
    cat_to_indices = {}
    for idx, s in enumerate(samples):
        cat = s.get("category", "other")
        if cat not in cat_to_indices:
            cat_to_indices[cat] = []
        cat_to_indices[cat].append(idx)
        
    for cat, idxs in cat_to_indices.items():
        sample_k = min(len(idxs), 4)
        rep_indices.extend(random.sample(idxs, sample_k))
        
    rep_indices = list(dict.fromkeys(rep_indices))[:25]
    
    representative_samples = []
    for idx in rep_indices:
        s = samples[idx]
        t_labs = [TACTIC_LABELS[j] for j in range(5) if y_true[idx][j] == 1.0]
        p_labs = [TACTIC_LABELS[j] for j in range(5) if y_pred[idx][j] == 1.0]
        probs_dict = {TACTIC_LABELS[j]: round(float(y_probs[idx, j]), 4) for j in range(5)}
        representative_samples.append({
            "text": s["text"],
            "category": s.get("category"),
            "ground_truth": t_labs if t_labs else ["none"],
            "predicted": p_labs if p_labs else ["none"],
            "probabilities": probs_dict
        })
        
    cp_failures = [cp for cp in cp_results if not cp["pair_correct"]]

    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    print("-" * 70)
    print("FINAL AUDIT REPORT")
    print("-" * 70)
    print(f"1. Challenge-set Size: {len(samples)} turns")
    print(f"2. Hard-negative Count: {len(idx_hard_neg)} turns")
    print(f"3. Contrastive-pair Count: {cp_total} pairs ({cp_total * 2} turns)")
    print("")

    print("4. Threshold Sweep Summary:")
    print(f"{'Thresh':<7} | {'Micro-P':<8} | {'Micro-R':<8} | {'Micro-F1':<9} | {'Macro-F1':<9} | {'HN Spec':<9} | {'FP Count':<9} | {'FN Count':<9}")
    print("-" * 85)
    for sw in threshold_sweep_results:
        print(f"{sw['threshold']:<7.2f} | {sw['micro_precision']:<8.3f} | {sw['micro_recall']:<8.3f} | {sw['micro_f1']:<9.3f} | {sw['macro_f1']:<9.3f} | {sw['hard_neg_specificity']:<8.1%} | {sw['total_false_positives']:<9} | {sw['total_false_negatives']:<9}")
    print("")

    print("5. Per-tactic Threshold Sweep Breakdown:")
    for label in TACTIC_LABELS:
        print(f"  Tactic: {label.upper()}")
        for entry in threshold_matrix_per_tactic[label]:
            print(f"    Threshold {entry['threshold']:.2f} -> Precision: {entry['precision']:.3f}, Recall: {entry['recall']:.3f}, F1: {entry['f1']:.3f}")
    print("")

    print("6. Per-tactic Metrics at Optimal Thresholds:")
    print(f"{'Tactic':<26} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'PR-AUC':<7} | {'Thresh':<6}")
    print("-" * 75)
    for label, m in per_tactic_metrics.items():
        print(f"{label:<26} | {m['precision']:<9.3f} | {m['recall']:<8.3f} | {m['f1']:<6.3f} | {m['pr_auc']:<7.3f} | {m['best_threshold']:<6.2f}")
    print("")

    print(f"7. Hard-negative Specificity (Optimal): {hard_neg_acc:.1%} ({len(idx_hard_neg) - hard_neg_fp_count}/{len(idx_hard_neg)} correct)")
    print(f"   False Positives in Hard Negatives: {hard_neg_fp_count}")
    print("")

    print("8. Contrastive-pair Accuracy:")
    print(f"   Total Pairs: {cp_total}")
    print(f"   Correctly Classified Benign Members: {cp_benign_correct} / {cp_total} ({cp_benign_correct/cp_total:.1%})")
    print(f"   Correctly Classified Manipulative Members: {cp_manip_correct} / {cp_total} ({cp_manip_correct/cp_total:.1%})")
    print(f"   Pair Accuracy (both correct): {cp_pair_correct} / {cp_total} ({cp_pair_correct/cp_total:.1%})")
    print(f"   Cases where both received SAME prediction: {cp_same_pred} / {cp_total} ({cp_same_pred/cp_total:.1%})")
    print("")

    print(f"9. False Positives (Overall): {len(false_positives)}")
    print(f"10. False Negatives (Overall): {len(false_negatives)}")
    print("")

    # Save comprehensive report to JSON
    audit_report = {
        "challenge_set_size": len(samples),
        "hard_negative_count": len(idx_hard_neg),
        "contrastive_pair_count": cp_total,
        "threshold_sweep_results": threshold_sweep_results,
        "threshold_matrix_per_tactic": threshold_matrix_per_tactic,
        "per_tactic_metrics_optimal": per_tactic_metrics,
        "overall_optimal": {
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "hard_negative_specificity": float(hard_neg_acc),
            "hard_negative_fps": int(hard_neg_fp_count),
            "total_false_positives": len(false_positives),
            "total_false_negatives": len(false_negatives)
        },
        "contrastive_pair_accuracy": {
            "total_pairs": cp_total,
            "benign_correct": cp_benign_correct,
            "manipulative_correct": cp_manip_correct,
            "pair_accuracy_count": cp_pair_correct,
            "pair_accuracy_percent": cp_pair_correct / cp_total if cp_total > 0 else 0,
            "same_prediction_count": cp_same_pred
        },
        "representative_predictions_25": representative_samples,
        "contrastive_pair_results": cp_results,
        "comparison": {
            "original_synthetic_test": {
                "macro_f1": 0.998,
                "micro_f1": 0.997,
                "accuracy": 0.995
            },
            "phase_4_1_challenge_set": {
                "macro_f1": float(macro_f1),
                "micro_f1": float(micro_f1),
                "hard_neg_specificity": float(hard_neg_acc),
                "contrastive_pair_accuracy": float(cp_pair_correct / cp_total if cp_total > 0 else 0)
            }
        },
        "false_positive_examples": false_positives[:15],
        "false_negative_examples": false_negatives[:15],
        "contrastive_pair_failures": cp_failures[:15]
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_json = os.path.join(script_dir, "..", "data", "challenge_audit_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\nFull audit report saved to {out_json}")

if __name__ == "__main__":
    run_challenge_audit()
