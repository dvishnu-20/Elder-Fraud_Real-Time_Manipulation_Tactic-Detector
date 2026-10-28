import os
import sys
import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TACTIC_LABELS = [
    "urgency",
    "isolation",
    "authority",
    "emotional_destabilization",
    "channel_control"
]
LABEL_TO_ID = {label: i for i, label in enumerate(TACTIC_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(TACTIC_LABELS)}

def evaluate_model_on_challenge(model_dir, challenge_file, model_name_tag="Model"):
    print(f"\n=======================================================")
    print(f"EVALUATING: {model_name_tag}")
    print(f"Path: {model_dir}")
    print(f"=======================================================")
    
    # Load config and validation-calibrated thresholds
    config_ext_path = os.path.join(model_dir, "config_ext.json")
    if os.path.exists(config_ext_path):
        with open(config_ext_path, "r") as f:
            config_ext = json.load(f)
        thresholds = config_ext.get("thresholds", [0.5] * len(TACTIC_LABELS))
    else:
        thresholds = [0.5] * len(TACTIC_LABELS)
        
    print(f"Validation-Calibrated Thresholds: {dict(zip(TACTIC_LABELS, thresholds))}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    samples = []
    with open(challenge_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
                
    all_texts = [s["text"] for s in samples]
    all_labels = []
    for s in samples:
        m = [0.0] * len(TACTIC_LABELS)
        for lbl in s.get("labels", []):
            if lbl in LABEL_TO_ID:
                m[LABEL_TO_ID[lbl]] = 1.0
        all_labels.append(m)
    all_labels = np.array(all_labels)
    
    # Batch inference
    batch_size = 16
    all_probs = []
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    
    # Apply thresholds
    bin_preds = np.zeros_like(all_probs)
    for i, t in enumerate(thresholds):
        bin_preds[:, i] = (all_probs[:, i] > t).astype(int)
        
    # Overall metrics
    micro_p = precision_score(all_labels, bin_preds, average='micro', zero_division=0)
    micro_r = recall_score(all_labels, bin_preds, average='micro', zero_division=0)
    micro_f1 = f1_score(all_labels, bin_preds, average='micro', zero_division=0)
    macro_p = precision_score(all_labels, bin_preds, average='macro', zero_division=0)
    macro_r = recall_score(all_labels, bin_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, bin_preds, average='macro', zero_division=0)
    
    total_fp = int(((bin_preds == 1) & (all_labels == 0)).sum())
    total_fn = int(((bin_preds == 0) & (all_labels == 1)).sum())
    
    # Hard negatives evaluation
    hard_neg_indices = [i for i, s in enumerate(samples) if s.get("category") == "benign_hard_negative" or s.get("difficulty") == "hard_negative" or len(s.get("labels", [])) == 0 or s.get("labels") == ["none"]]
    hard_neg_correct = 0
    hard_neg_fp = 0
    for idx in hard_neg_indices:
        if bin_preds[idx].sum() == 0:
            hard_neg_correct += 1
        else:
            hard_neg_fp += 1
    hn_specificity = (hard_neg_correct / len(hard_neg_indices)) if hard_neg_indices else 0.0
    
    # Per-tactic metrics
    per_tactic = {}
    for i, tac in enumerate(TACTIC_LABELS):
        p = precision_score(all_labels[:, i], bin_preds[:, i], zero_division=0)
        r = recall_score(all_labels[:, i], bin_preds[:, i], zero_division=0)
        f = f1_score(all_labels[:, i], bin_preds[:, i], zero_division=0)
        try:
            pr_auc = average_precision_score(all_labels[:, i], all_probs[:, i])
        except Exception:
            pr_auc = 0.0
        per_tactic[tac] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "pr_auc": float(pr_auc),
            "threshold": float(thresholds[i]),
            "support": int(all_labels[:, i].sum())
        }
        
    # Contrastive pairs evaluation
    pair_map = {}
    for i, s in enumerate(samples):
        pid = s.get("pair_id")
        if pid:
            if pid not in pair_map:
                pair_map[pid] = {}
            if "benign" in s.get("category", "") or len(s.get("labels", [])) == 0 or s.get("labels") == ["none"]:
                pair_map[pid]["benign"] = {
                    "idx": i,
                    "text": s["text"],
                    "true": s.get("labels", []),
                    "pred": [ID_TO_LABEL[j] for j in range(5) if bin_preds[i, j] == 1],
                    "probs": {ID_TO_LABEL[j]: float(round(all_probs[i, j], 4)) for j in range(5)},
                    "correct": (bin_preds[i].sum() == 0)
                }
            else:
                true_tacs = [lbl for lbl in s.get("labels", []) if lbl in LABEL_TO_ID]
                pred_tacs = [ID_TO_LABEL[j] for j in range(5) if bin_preds[i, j] == 1]
                pair_map[pid]["manipulative"] = {
                    "idx": i,
                    "text": s["text"],
                    "true": true_tacs,
                    "pred": pred_tacs,
                    "probs": {ID_TO_LABEL[j]: float(round(all_probs[i, j], 4)) for j in range(5)},
                    "correct": (set(true_tacs) == set(pred_tacs))
                }
                
    valid_pairs = [p for p in pair_map.values() if "benign" in p and "manipulative" in p]
    total_pairs = len(valid_pairs)
    benign_correct = sum(1 for p in valid_pairs if p["benign"]["correct"])
    manip_correct = sum(1 for p in valid_pairs if p["manipulative"]["correct"])
    pair_accuracy = sum(1 for p in valid_pairs if p["benign"]["correct"] and p["manipulative"]["correct"])
    
    results = {
        "model_tag": model_name_tag,
        "model_dir": model_dir,
        "thresholds": thresholds,
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "hard_neg_specificity": float(hn_specificity),
        "hard_neg_total": len(hard_neg_indices),
        "hard_neg_correct": hard_neg_correct,
        "hard_neg_fp": hard_neg_fp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_tactic": per_tactic,
        "contrastive_pairs": {
            "total_pairs": total_pairs,
            "benign_correct": benign_correct,
            "benign_accuracy": (benign_correct / total_pairs) if total_pairs else 0.0,
            "manipulative_correct": manip_correct,
            "manipulative_accuracy": (manip_correct / total_pairs) if total_pairs else 0.0,
            "pair_accuracy_count": pair_accuracy,
            "pair_accuracy": (pair_accuracy / total_pairs) if total_pairs else 0.0
        },
        "samples_analysis": {
            "samples": samples,
            "probs": all_probs,
            "preds": bin_preds,
            "labels": all_labels
        }
    }
    
    print(f"Micro F1: {micro_f1:.3f} | Macro F1: {macro_f1:.3f}")
    print(f"Micro Precision: {micro_p:.3f} | Micro Recall: {micro_r:.3f}")
    print(f"Hard Negative Specificity: {hn_specificity*100:.1f}% ({hard_neg_correct}/{len(hard_neg_indices)})")
    print(f"Contrastive Pair Accuracy: {results['contrastive_pairs']['pair_accuracy']*100:.1f}% (Benign: {benign_correct}/{total_pairs}, Manip: {manip_correct}/{total_pairs})")
    
    return results

def run_comparison():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    old_model_dir = os.path.join(project_root, "ml", "models", "tactic_classifier", "final")
    new_model_dir = os.path.join(project_root, "ml", "models", "tactic_classifier", "retrained_v2")
    challenge_file = os.path.join(project_root, "data", "challenge_set.jsonl")
    
    old_results = evaluate_model_on_challenge(old_model_dir, challenge_file, "OLD BASELINE (Phase 4.1)")
    new_results = evaluate_model_on_challenge(new_model_dir, challenge_file, "NEW RETRAINED V2 (Phase 4.3)")
    
    # Detailed Error Analysis on NEW Model
    new_samples = new_results["samples_analysis"]["samples"]
    new_probs = new_results["samples_analysis"]["probs"]
    new_preds = new_results["samples_analysis"]["preds"]
    new_labels = new_results["samples_analysis"]["labels"]
    
    fps = []
    fns = []
    tps = []
    tns = []
    
    for i, s in enumerate(new_samples):
        true_tacs = [lbl for lbl in s.get("labels", []) if lbl in LABEL_TO_ID]
        pred_tacs = [ID_TO_LABEL[j] for j in range(5) if new_preds[i, j] == 1]
        probs_dict = {ID_TO_LABEL[j]: float(round(new_probs[i, j], 4)) for j in range(5)}
        
        is_benign = len(true_tacs) == 0
        is_pred_benign = len(pred_tacs) == 0
        
        entry = {
            "text": s["text"],
            "category": s.get("category", "standard"),
            "ground_truth": true_tacs if true_tacs else ["none"],
            "prediction": pred_tacs if pred_tacs else ["none"],
            "probabilities": probs_dict,
        }
        
        # True Negatives (Correctly rejected benign)
        if is_benign and is_pred_benign and len(tns) < 10:
            entry["diagnosis"] = "Correctly rejected benign text with legitimate domain vocabulary."
            entry["category_type"] = "true_negative"
            tns.append(entry)
            
        # True Positives (Correctly detected manipulation)
        if not is_benign and set(true_tacs) == set(pred_tacs) and len(tps) < 10:
            entry["diagnosis"] = "Correctly identified manipulative behavior and exact tactic set."
            entry["category_type"] = "true_positive"
            tps.append(entry)
            
        # False Positives (Benign text with predicted tactics or extra tactics)
        if is_benign and not is_pred_benign and len(fps) < 10:
            entry["diagnosis"] = "Lexical shortcut / Over-sensitivity to benign domain vocabulary."
            entry["category_type"] = "lexical_shortcut"
            fps.append(entry)
            
        # False Negatives (Missed tactic)
        if not is_benign and len(set(true_tacs) - set(pred_tacs)) > 0 and len(fns) < 10:
            entry["diagnosis"] = "Missed subtle/coercive manipulation tactic."
            entry["category_type"] = "semantic_confusion"
            fns.append(entry)
            
    # Clean up samples_analysis from saved JSON to keep it concise
    del old_results["samples_analysis"]
    del new_results["samples_analysis"]
    
    comparison_report = {
        "old_baseline": old_results,
        "new_retrained_v2": new_results,
        "error_analysis_10_samples": {
            "10_false_positives": fps,
            "10_false_negatives": fns,
            "10_true_positives": tps,
            "10_true_negatives": tns
        }
    }
    
    out_path = os.path.join(project_root, "data", "model_comparison_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
        
    print(f"\nSaved full comparative report to: {out_path}")
    
    # Print side-by-side comparison summary
    print("\n" + "="*70)
    print("PHASE 4.3 GENERALIZATION BENCHMARK: OLD VS NEW COMPARISON")
    print("="*70)
    print(f"{'Metric':<32} | {'OLD (final)':<15} | {'NEW (retrained_v2)':<18}")
    print("-" * 70)
    print(f"{'Macro F1':<32} | {old_results['macro_f1']:<15.3f} | {new_results['macro_f1']:<18.3f}")
    print(f"{'Micro F1':<32} | {old_results['micro_f1']:<15.3f} | {new_results['micro_f1']:<18.3f}")
    print(f"{'Micro Precision':<32} | {old_results['micro_precision']:<15.3f} | {new_results['micro_precision']:<18.3f}")
    print(f"{'Micro Recall':<32} | {old_results['micro_recall']:<15.3f} | {new_results['micro_recall']:<18.3f}")
    print(f"{'Hard-Negative Specificity':<32} | {old_results['hard_neg_specificity']*100:<14.1f}% | {new_results['hard_neg_specificity']*100:<17.1f}%")
    print(f"{'Hard-Negative False Positives':<32} | {old_results['hard_neg_fp']:<15} | {new_results['hard_neg_fp']:<18}")
    print(f"{'Total False Positives':<32} | {old_results['total_fp']:<15} | {new_results['total_fp']:<18}")
    print(f"{'Total False Negatives':<32} | {old_results['total_fn']:<15} | {new_results['total_fn']:<18}")
    print(f"{'Benign Contrastive Accuracy':<32} | {old_results['contrastive_pairs']['benign_accuracy']*100:<14.1f}% | {new_results['contrastive_pairs']['benign_accuracy']*100:<17.1f}%")
    print(f"{'Manipulative Contrastive Acc':<32} | {old_results['contrastive_pairs']['manipulative_accuracy']*100:<14.1f}% | {new_results['contrastive_pairs']['manipulative_accuracy']*100:<17.1f}%")
    print(f"{'Complete Pair Accuracy':<32} | {old_results['contrastive_pairs']['pair_accuracy']*100:<14.1f}% | {new_results['contrastive_pairs']['pair_accuracy']*100:<17.1f}%")
    print("="*70)

if __name__ == "__main__":
    run_comparison()
