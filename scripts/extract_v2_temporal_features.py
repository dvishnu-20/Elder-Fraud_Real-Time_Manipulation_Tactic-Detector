import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

# Set PyTorch threads for CPU inference
cpu_count = os.cpu_count() or 4
torch.set_num_threads(min(16, cpu_count))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer, AutoModelForSequenceClassification

TACTIC_LABELS = [
    "urgency",
    "isolation",
    "authority",
    "emotional_destabilization",
    "channel_control"
]

def extract_and_audit_v2_features():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(project_root, "ml", "models", "tactic_classifier", "v2_temporal_validity")
    raw_v2_dir = os.path.join(project_root, "data", "processed", "temporal_v2")
    out_dir = os.path.join(project_root, "data", "processed", "temporal_v2_features")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("V2 TEMPORAL FEATURE EXTRACTION & STATISTICAL AUDIT")
    print("=" * 80)
    print(f"Model Directory: {model_dir}")
    print(f"Source Dataset:  {raw_v2_dir}")
    print(f"Output Directory: {out_dir}")
    
    # -------------------------------------------------------------
    # 1. LOAD AND VERIFY FROZEN V2 DISTILBERT MODEL
    # -------------------------------------------------------------
    with open(os.path.join(model_dir, "threshold_config.json"), "r") as f:
        threshold_cfg = json.load(f)
    thresholds = threshold_cfg["thresholds"]
    thresh_map = threshold_cfg["threshold_mapping"]
    
    print(f"\nLoaded V2 Threshold Configuration: {thresh_map}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device} (threads={torch.get_num_threads()})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Verify model is strictly frozen
    for p in model.parameters():
        p.requires_grad = False
    assert not any(p.requires_grad for p in model.parameters()), "Model parameters must have requires_grad=False"
    print("Verification: Model parameters frozen (requires_grad=False) and model.eval() active.")
    
    # -------------------------------------------------------------
    # 2. LOAD SPLITS AND VERIFY DISJOINT IDS
    # -------------------------------------------------------------
    splits = ["train", "validation", "test"]
    split_files = {
        "train": os.path.join(raw_v2_dir, "train.jsonl"),
        "validation": os.path.join(raw_v2_dir, "validation.jsonl"),
        "test": os.path.join(raw_v2_dir, "test.jsonl")
    }
    
    split_conv_turns = {}
    split_cids = {}
    
    for s in splits:
        conv_turns = defaultdict(list)
        with open(split_files[s], "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    cid = item["conversation_id"]
                    conv_turns[cid].append(item)
        split_conv_turns[s] = conv_turns
        split_cids[s] = set(conv_turns.keys())
        
    print("\n--- SPLIT INTEGRITY VERIFICATION ---")
    print(f"Train:      {len(split_cids['train'])} conversations, {sum(len(t) for t in split_conv_turns['train'].values())} turns")
    print(f"Validation: {len(split_cids['validation'])} conversations, {sum(len(t) for t in split_conv_turns['validation'].values())} turns")
    print(f"Test:       {len(split_cids['test'])} conversations, {sum(len(t) for t in split_conv_turns['test'].values())} turns")
    
    overlap_tv = len(split_cids["train"].intersection(split_cids["validation"]))
    overlap_tt = len(split_cids["train"].intersection(split_cids["test"]))
    overlap_vt = len(split_cids["validation"].intersection(split_cids["test"]))
    
    print(f"ID Overlaps: Train-Val={overlap_tv}, Train-Test={overlap_tt}, Val-Test={overlap_vt}")
    assert overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0, "Conversation IDs must be strictly disjoint across splits!"
    
    # -------------------------------------------------------------
    # 3. EXTRACT 5-DIMENSIONAL CONTINUOUS PROBABILITY SEQUENCES
    # -------------------------------------------------------------
    out_files = {
        "train": os.path.join(out_dir, "train_features.jsonl"),
        "validation": os.path.join(out_dir, "validation_features.jsonl"),
        "test": os.path.join(out_dir, "test_features.jsonl")
    }
    
    extracted_datasets = {}
    
    for s in splits:
        print(f"\nExtracting features for split: {s.upper()} ({len(split_conv_turns[s])} conversations)...")
        records = []
        writer = open(out_files[s], "w", encoding="utf-8")
        
        for cid, turns in split_conv_turns[s].items():
            # Sort turns strictly by turn_id to preserve chronology
            sorted_turns = sorted(turns, key=lambda x: int(x["turn_id"]))
            
            cat = sorted_turns[0].get("conversation_level_label", "benign")
            label = 1 if cat == "fraud" else 0
            topic = sorted_turns[0].get("source", "unknown")
            
            turn_texts = [t["text"] for t in sorted_turns]
            turn_speakers = [t.get("speaker", "unknown") for t in sorted_turns]
            turn_labels = [t.get("labels", ["none"]) for t in sorted_turns]
            
            # Batch inference on conversation turns
            enc = tokenizer(turn_texts, truncation=True, padding=True, max_length=64, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                continuous_features = [[float(round(p, 5)) for p in row] for row in probs]
                
            rec = {
                "conversation_id": cid,
                "label": label,
                "conversation_level_label": cat,
                "topic": topic,
                "num_turns": len(sorted_turns),
                "features": continuous_features,
                "turn_texts": turn_texts,
                "turn_speakers": turn_speakers,
                "turn_labels": turn_labels
            }
            writer.write(json.dumps(rec) + "\n")
            records.append(rec)
            
        writer.close()
        extracted_datasets[s] = records
        print(f"  Saved {len(records)} sequence records to {out_files[s]}")

    # -------------------------------------------------------------
    # 4. SEQUENCE QUALITY AUDIT
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("4. SEQUENCE QUALITY AUDIT")
    print("=" * 50)
    
    quality_report = {}
    for s in splits:
        recs = extracted_datasets[s]
        lengths = [r["num_turns"] for r in recs]
        fraud_recs = [r for r in recs if r["label"] == 1]
        benign_recs = [r for r in recs if r["label"] == 0]
        
        nan_count = 0
        range_error_count = 0
        empty_count = 0
        
        for r in recs:
            feat = np.array(r["features"])
            if len(feat) == 0:
                empty_count += 1
            if np.isnan(feat).any():
                nan_count += 1
            if (feat < 0.0).any() or (feat > 1.0).any():
                range_error_count += 1
                
        pct_ge_6 = sum(1 for l in lengths if l >= 6) / len(lengths) * 100.0
        pct_ge_10 = sum(1 for l in lengths if l >= 10) / len(lengths) * 100.0
        
        s_stats = {
            "total_sequences": len(recs),
            "fraud_sequences": len(fraud_recs),
            "benign_sequences": len(benign_recs),
            "total_turns": sum(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "mean_length": round(float(np.mean(lengths)), 2),
            "median_length": float(np.median(lengths)),
            "pct_ge_6": round(pct_ge_6, 2),
            "pct_ge_10": round(pct_ge_10, 2),
            "empty_sequences": empty_count,
            "nan_features": nan_count,
            "out_of_bounds_features": range_error_count
        }
        quality_report[s] = s_stats
        
        print(f"\nSplit: {s.upper()}")
        print(f"  Sequences: {len(recs)} (Fraud: {len(fraud_recs)}, Benign: {len(benign_recs)})")
        print(f"  Turns: {sum(lengths)} | Lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}, median={np.median(lengths):.1f}")
        print(f"  % Length >= 6: {pct_ge_6:.2f}% | % Length >= 10: {pct_ge_10:.2f}%")
        print(f"  Empty: {empty_count} | NaNs: {nan_count} | Out-of-bounds: {range_error_count}")

    # -------------------------------------------------------------
    # 5. FEATURE DISTRIBUTION AUDIT ACROSS PROBABILITY DIMENSIONS
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("5. FEATURE DISTRIBUTION AUDIT (CONTINUOUS 5-TACTIC PROBABILITIES)")
    print("=" * 50)
    
    distribution_report = {}
    
    for s in splits:
        recs = extracted_datasets[s]
        all_feats = np.concatenate([np.array(r["features"]) for r in recs], axis=0) # (Total turns, 5)
        
        fraud_turns_feat = []
        benign_turns_feat = []
        for r in recs:
            for t_idx, f_vec in enumerate(r["features"]):
                if r["label"] == 1:
                    fraud_turns_feat.append(f_vec)
                else:
                    benign_turns_feat.append(f_vec)
        fraud_turns_feat = np.array(fraud_turns_feat)
        benign_turns_feat = np.array(benign_turns_feat)
        
        s_dist = {}
        print(f"\n--- Feature Stats: {s.upper()} Split (N = {len(all_feats)} turns) ---")
        print(f"{'Tactic Dimension':<25} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Median':<8} | {'95th %':<8} | {'Max':<8}")
        print("-" * 80)
        
        for j, tac in enumerate(TACTIC_LABELS):
            col = all_feats[:, j]
            mean_v = float(np.mean(col))
            std_v = float(np.std(col))
            min_v = float(np.min(col))
            max_v = float(np.max(col))
            med_v = float(np.median(col))
            p95_v = float(np.percentile(col, 95))
            
            f_mean = float(np.mean(fraud_turns_feat[:, j]))
            b_mean = float(np.mean(benign_turns_feat[:, j]))
            
            s_dist[tac] = {
                "overall_mean": round(mean_v, 4),
                "std": round(std_v, 4),
                "min": round(min_v, 4),
                "max": round(max_v, 4),
                "median": round(med_v, 4),
                "p95": round(p95_v, 4),
                "fraud_turns_mean": round(f_mean, 4),
                "benign_turns_mean": round(b_mean, 4)
            }
            print(f"{tac:<25} | {mean_v:<8.4f} | {std_v:<8.4f} | {min_v:<8.4f} | {med_v:<8.4f} | {p95_v:<8.4f} | {max_v:<8.4f}")
            
        distribution_report[s] = s_dist

    # -------------------------------------------------------------
    # 6. TEMPORAL STRUCTURE AUDIT (Gaps, Progression, First Tactic)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("6. TEMPORAL STRUCTURE AUDIT (PACING, GAPS & FIRST DETECTION)")
    print("=" * 50)
    
    temporal_structure_report = {}
    
    for s in splits:
        recs = extracted_datasets[s]
        fraud_recs = [r for r in recs if r["label"] == 1]
        benign_recs = [r for r in recs if r["label"] == 0]
        
        def compute_temporal_dynamics(record_list):
            first_tactic_turns = []
            tactic_bearing_counts = []
            normal_before_first_counts = []
            longest_gaps = []
            multi_gaps_count = 0
            
            for r in record_list:
                feat = np.array(r["features"]) # (T, 5)
                # Tactic turn definition: any probability > calibrated threshold
                has_tactic = np.array([(feat[t] > thresholds).any() for t in range(len(feat))])
                t_count = int(has_tactic.sum())
                tactic_bearing_counts.append(t_count)
                
                t_indices = np.where(has_tactic)[0] + 1 # 1-indexed turn numbers
                if len(t_indices) > 0:
                    first_t = int(t_indices[0])
                    first_tactic_turns.append(first_t)
                    normal_before_first_counts.append(first_t - 1)
                else:
                    first_tactic_turns.append(None)
                    normal_before_first_counts.append(len(feat))
                    
                if len(t_indices) >= 2:
                    gaps = [int(t_indices[i] - t_indices[i-1] - 1) for i in range(1, len(t_indices))]
                    longest_gaps.append(max(gaps))
                    if any(g >= 2 for g in gaps):
                        multi_gaps_count += 1
                else:
                    longest_gaps.append(0)
                    
            return {
                "tactic_bearing_counts": tactic_bearing_counts,
                "first_tactic_turns": [t for t in first_tactic_turns if t is not None],
                "normal_before_first": normal_before_first_counts,
                "longest_gaps": longest_gaps,
                "multi_gaps_count": multi_gaps_count,
                "total_conversations": len(record_list)
            }
            
        f_dyn = compute_temporal_dynamics(fraud_recs)
        b_dyn = compute_temporal_dynamics(benign_recs)
        
        print(f"\n--- Temporal Dynamics ({s.upper()} Split) ---")
        print(f"Fraud Conversations (N={len(fraud_recs)}):")
        print(f"  Mean tactic-bearing turns: {np.mean(f_dyn['tactic_bearing_counts']):.2f}")
        print(f"  Mean turn of first tactic: {np.mean(f_dyn['first_tactic_turns']):.2f} (min={min(f_dyn['first_tactic_turns'])}, max={max(f_dyn['first_tactic_turns'])})")
        print(f"  Convs with >= 2 normal turns before first tactic: {sum(1 for n in f_dyn['normal_before_first'] if n >= 2)} ({sum(1 for n in f_dyn['normal_before_first'] if n >= 2)/len(fraud_recs)*100:.1f}%)")
        print(f"  Convs with multi-turn normal gaps (gap >= 2 turns): {f_dyn['multi_gaps_count']} ({f_dyn['multi_gaps_count']/len(fraud_recs)*100:.1f}%)")
        
        print(f"Benign Conversations (N={len(benign_recs)}):")
        print(f"  Mean tactic-bearing turns: {np.mean(b_dyn['tactic_bearing_counts']):.2f}")
        print(f"  Convs with >= 1 tactic activation: {sum(1 for c in b_dyn['tactic_bearing_counts'] if c >= 1)} ({sum(1 for c in b_dyn['tactic_bearing_counts'] if c >= 1)/len(benign_recs)*100:.1f}%)")
        print(f"  Convs with >= 2 tactic activations: {sum(1 for c in b_dyn['tactic_bearing_counts'] if c >= 2)} ({sum(1 for c in b_dyn['tactic_bearing_counts'] if c >= 2)/len(benign_recs)*100:.1f}%)")
        
        temporal_structure_report[s] = {
            "fraud": {
                "mean_tactic_turns": round(float(np.mean(f_dyn["tactic_bearing_counts"])), 2),
                "mean_first_tactic_turn": round(float(np.mean(f_dyn["first_tactic_turns"])), 2),
                "pct_normal_openers_ge_2": round(sum(1 for n in f_dyn["normal_before_first"] if n >= 2) / len(fraud_recs) * 100, 2),
                "pct_multi_turn_gaps": round(f_dyn["multi_gaps_count"] / len(fraud_recs) * 100, 2)
            },
            "benign": {
                "mean_tactic_turns": round(float(np.mean(b_dyn["tactic_bearing_counts"])), 2),
                "pct_ge_1_tactic": round(sum(1 for c in b_dyn["tactic_bearing_counts"] if c >= 1) / len(benign_recs) * 100, 2),
                "pct_ge_2_tactics": round(sum(1 for c in b_dyn["tactic_bearing_counts"] if c >= 2) / len(benign_recs) * 100, 2)
            }
        }

    # -------------------------------------------------------------
    # 7. DESCRIPTIVE BASELINE SHORTCUT AUDIT
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("7. DESCRIPTIVE BASELINE AUDIT (SHORTCUT RESISTANCE)")
    print("=" * 50)
    
    # Evaluate descriptive metrics on validation and test sets
    test_recs = extracted_datasets["test"]
    y_test = np.array([r["label"] for r in test_recs])
    
    # A. Any tactic detected
    any_tactic_pred = np.array([int((np.array(r["features"]) > thresholds).any()) for r in test_recs])
    p_fraud_any = float(y_test[any_tactic_pred == 1].mean()) if any_tactic_pred.sum() > 0 else 0.0
    fp_any = int(((y_test == 0) & (any_tactic_pred == 1)).sum())
    
    # B. Count of activations
    t_counts = np.array([int((np.array(r["features"]) > thresholds).sum()) for r in test_recs])
    
    # C. Maximum tactic probability across dialogue
    max_probs = np.array([float(np.array(r["features"]).max()) for r in test_recs])
    
    # D. Mean tactic probability
    mean_probs = np.array([float(np.array(r["features"]).mean()) for r in test_recs])
    
    print(f"Descriptive Shortcut Check on Test Set (N = {len(test_recs)} convs, 150 Fraud / 150 Benign):")
    print(f"  A. P(Fraud | Any Tactic Predicted): {p_fraud_any:.4f} (Benign FPs on any tactic = {fp_any}/150 = {fp_any/150*100:.1f}%)")
    print(f"  B. Mean tactic activations: Fraud = {t_counts[y_test==1].mean():.2f}, Benign = {t_counts[y_test==0].mean():.2f}")
    print(f"  C. Max probability:         Fraud = {max_probs[y_test==1].mean():.4f}, Benign = {max_probs[y_test==0].mean():.4f}")
    print(f"  D. Mean probability:        Fraud = {mean_probs[y_test==1].mean():.4f}, Benign = {mean_probs[y_test==0].mean():.4f}")

    # -------------------------------------------------------------
    # 8. SAVE FEATURE MANIFEST
    # -------------------------------------------------------------
    manifest = {
        "dataset_version": "temporal_v2_features",
        "parent_dataset": "data/processed/temporal_v2/",
        "frozen_distilbert_model": model_dir,
        "feature_dim": 5,
        "labels_order": TACTIC_LABELS,
        "calibrated_thresholds": thresh_map,
        "seed": 42,
        "extraction_timestamp": datetime.now().isoformat(),
        "splits": {
            "train": {
                "file": "train_features.jsonl",
                "conversations": quality_report["train"]["total_sequences"],
                "fraud": quality_report["train"]["fraud_sequences"],
                "benign": quality_report["train"]["benign_sequences"],
                "turns": quality_report["train"]["total_turns"]
            },
            "validation": {
                "file": "validation_features.jsonl",
                "conversations": quality_report["validation"]["total_sequences"],
                "fraud": quality_report["validation"]["fraud_sequences"],
                "benign": quality_report["validation"]["benign_sequences"],
                "turns": quality_report["validation"]["total_turns"]
            },
            "test": {
                "file": "test_features.jsonl",
                "conversations": quality_report["test"]["total_sequences"],
                "fraud": quality_report["test"]["fraud_sequences"],
                "benign": quality_report["test"]["benign_sequences"],
                "turns": quality_report["test"]["total_turns"]
            }
        },
        "quality_metrics": quality_report,
        "feature_distributions": distribution_report,
        "temporal_structure": temporal_structure_report
    }
    
    manifest_path = os.path.join(out_dir, "feature_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nSuccessfully wrote feature manifest to: {manifest_path}")

if __name__ == "__main__":
    extract_and_audit_v2_features()
