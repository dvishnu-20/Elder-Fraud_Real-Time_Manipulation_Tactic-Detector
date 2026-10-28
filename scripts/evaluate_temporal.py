import os
import sys
import json
import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.temporal_dataset import ConversationSequenceDataset, get_aggregated_features_for_lr
from ml.temporal_model import CausalLSTMRiskModel
from ml.temporal_baselines import (
    MajorityClassBaseline,
    TacticCountRuleBaseline,
    AggregatedLogisticRegressionBaseline,
    calculate_full_classification_metrics
)

def evaluate_temporal_system():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data", "processed", "temporal")
    model_dir = os.path.join(project_root, "ml", "models", "temporal_risk", "lstm_v1")
    
    train_path = os.path.join(data_dir, "train_seq.jsonl")
    val_path = os.path.join(data_dir, "val_seq.jsonl")
    test_path = os.path.join(data_dir, "test_seq.jsonl")
    
    print("=" * 70)
    print("PHASE 5: TEMPORAL RISK MODEL EVALUATION & REAL-TIME BENCHMARK")
    print("=" * 70)
    
    train_ds = ConversationSequenceDataset(train_path)
    val_ds = ConversationSequenceDataset(val_path)
    test_ds = ConversationSequenceDataset(test_path)
    
    # Load config and trained LSTM weights
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
        
    threshold = config["calibrated_threshold"]
    print(f"Loaded Causal LSTM Configuration from: {model_dir}")
    print(f"Calibrated Decision Threshold (Validation): {threshold}")
    
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
    
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # -------------------------------------------------------------
    # 1. TEST SET EVALUATION: BASELINES VS LSTM
    # -------------------------------------------------------------
    print("\n--- TEST SET EVALUATION (Full Conversations) ---")
    y_train = np.array([s["label"] for s in train_ds.samples])
    y_val = np.array([s["label"] for s in val_ds.samples])
    y_test = np.array([s["label"] for s in test_ds.samples])
    
    # A. Majority Class
    maj_model = MajorityClassBaseline()
    maj_model.fit(y_train)
    test_probs_maj = maj_model.predict_proba(len(y_test))
    test_m_maj = calculate_full_classification_metrics(y_test, test_probs_maj, threshold=0.5)
    
    # B. Rule / Tactic Count
    rule_model = TacticCountRuleBaseline(tactic_thresh=0.2)
    val_probs_rule = rule_model.predict_proba(val_ds)
    test_probs_rule = rule_model.predict_proba(test_ds)
    # Use validation-calibrated rule threshold
    rule_thresh = config["baseline_val_metrics"]["rule"]["threshold"]
    test_m_rule = calculate_full_classification_metrics(y_test, test_probs_rule, threshold=rule_thresh)
    
    # C. Logistic Regression
    X_train_agg, y_train_agg, _ = get_aggregated_features_for_lr(train_ds)
    X_val_agg, y_val_agg, _ = get_aggregated_features_for_lr(val_ds)
    X_test_agg, y_test_agg, _ = get_aggregated_features_for_lr(test_ds)
    
    lr_model = AggregatedLogisticRegressionBaseline(C=1.0, seed=config["seed"])
    lr_model.fit(X_train_agg, y_train_agg)
    test_probs_lr = lr_model.predict_proba(X_test_agg)
    lr_thresh = config["baseline_val_metrics"]["logistic_regression"]["threshold"]
    test_m_lr = calculate_full_classification_metrics(y_test, test_probs_lr, threshold=lr_thresh)
    
    # D. Causal LSTM
    all_lstm_probs = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["length"].to(device)
            logits = lstm_model(features, mask, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_lstm_probs.extend(probs)
    all_lstm_probs = np.array(all_lstm_probs)
    test_m_lstm = calculate_full_classification_metrics(y_test, all_lstm_probs, threshold=threshold)
    
    print(f"Baseline A (Majority):     PR-AUC={test_m_maj['pr_auc']:.3f}, ROC-AUC={test_m_maj['roc_auc']:.3f}, F1={test_m_maj['f1']:.3f}")
    print(f"Baseline B (Tactic Count): PR-AUC={test_m_rule['pr_auc']:.3f}, ROC-AUC={test_m_rule['roc_auc']:.3f}, F1={test_m_rule['f1']:.3f}")
    print(f"Baseline C (Logistic Reg): PR-AUC={test_m_lr['pr_auc']:.3f}, ROC-AUC={test_m_lr['roc_auc']:.3f}, F1={test_m_lr['f1']:.3f}")
    print(f"Causal LSTM (Phase 5):     PR-AUC={test_m_lstm['pr_auc']:.3f}, ROC-AUC={test_m_lstm['roc_auc']:.3f}, F1={test_m_lstm['f1']:.3f}")
    
    # -------------------------------------------------------------
    # 2. CAUSAL STEP-BY-STEP REAL-TIME SIMULATION (Prefix Evaluation)
    # -------------------------------------------------------------
    print("\n--- REAL-TIME CAUSAL PREFIX EVALUATION ---")
    prefix_checkpoints = [1, 2, 3, 5, 7]
    prefix_results = {}
    
    for k in prefix_checkpoints:
        prefix_probs = []
        prefix_labels = []
        
        for item in test_ds.samples:
            turn_probs = item["turn_probabilities"]
            if len(turn_probs) >= k:
                sub_probs = turn_probs[:k]
                sub_tensor = torch.tensor(np.array(sub_probs, dtype=np.float32)).unsqueeze(0).to(device) # (1, k, 5)
                sub_mask = torch.ones((1, k), dtype=torch.float32).to(device)
                sub_len = torch.tensor([k], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    logit = lstm_model(sub_tensor, sub_mask, sub_len)
                    r = float(torch.sigmoid(logit).item())
                prefix_probs.append(r)
                prefix_labels.append(item["label"])
                
        if prefix_labels:
            m_k = calculate_full_classification_metrics(np.array(prefix_labels), np.array(prefix_probs), threshold=threshold)
            prefix_results[f"turn_{k}"] = {
                "num_eval_conversations": len(prefix_labels),
                "metrics": m_k
            }
            print(f"  At Turn {k} (n={len(prefix_labels)}): PR-AUC={m_k['pr_auc']:.3f}, Precision={m_k['precision']:.3f}, Recall={m_k['recall']:.3f}, F1={m_k['f1']:.3f}")
            
    # -------------------------------------------------------------
    # 3. TIME-TO-DETECTION METRICS (Fraud Conversations)
    # -------------------------------------------------------------
    print("\n--- TIME-TO-DETECTION METRICS (Fraud Conversations) ---")
    fraud_samples = [s for s in test_ds.samples if s["label"] == 1]
    detection_turns = []
    detected_before_end = 0
    
    for item in fraud_samples:
        turn_probs = item["turn_probabilities"]
        detected_at = None
        
        # Step-by-step causal forward pass
        for t_idx in range(1, len(turn_probs) + 1):
            sub_probs = turn_probs[:t_idx]
            sub_tensor = torch.tensor(np.array(sub_probs, dtype=np.float32)).unsqueeze(0).to(device)
            sub_mask = torch.ones((1, t_idx), dtype=torch.float32).to(device)
            sub_len = torch.tensor([t_idx], dtype=torch.long).to(device)
            
            with torch.no_grad():
                r = float(torch.sigmoid(lstm_model(sub_tensor, sub_mask, sub_len)).item())
                
            if r >= threshold:
                detected_at = t_idx
                break
                
        if detected_at is not None:
            detection_turns.append(detected_at)
            detected_before_end += 1
            
    n_fraud = len(fraud_samples)
    mean_turn = float(np.mean(detection_turns)) if detection_turns else float("nan")
    median_turn = float(np.median(detection_turns)) if detection_turns else float("nan")
    det_by_2 = sum(1 for t in detection_turns if t <= 2) / n_fraud
    det_by_3 = sum(1 for t in detection_turns if t <= 3) / n_fraud
    det_by_5 = sum(1 for t in detection_turns if t <= 5) / n_fraud
    det_by_7 = sum(1 for t in detection_turns if t <= 7) / n_fraud
    det_before_end = detected_before_end / n_fraud
    
    time_to_detect = {
        "total_fraud_conversations": n_fraud,
        "total_detected": detected_before_end,
        "mean_detection_turn": mean_turn,
        "median_detection_turn": median_turn,
        "pct_detected_by_turn_2": det_by_2,
        "pct_detected_by_turn_3": det_by_3,
        "pct_detected_by_turn_5": det_by_5,
        "pct_detected_by_turn_7": det_by_7,
        "pct_detected_before_end": det_before_end
    }
    
    print(f"Total Fraud Dialogues: {n_fraud}")
    print(f"Mean Detection Turn:   {mean_turn:.2f}")
    print(f"Median Detection Turn: {median_turn:.1f}")
    print(f"% Detected by Turn 2:  {det_by_2*100:.1f}%")
    print(f"% Detected by Turn 3:  {det_by_3*100:.1f}%")
    print(f"% Detected by Turn 5:  {det_by_5*100:.1f}%")
    print(f"% Detected by Turn 7:  {det_by_7*100:.1f}%")
    print(f"% Detected Before End: {det_before_end*100:.1f}%")
    
    # -------------------------------------------------------------
    # 4. EXPLAINABILITY TRAJECTORIES (Sample Cases)
    # -------------------------------------------------------------
    explainability_cases = []
    for item in test_ds.samples[:8]:
        turn_probs = item["turn_probabilities"]
        turn_texts = item.get("turn_texts", [])
        turn_speakers = item.get("turn_speakers", [])
        turn_labels = item.get("turn_labels", [])
        
        trajectory = []
        for t_idx in range(1, len(turn_probs) + 1):
            sub_probs = turn_probs[:t_idx]
            sub_tensor = torch.tensor(np.array(sub_probs, dtype=np.float32)).unsqueeze(0).to(device)
            sub_mask = torch.ones((1, t_idx), dtype=torch.float32).to(device)
            sub_len = torch.tensor([t_idx], dtype=torch.long).to(device)
            
            with torch.no_grad():
                r = float(torch.sigmoid(lstm_model(sub_tensor, sub_mask, sub_len)).item())
                
            trajectory.append({
                "turn_index": t_idx,
                "speaker": turn_speakers[t_idx-1] if t_idx-1 < len(turn_speakers) else "unknown",
                "text": turn_texts[t_idx-1] if t_idx-1 < len(turn_texts) else "",
                "tactic_probabilities": {
                    "urgency": float(round(turn_probs[t_idx-1][0], 4)),
                    "isolation": float(round(turn_probs[t_idx-1][1], 4)),
                    "authority": float(round(turn_probs[t_idx-1][2], 4)),
                    "emotional": float(round(turn_probs[t_idx-1][3], 4)),
                    "channel": float(round(turn_probs[t_idx-1][4], 4))
                },
                "cumulative_fraud_risk": float(round(r, 4))
            })
            
        explainability_cases.append({
            "conversation_id": item["conversation_id"],
            "ground_truth_label": item["conversation_level_label"],
            "topic": item.get("topic", "unknown"),
            "final_predicted_risk": float(round(trajectory[-1]["cumulative_fraud_risk"], 4)),
            "trajectory": trajectory
        })
        
    full_report = {
        "test_metrics": {
            "causal_lstm": test_m_lstm,
            "baseline_majority": test_m_maj,
            "baseline_tactic_count": test_m_rule,
            "baseline_logistic_regression": test_m_lr
        },
        "prefix_detection_metrics": prefix_results,
        "time_to_detection": time_to_detect,
        "explainability_traces": explainability_cases
    }
    
    out_file = os.path.join(project_root, "data", "temporal_evaluation_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"\nSaved full temporal evaluation report to: {out_file}")

if __name__ == "__main__":
    evaluate_temporal_system()
