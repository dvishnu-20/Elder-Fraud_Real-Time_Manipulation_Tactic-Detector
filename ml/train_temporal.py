import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.temporal_dataset import ConversationSequenceDataset, get_aggregated_features_for_lr
from ml.temporal_model import CausalLSTMRiskModel
from ml.temporal_baselines import (
    MajorityClassBaseline,
    TacticCountRuleBaseline,
    AggregatedLogisticRegressionBaseline,
    calculate_full_classification_metrics
)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(features, mask, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_probs)

def calibrate_risk_threshold(y_true, y_prob):
    # Sweep thresholds from 0.10 to 0.90
    best_thresh = 0.5
    best_f1 = 0.0
    
    for thresh in np.arange(0.10, 0.95, 0.05):
        m = calculate_full_classification_metrics(y_true, y_prob, threshold=thresh)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = float(round(thresh, 3))
            
    return best_thresh

def train_temporal():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, default="final", choices=["final", "masked_mean"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data", "processed", "temporal")
    save_dir = os.path.join(project_root, "ml", "models", "temporal_risk", "lstm_v1")
    os.makedirs(save_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train_seq.jsonl")
    val_path = os.path.join(data_dir, "val_seq.jsonl")
    test_path = os.path.join(data_dir, "test_seq.jsonl")
    
    print("=" * 60)
    print("PHASE 5: TRAINING CAUSAL TEMPORAL RISK MODEL")
    print("=" * 60)
    print(f"Loading sequence datasets from: {data_dir}")
    
    train_ds = ConversationSequenceDataset(train_path)
    val_ds = ConversationSequenceDataset(val_path)
    test_ds = ConversationSequenceDataset(test_path)
    
    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # -------------------------------------------------------------
    # 1. EVALUATE BASELINES ON VALIDATION
    # -------------------------------------------------------------
    print("\n--- BASELINE EVALUATIONS (Validation Set) ---")
    
    # A. Majority Class
    y_train = np.array([s["label"] for s in train_ds.samples])
    y_val = np.array([s["label"] for s in val_ds.samples])
    
    maj_model = MajorityClassBaseline()
    maj_model.fit(y_train)
    val_probs_maj = maj_model.predict_proba(len(y_val))
    val_m_maj = calculate_full_classification_metrics(y_val, val_probs_maj, threshold=0.5)
    print(f"Baseline A (Majority Class): ROC-AUC={val_m_maj['roc_auc']:.3f}, PR-AUC={val_m_maj['pr_auc']:.3f}, F1={val_m_maj['f1']:.3f}")
    
    # B. Tactic Count Rule
    rule_model = TacticCountRuleBaseline(tactic_thresh=0.2)
    val_probs_rule = rule_model.predict_proba(val_ds)
    rule_thresh = calibrate_risk_threshold(y_val, val_probs_rule)
    val_m_rule = calculate_full_classification_metrics(y_val, val_probs_rule, threshold=rule_thresh)
    print(f"Baseline B (Tactic Count Rule): ROC-AUC={val_m_rule['roc_auc']:.3f}, PR-AUC={val_m_rule['pr_auc']:.3f}, F1={val_m_rule['f1']:.3f} (Thresh={rule_thresh})")
    
    # C. Logistic Regression on Aggregated Features
    X_train_agg, y_train_agg, _ = get_aggregated_features_for_lr(train_ds)
    X_val_agg, y_val_agg, _ = get_aggregated_features_for_lr(val_ds)
    
    lr_model = AggregatedLogisticRegressionBaseline(C=1.0, seed=args.seed)
    lr_model.fit(X_train_agg, y_train_agg)
    val_probs_lr = lr_model.predict_proba(X_val_agg)
    lr_thresh = calibrate_risk_threshold(y_val, val_probs_lr)
    val_m_lr = calculate_full_classification_metrics(y_val, val_probs_lr, threshold=lr_thresh)
    print(f"Baseline C (Logistic Regression): ROC-AUC={val_m_lr['roc_auc']:.3f}, PR-AUC={val_m_lr['pr_auc']:.3f}, F1={val_m_lr['f1']:.3f} (Thresh={lr_thresh})")
    
    # -------------------------------------------------------------
    # 2. TRAIN CAUSAL LSTM
    # -------------------------------------------------------------
    print(f"\n--- TRAINING CAUSAL LSTM (Pooling={args.pooling}, Hidden={args.hidden_dim}, Layers={args.num_layers}) ---")
    model = CausalLSTMRiskModel(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling_head=args.pooling
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    start_time = time.time()
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(features, mask, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(labels)
            
        train_loss = total_loss / len(train_ds)
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                mask = batch["mask"].to(device)
                lengths = batch["length"].to(device)
                labels = batch["label"].to(device)
                logits = model(features, mask, lengths)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
        val_loss /= len(val_ds)
        
        # Validation metrics
        val_true, val_probs = evaluate_model(model, val_loader, device)
        val_f1 = calculate_full_classification_metrics(val_true, val_probs, threshold=0.5)["f1"]
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1 (0.5): {val_f1:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val Loss {best_val_loss:.4f}")
                break
                
    train_duration = time.time() - start_time
    print(f"Training finished in {train_duration:.2f} seconds.")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    
    # Calibrate risk decision threshold on validation set
    val_true, val_probs = evaluate_model(model, val_loader, device)
    calibrated_threshold = calibrate_risk_threshold(val_true, val_probs)
    val_metrics = calculate_full_classification_metrics(val_true, val_probs, threshold=calibrated_threshold)
    
    print("\n--- VALIDATION CALIBRATION RESULTS (Causal LSTM) ---")
    print(f"Selected Threshold: {calibrated_threshold}")
    print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.3f}")
    print(f"Validation PR-AUC:  {val_metrics['pr_auc']:.3f}")
    print(f"Validation F1:      {val_metrics['f1']:.3f}")
    print(f"Validation Precision: {val_metrics['precision']:.3f}")
    print(f"Validation Recall:    {val_metrics['recall']:.3f}")
    
    # Save model weights as model.pt and best_model.pt
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    
    # Save model metadata and config
    config = {
        "model_type": "CausalLSTMRiskModel",
        "input_dim": 5,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pooling_head": args.pooling,
        "calibrated_threshold": calibrated_threshold,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "training_duration_seconds": train_duration,
        "best_epoch": best_epoch,
        "validation_metrics": val_metrics,
        "baseline_val_metrics": {
            "majority": val_m_maj,
            "rule": val_m_rule,
            "logistic_regression": val_m_lr
        }
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    threshold_config = {
        "calibrated_threshold": calibrated_threshold,
        "calibration_dataset": "data/processed/temporal/val_seq.jsonl",
        "optimization_criteria": "Max F1 on Validation Set",
        "sweep_range": [0.10, 0.95, 0.05]
    }
    with open(os.path.join(save_dir, "threshold_config.json"), "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)
        
    training_metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_duration_seconds": train_duration,
        "validation_metrics": val_metrics
    }
    with open(os.path.join(save_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)
        
    feature_def = {
        "input_dim": 5,
        "feature_order": [
            "urgency",
            "isolation",
            "authority",
            "emotional_destabilization",
            "channel_control"
        ],
        "feature_type": "continuous_probabilities",
        "value_range": [0.0, 1.0],
        "source_classifier": "ml/models/tactic_classifier/retrained_v2 (FROZEN)",
        "causal_guarantee": "No future turns used; unidirectional timestep processing"
    }
    with open(os.path.join(save_dir, "feature_definition.json"), "w", encoding="utf-8") as f:
        json.dump(feature_def, f, indent=2)
        
    print(f"\nAll model weights and metadata JSON files successfully saved to: {save_dir}")

if __name__ == "__main__":
    train_temporal()
