import os
import sys
import json
import torch
import numpy as np
import argparse
from datetime import datetime

# Optimize CPU parallelism
cpu_count = os.cpu_count() or 4
torch.set_num_threads(min(16, cpu_count))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch import nn
from transformers import Trainer, TrainingArguments, set_seed
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
from ml.dataset import TacticDataset, TACTIC_LABELS, LABEL_TO_ID
from ml.model import get_model_and_tokenizer

class ImbalancedMultiLabelTrainer(Trainer):
    def __init__(self, pos_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights
        if self.pos_weights is not None:
            self.pos_weights = self.pos_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_class_weights(dataset):
    labels = np.array([item['labels'] for item in dataset.samples])
    pos_counts = labels.sum(axis=0)
    total = len(labels)
    neg_counts = total - pos_counts
    
    pos_weights = []
    for p, n in zip(pos_counts, neg_counts):
        if p == 0:
            pos_weights.append(1.0)
        else:
            pos_weights.append(float(n / p))
    return torch.tensor(pos_weights, dtype=torch.float)

def calibrate_thresholds(trainer, val_dataset):
    print("\n--- CALIBRATING DECISION THRESHOLDS ON V2 VALIDATION SET ---")
    preds_output = trainer.predict(val_dataset)
    logits = preds_output.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = preds_output.label_ids
    
    threshold_options = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_thresholds = [0.5] * len(TACTIC_LABELS)
    val_metrics = {}
    
    for i, tactic in enumerate(TACTIC_LABELS):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in threshold_options:
            pred_bin = (probs[:, i] > thresh).astype(int)
            f1 = f1_score(labels[:, i], pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        best_thresholds[i] = round(float(best_thresh), 3)
        try:
            prauc = float(average_precision_score(labels[:, i], probs[:, i]))
        except Exception:
            prauc = 0.0
        p = float(precision_score(labels[:, i], (probs[:, i] > best_thresholds[i]).astype(int), zero_division=0))
        r = float(recall_score(labels[:, i], (probs[:, i] > best_thresholds[i]).astype(int), zero_division=0))
        
        val_metrics[tactic] = {
            "best_threshold": best_thresholds[i],
            "val_precision": round(p, 4),
            "val_recall": round(r, 4),
            "val_f1": round(float(best_f1), 4),
            "val_pr_auc": round(prauc, 4)
        }
        print(f"  {tactic:<25}: Best Thresh={best_thresholds[i]:<5} (Val P={p:.4f}, R={r:.4f}, F1={best_f1:.4f}, PR-AUC={prauc:.4f})")
    
    return best_thresholds, val_metrics

def evaluate_subset_detailed(name, indices, probs, labels, thresholds):
    if not indices:
        return {"name": name, "count": 0, "micro_f1": 0.0, "micro_prec": 0.0, "micro_rec": 0.0}
    p = probs[indices]
    l = labels[indices]
    
    preds = np.zeros_like(p)
    for i, t in enumerate(thresholds):
        preds[:, i] = (p[:, i] > t).astype(int)
        
    micro_f1 = float(f1_score(l, preds, average='micro', zero_division=0))
    micro_p = float(precision_score(l, preds, average='micro', zero_division=0))
    micro_r = float(recall_score(l, preds, average='micro', zero_division=0))
    print(f"  {name:<45} (n={len(indices):<4}): Micro F1 = {micro_f1:.4f} | Prec = {micro_p:.4f} | Rec = {micro_r:.4f}")
    return {
        "name": name,
        "count": len(indices),
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4)
    }

def run_test_evaluation(trainer, test_dataset, thresholds, val_metrics, out_dir):
    print("\n" + "=" * 80)
    print("V2 DISTILBERT TACTIC CLASSIFIER: TEST SET EVALUATION REPORT")
    print("=" * 80)
    
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = preds_output.label_ids
    
    preds = np.zeros_like(probs)
    for i, t in enumerate(thresholds):
        preds[:, i] = (probs[:, i] > t).astype(int)
        
    micro_p = float(precision_score(labels, preds, average='micro', zero_division=0))
    micro_r = float(recall_score(labels, preds, average='micro', zero_division=0))
    micro_f1 = float(f1_score(labels, preds, average='micro', zero_division=0))
    
    macro_p = float(precision_score(labels, preds, average='macro', zero_division=0))
    macro_r = float(recall_score(labels, preds, average='macro', zero_division=0))
    macro_f1 = float(f1_score(labels, preds, average='macro', zero_division=0))
    
    print("\n--- 1. OVERALL TEST METRICS ---")
    print(f"Micro Precision: {micro_p:.4f}")
    print(f"Micro Recall:    {micro_r:.4f}")
    print(f"Micro F1 Score:  {micro_f1:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Macro F1 Score:  {macro_f1:.4f}")
    
    # 2. Per-Tactic Metrics
    print("\n--- 2. PER-TACTIC TEST METRICS ---")
    print(f"{'Tactic':<25} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'PR-AUC':<10} | {'Threshold':<10}")
    print("-" * 85)
    per_tactic_results = {}
    for i, tactic in enumerate(TACTIC_LABELS):
        p = float(precision_score(labels[:, i], preds[:, i], zero_division=0))
        r = float(recall_score(labels[:, i], preds[:, i], zero_division=0))
        f = float(f1_score(labels[:, i], preds[:, i], zero_division=0))
        try:
            pr_auc = float(average_precision_score(labels[:, i], probs[:, i]))
        except Exception:
            pr_auc = 0.0
        print(f"{tactic:<25} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f} | {pr_auc:<10.4f} | {thresholds[i]:<10.3f}")
        per_tactic_results[tactic] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "pr_auc": round(pr_auc, 4),
            "threshold": thresholds[i]
        }
        
    # 3. Sub-population Evaluation
    print("\n--- 3. SUB-POPULATION TEST EVALUATION ---")
    samples = test_dataset.get_metadata()
    
    # Subsets
    idx_benign_tactic = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'benign' and any(l in TACTIC_LABELS for l in s['raw_labels'])]
    idx_fraud_tactic = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'fraud' and any(l in TACTIC_LABELS for l in s['raw_labels'])]
    idx_benign_hard_neg = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'benign' and "none" in s['raw_labels'] and s.get('speaker') == 'caller']
    
    idx_legit_auth = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'benign' and "authority" in s['raw_labels']]
    idx_legit_urg = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'benign' and "urgency" in s['raw_labels']]
    idx_legit_emo = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'benign' and "emotional_destabilization" in s['raw_labels']]
    idx_legit_priv_chn = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'benign' and any(l in ["isolation", "channel_control"] for l in s['raw_labels'])]
    idx_manipulative = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'fraud' and any(l in TACTIC_LABELS for l in s['raw_labels'])]
    idx_multi_label = [i for i, s in enumerate(samples) if len([l for l in s['raw_labels'] if l in TACTIC_LABELS]) > 1]
    
    sub_results = []
    sub_results.append(evaluate_subset_detailed("1. Benign Tactic-Bearing Turns", idx_benign_tactic, probs, labels, thresholds))
    sub_results.append(evaluate_subset_detailed("2. Fraud Tactic-Bearing Turns", idx_fraud_tactic, probs, labels, thresholds))
    sub_results.append(evaluate_subset_detailed("3. Benign Hard Negatives (Caller None)", idx_benign_hard_neg, probs, labels, thresholds))
    sub_results.append(evaluate_subset_detailed("4. Legitimate Authority (Benign)", idx_legit_auth, probs, labels, thresholds))
    sub_results.append(evaluate_subset_detailed("5. Legitimate Urgency (Benign)", idx_legit_urg, probs, labels, thresholds))
    sub_results.append(evaluate_subset_detailed("6. Legitimate Emotional Language (Benign)", idx_legit_emo, probs, labels, thresholds))
    sub_results.append(evaluate_subset_detailed("7. Legitimate Privacy/Channel (Benign)", idx_legit_priv_chn, probs, labels, thresholds))
    sub_results.append(evaluate_subset_detailed("8. Manipulative Tactic Turns (Fraud)", idx_manipulative, probs, labels, thresholds))
    sub_results.append(evaluate_subset_detailed("9. Multi-label Turns", idx_multi_label, probs, labels, thresholds))
    
    # Representative Error Analysis
    print("\n--- 4. ERROR ANALYSIS EXAMPLES ---")
    fp_examples = []
    fn_examples = []
    for i, s in enumerate(samples):
        true_lbls = [TACTIC_LABELS[j] for j in range(5) if labels[i][j] == 1.0]
        pred_lbls = [TACTIC_LABELS[j] for j in range(5) if preds[i][j] == 1.0]
        
        if set(pred_lbls) != set(true_lbls):
            record = {
                "text": s["text"],
                "speaker": s["speaker"],
                "conversation_level_label": s["conversation_level_label"],
                "true_labels": true_lbls,
                "pred_labels": pred_lbls,
                "probabilities": {TACTIC_LABELS[j]: round(float(probs[i][j]), 4) for j in range(5)}
            }
            if len(pred_lbls) > len(true_lbls) and len(fp_examples) < 3:
                fp_examples.append(record)
                print(f"[OVER-PREDICTION] Text: '{s['text']}'\n  True: {true_lbls} | Pred: {pred_lbls} | Context: {s['conversation_level_label']}")
            elif len(pred_lbls) < len(true_lbls) and len(fn_examples) < 3:
                fn_examples.append(record)
                print(f"[UNDER-PREDICTION] Text: '{s['text']}'\n  True: {true_lbls} | Pred: {pred_lbls} | Context: {s['conversation_level_label']}")

    full_report = {
        "overall_test_metrics": {
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4)
        },
        "per_tactic_test_metrics": per_tactic_results,
        "sub_population_evaluations": sub_results,
        "validation_metrics": val_metrics,
        "error_examples": {
            "false_positives": fp_examples,
            "false_negatives": fn_examples
        }
    }
    
    with open(os.path.join(out_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    with open("data/v2_distilbert_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"\nSaved evaluation reports to:")
    print(f"  {os.path.join(out_dir, 'training_metrics.json')}")
    print(f"  data/v2_distilbert_evaluation_report.json")
    
    return full_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=48)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()

    set_seed(args.seed)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, "ml", "models", "tactic_classifier", "v2_temporal_validity")
    os.makedirs(out_dir, exist_ok=True)
    
    train_path = os.path.join(project_root, "data", "processed", "temporal_v2", "train.jsonl")
    val_path = os.path.join(project_root, "data", "processed", "temporal_v2", "validation.jsonl")
    test_path = os.path.join(project_root, "data", "processed", "temporal_v2", "test.jsonl")
    
    print("=" * 80)
    print("PHASE 3.5 / V2: TRAINING NEW TURN-LEVEL DISTILBERT TACTIC CLASSIFIER")
    print("=" * 80)
    print(f"Output Directory: {out_dir}")
    print(f"Train Dataset:    {train_path}")
    print(f"Validation:       {val_path}")
    print(f"Test Dataset:     {test_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device} (CPU threads={torch.get_num_threads()})")
    
    model, tokenizer = get_model_and_tokenizer(args.model_name, num_labels=5)
    
    # Freeze lower layers (embeddings + layers 0..4) to accelerate CPU backprop
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
    for layer in model.distilbert.transformer.layer[:5]:
        for param in layer.parameters():
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    model.to(device)
    
    print("\nLoading datasets...")
    train_dataset = TacticDataset(train_path, tokenizer, max_length=args.max_length)
    val_dataset = TacticDataset(val_path, tokenizer, max_length=args.max_length)
    test_dataset = TacticDataset(test_path, tokenizer, max_length=args.max_length)
    
    pos_weights = compute_class_weights(train_dataset)
    print(f"Calculated BCE Class Weights from V2 Train: {pos_weights.numpy()}")
    
    chk_dir = os.path.join(project_root, "ml", "models", "tactic_classifier_v2_checkpoints")
    training_args = TrainingArguments(
        output_dir=chk_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        logging_steps=50,
        dataloader_num_workers=0,
        disable_tqdm=False
    )
    
    trainer = ImbalancedMultiLabelTrainer(
        pos_weights=pos_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    print("\nStarting DistilBERT V2 Training...")
    train_result = trainer.train()
    print("Training completed.")
    
    # Threshold Calibration on Validation
    thresholds, val_metrics = calibrate_thresholds(trainer, val_dataset)
    
    # Save Model & Tokenizer
    print(f"\nSaving final model and artifacts to: {out_dir}")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    # Save threshold config
    threshold_config = {
        "labels": TACTIC_LABELS,
        "thresholds": thresholds,
        "threshold_mapping": {TACTIC_LABELS[i]: thresholds[i] for i in range(len(TACTIC_LABELS))},
        "calibration_dataset": "data/processed/temporal_v2/validation.jsonl",
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(out_dir, "threshold_config.json"), "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)
        
    # Save feature definition
    feature_def = {
        "feature_dim": 5,
        "feature_names": TACTIC_LABELS,
        "description": "5-dimensional continuous DistilBERT probabilities for multi-label manipulation tactics",
        "model_architecture": "DistilBERT for Sequence Classification (BCEWithLogitsLoss)",
        "dataset_version": "temporal_v2_repaired"
    }
    with open(os.path.join(out_dir, "feature_definition.json"), "w", encoding="utf-8") as f:
        json.dump(feature_def, f, indent=2)
        
    # Run test evaluation
    run_test_evaluation(trainer, test_dataset, thresholds, val_metrics, out_dir)
    
    print("\n" + "=" * 80)
    print("V2 DISTILBERT TACTIC CLASSIFIER TRAINING & AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
