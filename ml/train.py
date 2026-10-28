import os
import sys
import json
import torch
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch import nn
from transformers import Trainer, TrainingArguments, set_seed
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
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
        if p == 0: pos_weights.append(1.0)
        else: pos_weights.append(n / p)
    return torch.tensor(pos_weights, dtype=torch.float)

def calibrate_thresholds(trainer, val_dataset):
    print("Calibrating thresholds on validation set...")
    preds_output = trainer.predict(val_dataset)
    logits = preds_output.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = preds_output.label_ids
    
    threshold_options = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
    best_thresholds = [0.5] * len(TACTIC_LABELS)
    for i in range(len(TACTIC_LABELS)):
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
            prauc = average_precision_score(labels[:, i], probs[:, i])
        except Exception:
            prauc = 0.0
        p = precision_score(labels[:, i], (probs[:, i] > best_thresholds[i]).astype(int), zero_division=0)
        r = recall_score(labels[:, i], (probs[:, i] > best_thresholds[i]).astype(int), zero_division=0)
        print(f"  {TACTIC_LABELS[i]}: Best Thresh={best_thresholds[i]} (Val P={p:.3f}, R={r:.3f}, F1={best_f1:.3f}, PR-AUC={prauc:.3f})")
    
    return best_thresholds

def evaluate_subset(name, indices, probs, labels, thresholds):
    if not indices: return
    p = probs[indices]
    l = labels[indices]
    
    preds = np.zeros_like(p)
    for i, t in enumerate(thresholds):
        preds[:, i] = (p[:, i] > t).astype(int)
        
    micro_f1 = f1_score(l, preds, average='micro', zero_division=0)
    print(f"  {name} (n={len(indices)}): Micro F1 = {micro_f1:.3f}")

def print_evaluation_report(trainer, test_dataset, thresholds, args):
    print("\n" + "="*50)
    print("PHASE 4: FINAL EVALUATION REPORT")
    print("="*50)
    
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = preds_output.label_ids
    
    preds = np.zeros_like(probs)
    for i, t in enumerate(thresholds):
        preds[:, i] = (probs[:, i] > t).astype(int)
        
    # 1. Overall Metrics
    print("\n--- OVERALL METRICS ---")
    print(f"Micro F1: {f1_score(labels, preds, average='micro', zero_division=0):.3f}")
    print(f"Macro F1: {f1_score(labels, preds, average='macro', zero_division=0):.3f}")
    print(f"Micro Precision: {precision_score(labels, preds, average='micro', zero_division=0):.3f}")
    print(f"Micro Recall: {recall_score(labels, preds, average='micro', zero_division=0):.3f}")
    
    # 2. Per-Tactic Metrics
    print("\n--- PER-TACTIC METRICS ---")
    for i, label in enumerate(TACTIC_LABELS):
        p = precision_score(labels[:, i], preds[:, i], zero_division=0)
        r = recall_score(labels[:, i], preds[:, i], zero_division=0)
        f = f1_score(labels[:, i], preds[:, i], zero_division=0)
        # Handle PR-AUC edge case where only 1 class exists in test slice
        try:
            pr_auc = average_precision_score(labels[:, i], probs[:, i])
        except:
            pr_auc = 0.0
        print(f"[{label.upper()}] P: {p:.3f} | R: {r:.3f} | F1: {f:.3f} | PR-AUC: {pr_auc:.3f} | Thresh: {thresholds[i]}")
        
    # 3. Sub-population Evaluation
    print("\n--- SUB-POPULATION EVALUATION ---")
    samples = test_dataset.get_metadata()
    
    idx_benign_hard = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'benign' and "none" in s['raw_labels'] and any(w in s['text'].lower() for w in ['bank', 'money', 'urgent'])]
    idx_benign_tactic = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'benign' and "none" not in s['raw_labels']]
    idx_fraud_tactic = [i for i, s in enumerate(samples) if s['conversation_level_label'] == 'fraud' and "none" not in s['raw_labels']]
    idx_multi = [i for i, s in enumerate(samples) if len([l for l in s['raw_labels'] if l != "none"]) > 1]
    idx_indirect = [i for i, s in enumerate(samples) if s['difficulty'] == 'hard']
    
    evaluate_subset("Benign Hard Negatives (Topics but no tactics)", idx_benign_hard, probs, labels, thresholds)
    evaluate_subset("Benign Tactic-Containing Turns (e.g. Doctor)", idx_benign_tactic, probs, labels, thresholds)
    evaluate_subset("Fraud Tactic Turns", idx_fraud_tactic, probs, labels, thresholds)
    evaluate_subset("Multi-label Turns", idx_multi, probs, labels, thresholds)
    evaluate_subset("Indirect/Hard Manipulation", idx_indirect, probs, labels, thresholds)
    
    print("\n--- TOPIC EVALUATION ---")
    topics = set(s['source'] for s in samples)
    for topic in topics:
        idx_topic = [i for i, s in enumerate(samples) if s['source'] == topic]
        evaluate_subset(f"Topic: {topic}", idx_topic, probs, labels, thresholds)
        
    # 4. Error Analysis (Representative Examples)
    print("\n--- ERROR ANALYSIS EXAMPLES ---")
    fp_found, fn_found, multi_miss_found, benign_mistake_found = False, False, False, False
    
    for i, s in enumerate(samples):
        true_labels = [TACTIC_LABELS[j] for j in range(5) if labels[i][j] == 1.0]
        pred_labels = [TACTIC_LABELS[j] for j in range(5) if preds[i][j] == 1.0]
        
        # False Positive
        if not fp_found and len(pred_labels) > len(true_labels) and len(true_labels) == 0:
            print(f"[FALSE POSITIVE] Text: '{s['text']}'\n  True: None | Pred: {pred_labels}")
            fp_found = True
            
        # False Negative
        if not fn_found and len(true_labels) > 0 and len(pred_labels) == 0:
            print(f"[FALSE NEGATIVE] Text: '{s['text']}'\n  True: {true_labels} | Pred: None")
            fn_found = True
            
        # Benign mistake
        if not benign_mistake_found and s['conversation_level_label'] == 'benign' and len(pred_labels) > len(true_labels):
            print(f"[BENIGN OVER-PREDICTION] Text: '{s['text']}'\n  True: {true_labels} | Pred: {pred_labels}")
            benign_mistake_found = True
            
        # Multi miss
        if not multi_miss_found and len(true_labels) > 1 and set(pred_labels) != set(true_labels):
            print(f"[MULTI-LABEL MISS] Text: '{s['text']}'\n  True: {true_labels} | Pred: {pred_labels}")
            multi_miss_found = True
            
    # Configuration metadata
    print("\n--- CONFIGURATION & REPRODUCIBILITY ---")
    print(f"Seed: {args.seed}")
    print(f"Model: {args.model_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Max Sequence Length: 128")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4, help="Set to 2 if CUDA OOM occurs")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--fp16", type=bool, default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print("\n--- GPU MEMORY STATUS (BEFORE TRAINING) ---")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Reserved VRAM: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print("-------------------------------------------\n")

    model, tokenizer = get_model_and_tokenizer(args.model_name, num_labels=5)
    model.to(device)

    print("Loading datasets...")
    train_dataset = TacticDataset("data/processed/train.jsonl", tokenizer)
    val_dataset = TacticDataset("data/processed/validation.jsonl", tokenizer)
    test_dataset = TacticDataset("data/processed/test.jsonl", tokenizer)
    
    pos_weights = compute_class_weights(train_dataset)
    print(f"Calculated BCE Class Weights to handle imbalance: {pos_weights.numpy()}")

    use_fp16 = args.fp16 and (device == "cuda")

    training_args = TrainingArguments(
        output_dir="ml/models/tactic_classifier_retrained_v2_checkpoints",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        logging_steps=50,
        disable_tqdm=False
    )

    trainer = ImbalancedMultiLabelTrainer(
        pos_weights=pos_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("Starting training...")
    trainer.train()
    
    if device == "cuda":
        print("\n--- GPU MEMORY STATUS (AFTER TRAINING) ---")
        print(f"Peak VRAM Allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        print(f"Peak VRAM Reserved: {torch.cuda.max_memory_reserved(0) / 1e9:.2f} GB")
        print("------------------------------------------\n")

    # Calibrate
    thresholds = calibrate_thresholds(trainer, val_dataset)

    # Save finalized configuration
    os.makedirs("ml/models/tactic_classifier/retrained_v2", exist_ok=True)
    trainer.save_model("ml/models/tactic_classifier/retrained_v2")
    tokenizer.save_pretrained("ml/models/tactic_classifier/retrained_v2")
    
    config = {
        "thresholds": thresholds,
        "labels": TACTIC_LABELS,
        "seed": args.seed,
        "dataset_version": "Phase4_Retrained"
    }
    with open("ml/models/tactic_classifier/retrained_v2/config_ext.json", "w") as f:
        json.dump(config, f, indent=2)

    # Evaluate
    print_evaluation_report(trainer, test_dataset, thresholds, args)

if __name__ == "__main__":
    main()
