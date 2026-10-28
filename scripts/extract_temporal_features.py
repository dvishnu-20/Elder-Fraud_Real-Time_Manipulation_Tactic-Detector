import os
import sys
import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TACTIC_LABELS = [
    "urgency",
    "isolation",
    "authority",
    "emotional_destabilization",
    "channel_control"
]

def extract_temporal_sequences():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    model_dir = os.path.join(project_root, "ml", "models", "tactic_classifier", "retrained_v2")
    raw_file = os.path.join(project_root, "data", "raw", "synthetic_conversations.jsonl")
    out_dir = os.path.join(project_root, "data", "processed", "temporal")
    os.makedirs(out_dir, exist_ok=True)
    
    # Identify conversation IDs in each split
    split_conv_ids = {"train": set(), "val": set(), "test": set()}
    with open(os.path.join(project_root, "data", "processed", "train.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                split_conv_ids["train"].add(json.loads(line)["conversation_id"])
                
    with open(os.path.join(project_root, "data", "processed", "validation.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                split_conv_ids["val"].add(json.loads(line)["conversation_id"])
                
    with open(os.path.join(project_root, "data", "processed", "test.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                split_conv_ids["test"].add(json.loads(line)["conversation_id"])
                
    print(f"Loaded split definitions:")
    print(f"  Train Convs: {len(split_conv_ids['train'])}")
    print(f"  Validation Convs: {len(split_conv_ids['val'])}")
    print(f"  Test Convs: {len(split_conv_ids['test'])}")
    
    # Load frozen DistilBERT
    print(f"\nLoading frozen DistilBERT model from: {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Load all conversations
    conversations = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line.strip()))
                
    print(f"Loaded {len(conversations)} conversations from {raw_file}.")
    
    # Process each conversation into probability sequence
    split_writers = {
        "train": open(os.path.join(out_dir, "train_seq.jsonl"), "w", encoding="utf-8"),
        "val": open(os.path.join(out_dir, "val_seq.jsonl"), "w", encoding="utf-8"),
        "test": open(os.path.join(out_dir, "test_seq.jsonl"), "w", encoding="utf-8")
    }
    
    counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    
    print("\nExtracting turn-level 5-tactic continuous probability sequences...")
    for idx, conv in enumerate(conversations):
        cid = conv["conversation_id"]
        cat = conv.get("conversation_level_label", "benign")
        topic = conv.get("topic", "general")
        turns = conv.get("turns", [])
        
        # Determine split
        if cid in split_conv_ids["train"]:
            split_key = "train"
        elif cid in split_conv_ids["val"]:
            split_key = "val"
        elif cid in split_conv_ids["test"]:
            split_key = "test"
        else:
            split_key = "unknown"
            continue
            
        turn_texts = [t["text"] for t in turns]
        turn_speakers = [t.get("speaker", "unknown") for t in turns]
        turn_labels = [t.get("labels", ["none"]) for t in turns]
        
        # Batch inference on turns of this conversation
        if turn_texts:
            enc = tokenizer(turn_texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                turn_probs = [[float(round(p, 5)) for p in row] for row in probs]
        else:
            turn_probs = []
            
        seq_record = {
            "conversation_id": cid,
            "topic": topic,
            "label": 1 if cat == "fraud" else 0,
            "conversation_level_label": cat,
            "num_turns": len(turns),
            "turn_probabilities": turn_probs,
            "turn_texts": turn_texts,
            "turn_speakers": turn_speakers,
            "turn_labels": turn_labels
        }
        
        split_writers[split_key].write(json.dumps(seq_record) + "\n")
        counts[split_key] += 1
        
        if (idx + 1) % 200 == 0 or (idx + 1) == len(conversations):
            print(f"  Processed {idx + 1}/{len(conversations)} conversations...")
            
    for f in split_writers.values():
        f.close()
        
    print("\nExtraction complete. Sequence datasets saved:")
    print(f"  Train sequences: {counts['train']} -> {os.path.join(out_dir, 'train_seq.jsonl')}")
    print(f"  Validation sequences: {counts['val']} -> {os.path.join(out_dir, 'val_seq.jsonl')}")
    print(f"  Test sequences: {counts['test']} -> {os.path.join(out_dir, 'test_seq.jsonl')}")

if __name__ == "__main__":
    extract_temporal_sequences()
