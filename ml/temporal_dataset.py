import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any

class ConversationSequenceDataset(Dataset):
    def __init__(self, filepath: str, max_seq_len: int = 20):
        self.samples = []
        self.max_seq_len = max_seq_len
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    self.samples.append(item)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        probs = item.get("features", item.get("turn_probabilities", []))
        length = len(probs)
        
        # Truncate if exceeds max_seq_len
        if length > self.max_seq_len:
            probs = probs[:self.max_seq_len]
            length = self.max_seq_len
            
        # Create padded sequence (max_seq_len, 5)
        feature_dim = 5
        padded_features = np.zeros((self.max_seq_len, feature_dim), dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        
        if length > 0:
            padded_features[:length] = np.array(probs, dtype=np.float32)
            mask[:length] = 1.0
            
        return {
            "features": torch.tensor(padded_features, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.float32),
            "conversation_id": item["conversation_id"],
            "topic": item.get("topic", "unknown")
        }

def get_aggregated_features_for_lr(dataset: ConversationSequenceDataset):
    X = []
    y = []
    cids = []
    
    for item in dataset.samples:
        probs = item.get("features", item.get("turn_probabilities", []))
        probs = np.array(probs, dtype=np.float32)
        label = item["label"]
        cids.append(item["conversation_id"])
        y.append(label)
        
        if len(probs) == 0:
            # 20 zero features (mean_5, max_5, sum_5, final_5)
            feat = np.zeros(20, dtype=np.float32)
        else:
            mean_probs = np.mean(probs, axis=0) # 5
            max_probs = np.max(probs, axis=0)   # 5
            sum_probs = np.sum(probs, axis=0)   # 5
            final_probs = probs[-1]             # 5
            feat = np.concatenate([mean_probs, max_probs, sum_probs, final_probs]) # 20
            
        X.append(feat)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), cids

