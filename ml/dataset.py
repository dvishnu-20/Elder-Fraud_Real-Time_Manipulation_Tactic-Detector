import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict

TACTIC_LABELS = [
    "urgency",
    "isolation",
    "authority",
    "emotional_destabilization",
    "channel_control"
]
LABEL_TO_ID = {label: i for i, label in enumerate(TACTIC_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(TACTIC_LABELS)}

class TacticDataset(Dataset):
    def __init__(self, filepath: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = data['text']
                labels = data['labels']
                
                # Multi-hot encoding (ignore 'none')
                multi_hot = [0.0] * len(TACTIC_LABELS)
                for label in labels:
                    if label in LABEL_TO_ID:
                        multi_hot[LABEL_TO_ID[label]] = 1.0
                        
                # We retain metadata for detailed evaluation reports later
                self.samples.append({
                    "text": text,
                    "labels": multi_hot,
                    "source": data.get("source", "unknown"),
                    "speaker": data.get("speaker", "unknown"),
                    "difficulty": data.get("difficulty", "none"),
                    "conversation_level_label": data.get("conversation_level_label", "unknown"),
                    "raw_labels": labels
                })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["labels"], dtype=torch.float)
        }

    def get_metadata(self):
        return self.samples
