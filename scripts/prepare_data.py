import json
import os
import random
import argparse
from collections import defaultdict

def prepare_and_split(input_file, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    
    conversations = []
    with open(input_file, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
            
    # Shuffle conversations using seed
    random.seed(seed)
    random.shuffle(conversations)
    
    total = len(conversations)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    
    train_convs = conversations[:train_end]
    val_convs = conversations[train_end:val_end]
    test_convs = conversations[val_end:]
    
    def write_split(convs, filename):
        path = os.path.join(out_dir, filename)
        with open(path, 'w') as f:
            for conv in convs:
                for turn in conv['turns']:
                    turn['conversation_level_label'] = conv['conversation_level_label']
                    f.write(json.dumps(turn) + "\n")

    write_split(train_convs, "train.jsonl")
    write_split(val_convs, "validation.jsonl")
    write_split(test_convs, "test.jsonl")
    
    print("Dataset strictly split at conversation level. No template leakage via random split.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    prepare_and_split("data/raw/synthetic_conversations.jsonl", "data/processed", seed=args.seed)
