import json
import os

VALID_LABELS = {"none", "urgency", "isolation", "authority", "emotional_destabilization", "channel_control"}

def validate_dataset(filepath):
    print(f"Validating {filepath}...")
    
    errors = []
    seen_turns = set()
    conv_splits = {}  # Track which split a conversation belongs to
    
    split_name = os.path.basename(filepath)
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {line_num}: Malformed JSON")
                continue
                
            # Check missing fields
            required = ["conversation_id", "turn_id", "speaker", "text", "labels", "synthetic", "source_type", "source"]
            missing = [req for req in required if req not in turn]
            if missing:
                errors.append(f"Line {line_num}: Missing fields {missing}")
                
            # Check duplicates
            turn_key = f"{turn.get('conversation_id')}_{turn.get('turn_id')}"
            if turn_key in seen_turns:
                errors.append(f"Line {line_num}: Duplicate turn ID {turn_key}")
            seen_turns.add(turn_key)
            
            # Empty text
            if not turn.get("text", "").strip():
                errors.append(f"Line {line_num}: Empty text")
                
            # Speaker
            if turn.get("speaker") not in ["caller", "recipient"]:
                errors.append(f"Line {line_num}: Invalid speaker '{turn.get('speaker')}'")
                
            # Labels
            labels = turn.get("labels", [])
            if not isinstance(labels, list) or len(labels) == 0:
                errors.append(f"Line {line_num}: Labels must be a non-empty list")
            else:
                invalid_labels = [l for l in labels if l not in VALID_LABELS]
                if invalid_labels:
                    errors.append(f"Line {line_num}: Invalid labels {invalid_labels}")
                if "none" in labels and len(labels) > 1:
                    errors.append(f"Line {line_num}: Invalid combination, 'none' combined with other labels")
                    
    if not errors:
        print(f"[OK] {filepath} is valid.")
        return True
    else:
        print(f"[FAIL] {filepath} has {len(errors)} errors:")
        for err in errors[:10]:
            print("  -", err)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more.")
        return False

def check_leakage(train_file, val_file, test_file):
    def get_convs(filepath):
        convs = set()
        with open(filepath, 'r') as f:
            for line in f:
                convs.add(json.loads(line)['conversation_id'])
        return convs
        
    train_convs = get_convs(train_file)
    val_convs = get_convs(val_file)
    test_convs = get_convs(test_file)
    
    leakage_val = train_convs.intersection(val_convs)
    leakage_test = train_convs.intersection(test_convs)
    
    if leakage_val or leakage_test:
        print(f"[FAIL] Data leakage detected! Train intersects Val: {len(leakage_val)}. Train intersects Test: {len(leakage_test)}.")
        return False
    else:
        print("[OK] No train/test conversation leakage.")
        return True

if __name__ == "__main__":
    val1 = validate_dataset("data/processed/train.jsonl")
    val2 = validate_dataset("data/processed/validation.jsonl")
    val3 = validate_dataset("data/processed/test.jsonl")
    
    leak = check_leakage("data/processed/train.jsonl", "data/processed/validation.jsonl", "data/processed/test.jsonl")
    
    if val1 and val2 and val3 and leak:
        print("\nAll dataset validations passed successfully.")
