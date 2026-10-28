"""
Phase 8.1 — Root Cause Diagnostic Script
Evaluates frozen DistilBERT V2 and Causal LSTM V2 to determine why
sensitivity collapsed in Phase 8.
- Paired Clean vs Degraded Text
- Utterance Length Analysis
- Error Decomposition
"""
import sys
import json
import random
import logging
from pathlib import Path
import os
import csv

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase8_1_diagnostic")

# Minimal set of Phase 8 cases to diagnose
DIAGNOSTIC_CASES = [
    {
        "id": "conv_1",
        "ground_truth": "fraud",
        "turns": [
            ("I'm calling from your bank's fraud department. Your account is compromised.", ["authority", "urgency"]),
            ("You need to act immediately to secure your funds.", ["urgency"]),
            ("Do not tell anyone about this, it is a federal investigation.", ["isolation", "authority"]),
            ("Stay on the line and transfer the funds to this secure account.", ["channel_control"])
        ]
    },
    {
        "id": "conv_2",
        "ground_truth": "fraud",
        "turns": [
            ("This is Officer Jones. There is a warrant out for your arrest.", ["authority"]),
            ("What? Why?", []),
            ("You owe back taxes. If you don't pay immediately, police will arrive in ten minutes.", ["urgency", "authority"]),
            ("Don't call anyone or hang up, or I will dispatch the officers now.", ["isolation", "channel_control"])
        ]
    },
    {
        "id": "conv_3",
        "ground_truth": "benign",
        "turns": [
            ("Hi Mom, how are you doing?", []),
            ("I'm good, just watching TV.", []),
            ("Great, I'll come over this weekend.", []),
            ("See you then.", [])
        ]
    }
]

def simulate_asr_degradation(text: str, error_rate: float) -> str:
    words = text.split()
    new_words = [w for w in words if random.random() >= error_rate]
    return " ".join(new_words)

def run_diagnostic():
    from app.services.model_service import model_service
    from app.services.temporal_engine import TemporalRiskEngine
    import torch
    
    logger.info("Loading frozen models...")
    model_service.load_models(verify_checksums=True)
    threshold = model_service.lstm_threshold
    
    results = {
        "clean": {"tp": 0, "fn": 0, "fp": 0, "tn": 0},
        "degraded": {"tp": 0, "fn": 0, "fp": 0, "tn": 0},
        "short_utterance_clean": {"tp": 0, "fn": 0},  # <= 6 words
        "long_utterance_clean": {"tp": 0, "fn": 0},   # > 6 words
        "errors": []
    }
    
    tactic_list = ["urgency", "isolation", "authority", "emotional_destabilization", "channel_control"]
    
    for case in DIAGNOSTIC_CASES:
        for condition in ["clean", "degraded"]:
            engine = TemporalRiskEngine(model_service)
            engine.reset()
            
            error_rate = 0.20 if condition == "degraded" else 0.0
            
            for t_idx, (text, gt_tactics) in enumerate(case["turns"]):
                process_text = simulate_asr_degradation(text, error_rate)
                word_count = len(process_text.split())
                
                # Manual inference to get raw probs
                tactic_result = model_service.classify_turn(process_text)
                tactic_probs = tactic_result["tactics"]
                probs = [tactic_probs[t] for t in tactic_list]
                
                # Check tactic hits (Threshold 0.5 for DistilBERT)
                tactic_detected = any(p >= 0.5 for p in probs)
                gt_present = len(gt_tactics) > 0
                
                if condition == "clean" and case["ground_truth"] == "fraud":
                    if word_count <= 6:
                        if tactic_detected: results["short_utterance_clean"]["tp"] += 1
                        else: results["short_utterance_clean"]["fn"] += 1
                    else:
                        if tactic_detected: results["long_utterance_clean"]["tp"] += 1
                        else: results["long_utterance_clean"]["fn"] += 1
                
                # Track errors
                if condition == "clean" and gt_present and not tactic_detected:
                    results["errors"].append({
                        "type": "classifier_miss",
                        "text": text,
                        "word_count": word_count,
                        "probs": probs,
                        "gt_tactics": gt_tactics
                    })
                    
                # Full step
                engine.process_turn(t_idx + 1, process_text, "SPEAKER", f"00:{t_idx:02d}")
            
            final_risk = engine.current_risk
            predicted = "fraud" if final_risk >= threshold else "benign"
            gt = case["ground_truth"]
            
            # Conv Level Metrics
            if gt == "fraud":
                if predicted == "fraud": results[condition]["tp"] += 1
                else: 
                    results[condition]["fn"] += 1
                    # Was it temporal miss or classifier miss?
                    # If tactic was detected anywhere but risk < 0.2, it's temporal miss.
                    results["errors"].append({
                        "type": "temporal_miss" if any(p >= 0.5 for p in probs) else "mixed_miss",
                        "final_risk": final_risk
                    })
            else:
                if predicted == "benign": results[condition]["tn"] += 1
                else: results[condition]["fp"] += 1

    # Save Results
    out_dir = PROJECT_ROOT / "reports" / "phase8_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "phase8_1_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Diagnostic completed.")

if __name__ == "__main__":
    run_diagnostic()
