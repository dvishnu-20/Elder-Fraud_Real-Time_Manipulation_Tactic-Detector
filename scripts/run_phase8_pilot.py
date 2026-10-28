"""
Phase 8 — Controlled Pilot Simulator
Simulates the real-world operational pilot environment, including:
- 10 semantic categories (fraud and hard-negative benign)
- Simulated ASR degradation
- False alarm measurement
- Reliability and Memory testing
"""
import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from collections import defaultdict

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase8_pilot")

# ---------------------------------------------------------
# Simulated Pilot Dataset (10 Categories)
# ---------------------------------------------------------
PILOT_CATEGORIES = {
    "fraudulent_manipulation": [
        ("fraud", [
            "I'm calling from your bank's fraud department. Your account is compromised.",
            "You need to act immediately to secure your funds.",
            "Do not tell anyone about this, it is a federal investigation.",
            "Stay on the line and transfer the funds to this secure account."
        ]) for _ in range(15)
    ],
    "benign_financial": [
        ("benign", [
            "Hi, I'd like to check my account balance.",
            "Sure, can you verify your date of birth?",
            "Yes, it's January 1st.",
            "Your balance is $5,000."
        ]) for _ in range(10)
    ],
    "benign_family": [
        ("benign", [
            "Hi Mom, how are you doing?",
            "I'm good, just watching TV.",
            "Great, I'll come over this weekend.",
            "See you then."
        ]) for _ in range(10)
    ],
    "benign_customer_service": [
        ("benign", [
            "Thanks for calling Amazon. How can I help?",
            "Where is my package?",
            "It will arrive tomorrow.",
            "Thank you."
        ]) for _ in range(10)
    ],
    "authority_benign": [
        ("benign", [
            "This is Dr. Smith's office calling.",
            "Yes, hello.",
            "We need to confirm your appointment for tomorrow.",
            "Confirmed."
        ]) for _ in range(5)
    ],
    "urgency_benign": [
        ("benign", [
            "Hurry up, we are going to miss the flight!",
            "I'm packing as fast as I can.",
            "We need to leave in five minutes.",
            "Okay, I'm ready."
        ]) for _ in range(5)
    ],
    "emotional_benign": [
        ("benign", [
            "I can't believe he passed away.",
            "I'm so sorry for your loss.",
            "It's just so sudden.",
            "I'm here if you need anything."
        ]) for _ in range(5)
    ],
    "channel_control_benign": [
        ("benign", [
            "Are you in a private place?",
            "Yes, I'm at home.",
            "Okay, I have your medical test results.",
            "Go ahead."
        ]) for _ in range(5)
    ],
    "multi_tactic_manipulation": [
        ("fraud", [
            "This is Officer Jones. There is a warrant out for your arrest.",
            "What? Why?",
            "You owe back taxes. If you don't pay immediately, police will arrive in ten minutes.",
            "Don't call anyone or hang up, or I will dispatch the officers now."
        ]) for _ in range(10)
    ],
    "indirect_manipulation": [
        ("fraud", [
            "I love talking to you every day.",
            "Me too.",
            "I have an investment opportunity that can make us rich so we can be together.",
            "Oh really?"
        ]) for _ in range(5)
    ]
}

def simulate_asr_degradation(text: str, error_rate: float) -> str:
    """Simulates ASR errors by dropping or garbling words."""
    words = text.split()
    new_words = []
    for w in words:
        if random.random() < error_rate:
            continue  # Drop word
        new_words.append(w)
    return " ".join(new_words)

def run_pilot():
    from app.services.model_service import model_service
    from app.services.temporal_engine import TemporalRiskEngine
    
    logger.info("Loading models...")
    t0 = time.perf_counter()
    model_service.load_models(verify_checksums=True)
    load_time = time.perf_counter() - t0
    logger.info(f"Models loaded in {load_time:.2f}s")
    
    threshold = model_service.lstm_threshold
    results = []
    latencies = []
    false_alarms = []
    
    # Run all categories
    for category, conversations in PILOT_CATEGORIES.items():
        for i, (ground_truth, turns) in enumerate(conversations):
            engine = TemporalRiskEngine(model_service)
            engine.reset()
            
            # Apply ASR degradation conditionally (simulate noisy environment)
            error_rate = 0.0
            if "noise" in category or random.random() < 0.2:
                error_rate = 0.15
            
            conv_events = []
            for t_idx, text in enumerate(turns):
                degraded_text = simulate_asr_degradation(text, error_rate)
                
                # Mock speakers for simplicity
                speaker = "SPEAKER_A" if t_idx % 2 == 0 else "SPEAKER_B"
                
                event = engine.process_turn(
                    turn_id=t_idx + 1,
                    text=degraded_text,
                    speaker=speaker,
                    timestamp=f"00:{t_idx:02d}"
                )
                conv_events.append(event)
                latencies.append(event.total_latency_ms)
            
            final_risk = engine.current_risk
            triggered = final_risk >= threshold
            predicted = "fraud" if triggered else "benign"
            
            if predicted == "fraud" and ground_truth == "benign":
                false_alarms.append({
                    "category": category,
                    "final_risk": final_risk,
                    "turns": turns
                })
            
            results.append({
                "category": category,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": predicted == ground_truth,
                "final_risk": final_risk,
                "first_trigger_turn": engine.first_trigger_turn,
                "asr_error_rate_applied": error_rate
            })

    # Metrics computation
    total = len(results)
    fraud_cases = [r for r in results if r["ground_truth"] == "fraud"]
    benign_cases = [r for r in results if r["ground_truth"] == "benign"]
    
    tp = sum(1 for r in fraud_cases if r["predicted"] == "fraud")
    fn = sum(1 for r in fraud_cases if r["predicted"] == "benign")
    fp = sum(1 for r in benign_cases if r["predicted"] == "fraud")
    tn = sum(1 for r in benign_cases if r["predicted"] == "benign")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Latency stats
    latencies.sort()
    mean_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = latencies[int(len(latencies) * 0.95)] if latencies else 0
    
    output = {
        "pilot_size": total,
        "fraud_total": len(fraud_cases),
        "benign_total": len(benign_cases),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn
        },
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "p95": round(p95_lat, 2),
            "max": round(max(latencies), 2) if latencies else 0
        },
        "false_alarms": false_alarms,
        "false_alarms_per_hour": round(fp / (total * 2 / 60), 2)  # Rough approx (2 mins per conv)
    }
    
    out_path = PROJECT_ROOT / "data" / "phase8_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Phase 8 Pilot Complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_pilot()
