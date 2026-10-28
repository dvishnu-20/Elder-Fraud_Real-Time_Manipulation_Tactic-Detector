"""
Phase 7 — Offline Replay Script
=================================
Replays recorded conversations through the COMPLETE real-time pipeline
as if they were arriving live.

Usage:
    python scripts/run_phase7_replay.py --demo
    python scripts/run_phase7_replay.py --dataset data/processed/temporal_v2/test.jsonl --max-conversations 10

This provides a deterministic test of the entire pipeline WITHOUT requiring:
- Microphone
- External ASR API
- External service credentials
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("phase7_replay")


def load_dataset(filepath: str) -> dict:
    """Load JSONL dataset and group turns by conversation_id."""
    conversations = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            conv_id = record.get("conversation_id", "unknown")
            conversations[conv_id].append(record)
    
    # Sort turns within each conversation by turn_id
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x.get("turn_id", 0))
    
    return dict(conversations)


def get_demo_script():
    """Built-in demo script for testing without dataset."""
    return {
        "demo_fraud_001": {
            "label": "fraud",
            "turns": [
                {"turn_id": 1, "speaker": "SPEAKER_A", "text": "Hello, I'm calling from your bank's fraud department."},
                {"turn_id": 2, "speaker": "SPEAKER_B", "text": "Oh, hello. Is everything alright?"},
                {"turn_id": 3, "speaker": "SPEAKER_A", "text": "We have detected suspicious activity on your account. Your account has been compromised and funds are being transferred out as we speak."},
                {"turn_id": 4, "speaker": "SPEAKER_B", "text": "Oh my god, what should I do?"},
                {"turn_id": 5, "speaker": "SPEAKER_A", "text": "You need to act immediately. We only have about ten minutes before the transfers become irreversible."},
                {"turn_id": 6, "speaker": "SPEAKER_B", "text": "I should call my daughter, she helps me with these things."},
                {"turn_id": 7, "speaker": "SPEAKER_A", "text": "Please do not contact your daughter or anyone else. This is a confidential federal investigation. Involving outside parties could compromise the operation."},
                {"turn_id": 8, "speaker": "SPEAKER_B", "text": "Okay, I won't tell anyone."},
                {"turn_id": 9, "speaker": "SPEAKER_A", "text": "Good. Stay on this line and do not hang up. I will guide you through transferring your funds to a secure government-protected account."},
            ]
        },
        "demo_benign_001": {
            "label": "benign",
            "turns": [
                {"turn_id": 1, "speaker": "SPEAKER_A", "text": "Thank you for calling customer service. How can I help you today?"},
                {"turn_id": 2, "speaker": "SPEAKER_B", "text": "Hi, I'd like to check the status of my recent order."},
                {"turn_id": 3, "speaker": "SPEAKER_A", "text": "Sure, I can help with that. Could you please provide your order number?"},
                {"turn_id": 4, "speaker": "SPEAKER_B", "text": "Yes, it's order number 12345."},
                {"turn_id": 5, "speaker": "SPEAKER_A", "text": "I can see your order was shipped yesterday. It should arrive within three to five business days."},
                {"turn_id": 6, "speaker": "SPEAKER_B", "text": "Great, thank you for your help!"},
                {"turn_id": 7, "speaker": "SPEAKER_A", "text": "You're welcome. Is there anything else I can help you with?"},
            ]
        }
    }


def run_replay(args):
    """Run the full replay pipeline."""
    from app.services.model_service import model_service
    from app.services.temporal_engine import TemporalRiskEngine
    from app.services.turn_segmenter import TurnSegmenter
    from app.services.explanation_engine import generate_explanation
    from app.services.privacy_service import privacy_service
    
    # Step 1: Load frozen models
    logger.info("=" * 70)
    logger.info("PHASE 7 — OFFLINE REPLAY")
    logger.info("=" * 70)
    
    t0 = time.perf_counter()
    model_service.load_models(verify_checksums=True)
    model_load_time = time.perf_counter() - t0
    logger.info(f"Models loaded in {model_load_time:.2f}s")
    logger.info(f"Device: {model_service.device}")
    logger.info(f"LSTM threshold: tau = {model_service.lstm_threshold}")
    
    # Step 2: Load conversations
    if args.demo:
        logger.info("Using built-in demo script")
        demo_data = get_demo_script()
        conversations = {}
        for conv_id, conv_data in demo_data.items():
            conversations[conv_id] = {
                "turns": conv_data["turns"],
                "ground_truth": conv_data["label"]
            }
    else:
        logger.info(f"Loading dataset: {args.dataset}")
        raw_convs = load_dataset(args.dataset)
        conversations = {}
        count = 0
        for conv_id, turns in raw_convs.items():
            if args.max_conversations and count >= args.max_conversations:
                break
            ground_truth = turns[0].get("conversation_level_label", "unknown")
            conversations[conv_id] = {
                "turns": [
                    {
                        "turn_id": t.get("turn_id", i+1),
                        "speaker": t.get("speaker", "UNKNOWN"),
                        "text": t.get("text", "")
                    }
                    for i, t in enumerate(turns)
                ],
                "ground_truth": ground_truth
            }
            count += 1
    
    logger.info(f"Loaded {len(conversations)} conversations")
    
    # Step 3: Process each conversation
    results = []
    all_latencies = []
    
    for conv_id, conv_data in conversations.items():
        turns = conv_data["turns"]
        ground_truth = conv_data["ground_truth"]
        
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Conversation: {conv_id} (ground_truth={ground_truth})")
        logger.info(f"{'─' * 60}")
        
        # Fresh engine and segmenter per conversation (session isolation)
        engine = TemporalRiskEngine(model_service)
        engine.reset()
        segmenter = TurnSegmenter()
        segmenter.reset()
        
        conv_events = []
        
        for turn_data in turns:
            turn = segmenter.finalize_turn(
                speaker=turn_data["speaker"],
                text=turn_data["text"]
            )
            
            if not turn:
                continue
            
            # Configurable timing delay
            if args.delay > 0:
                time.sleep(args.delay)
            
            # Process through pipeline
            event = engine.process_turn(
                turn_id=turn.turn_id,
                text=turn.text,
                speaker=turn.speaker,
                timestamp=turn.timestamp
            )
            
            conv_events.append(event.to_dict())
            
            # Log turn result
            tactic_str = ", ".join(
                f"{k}={v:.3f}" for k, v in event.tactics.items() if v > 0.1
            )
            logger.info(
                f"  Turn {turn.turn_id:2d} [{turn.speaker:10s}]: "
                f"risk={event.temporal_risk:.4f} "
                f"state={event.risk_state:10s} "
                f"triggered={'✓' if event.risk_triggered else '·'} "
                f"latency={event.total_latency_ms:.1f}ms "
                f"| {tactic_str or 'no active tactics'}"
            )
        
        # Conversation result
        final_risk = engine.current_risk
        final_triggered = final_risk >= model_service.lstm_threshold
        first_trigger = engine.first_trigger_turn
        predicted_label = "fraud" if final_triggered else "benign"
        correct = (predicted_label == ground_truth) if ground_truth != "unknown" else None
        
        result = {
            "conversation_id": conv_id,
            "ground_truth": ground_truth,
            "predicted": predicted_label,
            "correct": correct,
            "final_risk": round(final_risk, 6),
            "risk_triggered": final_triggered,
            "first_trigger_turn": first_trigger,
            "total_turns": len(conv_events),
            "latencies": engine.latencies,
            "risk_progression": engine.risk_history
        }
        results.append(result)
        all_latencies.extend(engine.latencies)
        
        logger.info(
            f"  RESULT: predicted={predicted_label} "
            f"ground_truth={ground_truth} "
            f"correct={'✓' if correct else ('✗' if correct is False else '?')} "
            f"final_risk={final_risk:.4f} "
            f"detection_turn={first_trigger or 'N/A'}"
        )
    
    # Step 4: Summary statistics
    logger.info(f"\n{'=' * 70}")
    logger.info("REPLAY SUMMARY")
    logger.info(f"{'=' * 70}")
    
    total = len(results)
    if any(r["correct"] is not None for r in results):
        correct_count = sum(1 for r in results if r["correct"] is True)
        evaluated = sum(1 for r in results if r["correct"] is not None)
        accuracy = correct_count / evaluated if evaluated > 0 else 0
        
        # Breakdown by class
        fraud_results = [r for r in results if r["ground_truth"] == "fraud"]
        benign_results = [r for r in results if r["ground_truth"] == "benign"]
        
        tp = sum(1 for r in fraud_results if r["predicted"] == "fraud")
        fn = sum(1 for r in fraud_results if r["predicted"] == "benign")
        fp = sum(1 for r in benign_results if r["predicted"] == "fraud")
        tn = sum(1 for r in benign_results if r["predicted"] == "benign")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Detection latency for fraud
        detection_turns = [r["first_trigger_turn"] for r in fraud_results if r["first_trigger_turn"] is not None]
        
        logger.info(f"  Total conversations: {total}")
        logger.info(f"  Fraud: {len(fraud_results)} | Benign: {len(benign_results)}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info(f"  FPR: {fpr:.4f}")
        logger.info(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
        
        if detection_turns:
            mean_det = sum(detection_turns) / len(detection_turns)
            detection_turns_sorted = sorted(detection_turns)
            median_det = detection_turns_sorted[len(detection_turns_sorted) // 2]
            logger.info(f"  Mean detection turn: {mean_det:.2f}")
            logger.info(f"  Median detection turn: {median_det}")
    
    # Latency summary
    if all_latencies:
        total_ms = [l["total_ms"] for l in all_latencies]
        clf_ms = [l["classification_ms"] for l in all_latencies]
        temp_ms = [l["temporal_ms"] for l in all_latencies]
        total_ms.sort()
        n = len(total_ms)
        
        logger.info(f"\n  LATENCY ({n} turns):")
        logger.info(f"    Total:  mean={sum(total_ms)/n:.2f}ms  median={total_ms[n//2]:.2f}ms  "
                     f"p95={total_ms[int(n*0.95)]:.2f}ms  max={max(total_ms):.2f}ms")
        logger.info(f"    Classif: mean={sum(clf_ms)/n:.2f}ms")
        logger.info(f"    Temporal: mean={sum(temp_ms)/n:.2f}ms")
    
    # Save results
    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    
    replay_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "demo" if args.demo else "dataset",
        "dataset": args.dataset if not args.demo else "built_in_demo",
        "conversations_processed": total,
        "model_load_time_s": round(model_load_time, 2),
        "lstm_threshold": model_service.lstm_threshold,
        "results": results,
        "latency_summary": {
            "total_turns": len(all_latencies),
            "mean_ms": round(sum(l["total_ms"] for l in all_latencies) / len(all_latencies), 2) if all_latencies else 0,
            "max_ms": round(max(l["total_ms"] for l in all_latencies), 2) if all_latencies else 0
        }
    }
    
    results_path = output_dir / "phase7_replay_results.json"
    with open(results_path, "w") as f:
        json.dump(replay_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"{'=' * 70}")
    
    return replay_results


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Offline Replay")
    parser.add_argument("--demo", action="store_true", 
                        help="Run built-in demo script (no dataset required)")
    parser.add_argument("--dataset", type=str,
                        default="data/processed/temporal_v2/test.jsonl",
                        help="Path to JSONL dataset to replay")
    parser.add_argument("--max-conversations", type=int, default=None,
                        help="Maximum number of conversations to replay")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between turns in seconds (simulates real-time)")
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(str(PROJECT_ROOT))
    
    run_replay(args)


if __name__ == "__main__":
    main()
