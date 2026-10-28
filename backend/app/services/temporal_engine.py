"""
Phase 7 — Temporal Risk Engine
Wraps the frozen Causal LSTM V2 for incremental, causal temporal risk estimation.

CRITICAL INVARIANT: prediction(t) uses ONLY turns 1..t, NEVER turn t+1 or later.
LSTM hidden state is managed per-session and MUST be reset between sessions.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RiskEvent:
    """Structured risk event generated after each processed turn."""
    timestamp: str
    turn_id: int
    transcript: str
    speaker: str
    tactics: dict
    temporal_risk: float
    threshold: float
    risk_triggered: bool
    risk_state: str
    first_trigger_turn: Optional[int]
    classification_latency_ms: float
    temporal_latency_ms: float
    total_latency_ms: float
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "turn_id": self.turn_id,
            "transcript": self.transcript,
            "speaker": self.speaker,
            "tactics": self.tactics,
            "temporal_risk": round(self.temporal_risk, 6),
            "threshold": self.threshold,
            "risk_triggered": self.risk_triggered,
            "risk_state": self.risk_state,
            "first_trigger_turn": self.first_trigger_turn,
            "classification_latency_ms": round(self.classification_latency_ms, 2),
            "temporal_latency_ms": round(self.temporal_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2)
        }


class TemporalRiskEngine:
    """
    Real-time temporal risk engine using the frozen Causal LSTM V2.
    
    For every finalized turn:
    1. Classify with DistilBERT V2 → 5 tactic probabilities
    2. Feed into Causal LSTM V2 → temporal risk
    3. Compare against tau = 0.20
    4. Generate structured RiskEvent
    
    Session state (LSTM hidden state) is fully isolated.
    """
    
    def __init__(self, model_service):
        self._model = model_service
        self._hidden_state = None  # LSTM (h, c) — reset per session
        self._risk_history: List[float] = []
        self._tactic_history: List[dict] = []
        self._events: List[RiskEvent] = []
        self._first_trigger_turn: Optional[int] = None
        self._latencies: List[dict] = []
    
    def reset(self):
        """
        Reset all temporal state for a new session.
        CRITICAL: LSTM hidden state MUST be cleared here.
        """
        self._hidden_state = None
        self._risk_history = []
        self._tactic_history = []
        self._events = []
        self._first_trigger_turn = None
        self._latencies = []
        logger.info("Temporal risk engine reset — LSTM hidden state cleared")
    
    def process_turn(self, turn_id: int, text: str, speaker: str, 
                     timestamp: str) -> RiskEvent:
        """
        Process a single finalized turn through the full pipeline.
        
        CAUSAL GUARANTEE: Only uses turns 1..turn_id.
        The hidden state accumulates causally from previous calls.
        """
        t_total_start = time.perf_counter()
        
        # Step 1: DistilBERT V2 tactic classification
        clf_result = self._model.classify_turn(text)
        tactics = clf_result["tactics"]
        clf_latency = clf_result["latency_ms"]
        
        self._tactic_history.append(tactics)
        
        # Step 2: Causal LSTM V2 temporal risk
        tactic_vector = [
            tactics["urgency"],
            tactics["isolation"],
            tactics["authority"],
            tactics["emotional_destabilization"],
            tactics["channel_control"]
        ]
        
        temporal_result = self._model.predict_temporal_risk(
            tactic_vector, 
            self._hidden_state
        )
        
        risk = temporal_result["risk"]
        risk_triggered = temporal_result["risk_triggered"]
        self._hidden_state = temporal_result["hidden_state"]
        temporal_latency = temporal_result["latency_ms"]
        
        self._risk_history.append(risk)
        
        # Track first trigger
        if risk_triggered and self._first_trigger_turn is None:
            self._first_trigger_turn = turn_id
        
        # Determine risk state
        risk_state = self._model.get_risk_state(risk)
        
        total_latency = (time.perf_counter() - t_total_start) * 1000
        
        # Record latency
        self._latencies.append({
            "turn_id": turn_id,
            "classification_ms": clf_latency,
            "temporal_ms": temporal_latency,
            "total_ms": total_latency
        })
        
        # Build risk event
        event = RiskEvent(
            timestamp=timestamp,
            turn_id=turn_id,
            transcript=text,
            speaker=speaker,
            tactics=tactics,
            temporal_risk=risk,
            threshold=self._model.lstm_threshold,
            risk_triggered=risk_triggered,
            risk_state=risk_state,
            first_trigger_turn=self._first_trigger_turn,
            classification_latency_ms=clf_latency,
            temporal_latency_ms=temporal_latency,
            total_latency_ms=total_latency
        )
        
        self._events.append(event)
        
        logger.info(
            f"Turn {turn_id}: risk={risk:.4f} state={risk_state} "
            f"triggered={risk_triggered} latency={total_latency:.1f}ms"
        )
        
        return event
    
    @property
    def current_risk(self) -> float:
        return self._risk_history[-1] if self._risk_history else 0.0
    
    @property
    def risk_history(self) -> List[float]:
        return self._risk_history.copy()
    
    @property
    def tactic_history(self) -> List[dict]:
        return self._tactic_history.copy()
    
    @property
    def events(self) -> List[RiskEvent]:
        return self._events.copy()
    
    @property
    def first_trigger_turn(self) -> Optional[int]:
        return self._first_trigger_turn
    
    @property
    def latencies(self) -> List[dict]:
        return self._latencies.copy()
    
    def get_hidden_state(self):
        """Expose hidden state for testing purposes only."""
        return self._hidden_state
