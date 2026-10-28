"""
Phase 7 — Enhanced Session Manager
Full session lifecycle with LSTM state isolation between sessions.
"""
import uuid
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents an active monitoring session."""
    session_id: str
    started_at: str
    status: str = "active"  # active, stopped, timed_out
    turn_count: int = 0
    current_risk: float = 0.0
    risk_state: str = "NORMAL"
    first_trigger_turn: Optional[int] = None
    events: List[dict] = field(default_factory=list)
    transcript: List[dict] = field(default_factory=list)
    risk_history: List[float] = field(default_factory=list)
    tactic_history: List[dict] = field(default_factory=list)
    latencies: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "status": self.status,
            "turn_count": self.turn_count,
            "current_risk": round(self.current_risk, 4),
            "risk_state": self.risk_state,
            "first_trigger_turn": self.first_trigger_turn
        }


class SessionManager:
    """
    Manages monitoring sessions with strict state isolation.
    
    CRITICAL: Each session has its own temporal state.
    LSTM hidden states MUST NOT leak between sessions.
    """
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
    
    def create_session(self, session_id: str = None) -> Session:
        """Create a new monitoring session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        session = Session(
            session_id=session_id,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        self._sessions[session_id] = session
        logger.info(f"Session created: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
    
    def stop_session(self, session_id: str) -> Optional[Session]:
        """Stop a session and mark it inactive."""
        session = self._sessions.get(session_id)
        if session:
            session.status = "stopped"
            logger.info(f"Session stopped: {session_id}")
        return session
    
    def update_session(self, session_id: str, risk_event: dict):
        """Update session with a new risk event."""
        session = self._sessions.get(session_id)
        if not session:
            return
        
        session.turn_count += 1
        session.current_risk = risk_event.get("temporal_risk", 0.0)
        session.risk_state = risk_event.get("risk_state", "NORMAL")
        session.first_trigger_turn = risk_event.get("first_trigger_turn")
        session.events.append(risk_event)
        session.risk_history.append(risk_event.get("temporal_risk", 0.0))
        session.tactic_history.append(risk_event.get("tactics", {}))
        session.transcript.append({
            "turn_id": risk_event.get("turn_id"),
            "speaker": risk_event.get("speaker", "UNKNOWN"),
            "text": risk_event.get("transcript", "")
        })
        if "total_latency_ms" in risk_event:
            session.latencies.append({
                "turn_id": risk_event.get("turn_id"),
                "classification_ms": risk_event.get("classification_latency_ms", 0),
                "temporal_ms": risk_event.get("temporal_latency_ms", 0),
                "total_ms": risk_event.get("total_latency_ms", 0)
            })
    
    def delete_session(self, session_id: str):
        """Delete a session and all its data."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session deleted: {session_id}")
    
    def list_sessions(self) -> List[dict]:
        """List all sessions."""
        return [s.to_dict() for s in self._sessions.values()]
    
    def get_session_count(self) -> int:
        return len(self._sessions)


# Global singleton
session_manager = SessionManager()
