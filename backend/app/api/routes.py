"""
Phase 7 — API Routes
Full REST API for session management, risk querying, and health checks.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

from app.services.session_manager import session_manager
from app.services.model_service import model_service
from app.services.privacy_service import privacy_service

router = APIRouter()


# --- Request Models ---
class StartSessionRequest(BaseModel):
    session_id: Optional[str] = None


class ProcessTurnRequest(BaseModel):
    text: str
    speaker: str = "UNKNOWN"


# --- Health ---
@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "models_loaded": model_service.is_loaded,
        "active_sessions": session_manager.get_session_count(),
        "device": str(model_service.device) if model_service.is_loaded else "not_loaded"
    }


# --- Model Info ---
@router.get("/model/info")
async def model_info():
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "distilbert": {
            "status": "FROZEN",
            "labels": model_service.labels,
            "tactic_thresholds": model_service.tactic_thresholds
        },
        "lstm": {
            "status": "FROZEN",
            "decision_threshold": model_service.lstm_threshold,
            "architecture": "CausalLSTMRiskModel(input_dim=5, hidden_dim=64)"
        },
        "load_times": model_service.get_load_times(),
        "scientific_status": {
            "phase5": "[B] MODEST TEMPORAL VALUE",
            "phase6": "[B] MODERATE GENERALIZATION",
            "phase7": "ENGINEERING INTEGRATION"
        }
    }


# --- Session Management ---
@router.post("/session/start")
async def start_session(request: StartSessionRequest = StartSessionRequest()):
    session_id = request.session_id or str(uuid.uuid4())
    session = session_manager.create_session(session_id)
    return {
        "session_id": session.session_id,
        "status": "started",
        "started_at": session.started_at
    }


@router.post("/session/{session_id}/stop")
async def stop_session(session_id: str):
    session = session_manager.stop_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.get("/session/{session_id}/transcript")
async def get_transcript(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Apply redaction to transcripts before returning
    redacted = []
    for t in session.transcript:
        redacted.append({
            **t,
            "text": privacy_service.redact_text(t.get("text", ""))
        })
    return {"session_id": session_id, "transcript": redacted}


@router.get("/session/{session_id}/risk")
async def get_risk(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "current_risk": session.current_risk,
        "risk_state": session.risk_state,
        "risk_history": session.risk_history,
        "threshold": model_service.lstm_threshold if model_service.is_loaded else 0.20,
        "first_trigger_turn": session.first_trigger_turn
    }


@router.get("/session/{session_id}/events")
async def get_events(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Redact transcripts in events
    redacted_events = []
    for e in session.events:
        re_event = dict(e)
        re_event["transcript"] = privacy_service.redact_text(re_event.get("transcript", ""))
        redacted_events.append(re_event)
    
    return {"session_id": session_id, "events": redacted_events}


@router.get("/session/{session_id}/latency")
async def get_latency(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    latencies = session.latencies
    if not latencies:
        return {"session_id": session_id, "latencies": [], "summary": {}}
    
    total_ms_list = [l["total_ms"] for l in latencies]
    total_ms_list.sort()
    n = len(total_ms_list)
    
    summary = {
        "count": n,
        "mean_ms": round(sum(total_ms_list) / n, 2),
        "median_ms": round(total_ms_list[n // 2], 2),
        "p95_ms": round(total_ms_list[int(n * 0.95)] if n >= 20 else total_ms_list[-1], 2),
        "max_ms": round(max(total_ms_list), 2),
        "min_ms": round(min(total_ms_list), 2)
    }
    
    return {"session_id": session_id, "latencies": latencies, "summary": summary}


# --- Sessions List ---
@router.get("/sessions")
async def list_sessions():
    return {"sessions": session_manager.list_sessions()}


# --- Privacy Config ---
@router.get("/privacy/config")
async def get_privacy_config():
    return privacy_service.get_privacy_config()
