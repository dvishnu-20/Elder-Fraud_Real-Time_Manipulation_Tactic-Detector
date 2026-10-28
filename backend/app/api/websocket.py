"""
Phase 7 — WebSocket Handler
Real-time streaming via WebSocket for live dashboard updates.
Supports both demo replay and live audio modes.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import json
import asyncio
import logging
import time
import os
import sys

from app.services.session_manager import session_manager
from app.services.model_service import model_service
from app.services.temporal_engine import TemporalRiskEngine
from app.services.turn_segmenter import TurnSegmenter
from app.services.explanation_engine import generate_explanation
from app.services.privacy_service import privacy_service

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections per session."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to {session_id}: {e}")
                self.disconnect(session_id)
    
    def is_connected(self, session_id: str) -> bool:
        return session_id in self.active_connections


manager = ConnectionManager()

# Per-session temporal engines and turn segmenters
_engines: Dict[str, TemporalRiskEngine] = {}
_segmenters: Dict[str, TurnSegmenter] = {}


def _get_engine(session_id: str) -> TemporalRiskEngine:
    """Get or create a temporal engine for a session."""
    if session_id not in _engines:
        engine = TemporalRiskEngine(model_service)
        engine.reset()  # Ensure clean LSTM state
        _engines[session_id] = engine
    return _engines[session_id]


def _get_segmenter(session_id: str) -> TurnSegmenter:
    """Get or create a turn segmenter for a session."""
    if session_id not in _segmenters:
        _segmenters[session_id] = TurnSegmenter()
    return _segmenters[session_id]


def _cleanup_session(session_id: str):
    """Clean up session resources."""
    if session_id in _engines:
        _engines[session_id].reset()  # Clear LSTM state
        del _engines[session_id]
    if session_id in _segmenters:
        del _segmenters[session_id]


@router.websocket("/session/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time session streaming.
    
    Client can send:
    - {"type": "turn", "text": "...", "speaker": "..."}
    - {"type": "stop"}
    
    Server sends:
    - {"type": "transcript", ...}
    - {"type": "tactic_detection", ...}
    - {"type": "risk_update", ...}
    - {"type": "risk_event", ...}
    - {"type": "explanation", ...}
    - {"type": "intervention", ...}
    - {"type": "session_end", ...}
    """
    await manager.connect(websocket, session_id)
    
    # Ensure session exists
    session = session_manager.get_session(session_id)
    if not session:
        session = session_manager.create_session(session_id)
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=600)
            except asyncio.TimeoutError:
                # Session idle timeout
                await manager.send_message({
                    "type": "session_end",
                    "reason": "idle_timeout"
                }, session_id)
                break
            
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_message({
                    "type": "error",
                    "message": "Invalid JSON"
                }, session_id)
                continue
            
            msg_type = msg.get("type", "")
            
            if msg_type == "turn":
                await _process_turn_message(session_id, msg)
            elif msg_type == "stop":
                session_manager.stop_session(session_id)
                await manager.send_message({
                    "type": "session_end",
                    "reason": "stopped",
                    "session": session_manager.get_session(session_id).to_dict()
                }, session_id)
                break
            elif msg_type == "ping":
                await manager.send_message({"type": "pong"}, session_id)
            else:
                await manager.send_message({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                }, session_id)
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        try:
            await manager.send_message({
                "type": "error",
                "message": f"Server error: {str(e)}"
            }, session_id)
        except Exception:
            pass
    finally:
        manager.disconnect(session_id)
        _cleanup_session(session_id)


async def _process_turn_message(session_id: str, msg: dict):
    """Process a turn message through the full pipeline."""
    text = msg.get("text", "").strip()
    speaker = msg.get("speaker", "UNKNOWN")
    
    if not text:
        await manager.send_message({
            "type": "error",
            "message": "Empty transcript"
        }, session_id)
        return
    
    if not model_service.is_loaded:
        await manager.send_message({
            "type": "error",
            "message": "Models not loaded"
        }, session_id)
        return
    
    engine = _get_engine(session_id)
    segmenter = _get_segmenter(session_id)
    
    # Finalize turn
    turn = segmenter.finalize_turn(speaker=speaker, text=text)
    if not turn:
        return
    
    # Send transcript update
    redacted_text = privacy_service.redact_text(text)
    await manager.send_message({
        "type": "transcript",
        "turn_id": turn.turn_id,
        "speaker": turn.speaker,
        "text": redacted_text,
        "timestamp": turn.timestamp
    }, session_id)
    
    # Process through temporal pipeline
    try:
        event = engine.process_turn(
            turn_id=turn.turn_id,
            text=text,  # Use unredacted for model inference
            speaker=turn.speaker,
            timestamp=turn.timestamp
        )
        
        event_dict = event.to_dict()
        # Redact transcript in the event before sending to client
        event_dict["transcript"] = redacted_text
        
        # Send tactic detection
        await manager.send_message({
            "type": "tactic_detection",
            "turn_id": turn.turn_id,
            "tactics": event.tactics
        }, session_id)
        
        # Send risk update
        await manager.send_message({
            "type": "risk_update",
            "turn_id": turn.turn_id,
            "temporal_risk": event.temporal_risk,
            "threshold": event.threshold,
            "risk_triggered": event.risk_triggered,
            "risk_state": event.risk_state,
            "first_trigger_turn": event.first_trigger_turn,
            "risk_history": engine.risk_history
        }, session_id)
        
        # Send full risk event
        await manager.send_message({
            "type": "risk_event",
            **event_dict
        }, session_id)
        
        # Generate explanation
        explanation = generate_explanation(
            tactics=event.tactics,
            temporal_risk=event.temporal_risk,
            threshold=event.threshold,
            tactic_thresholds=model_service.tactic_thresholds,
            risk_state=event.risk_state,
            turn_id=turn.turn_id,
            risk_history=engine.risk_history
        )
        
        await manager.send_message({
            "type": "explanation",
            **explanation
        }, session_id)
        
        # Send intervention alert if risk triggered
        if event.risk_triggered and explanation.get("alert_message"):
            await manager.send_message({
                "type": "intervention",
                "severity": event.risk_state,
                "message": explanation["alert_message"],
                "turn_id": turn.turn_id,
                "temporal_risk": event.temporal_risk
            }, session_id)
        
        # Update session state
        session_manager.update_session(session_id, event_dict)
    
    except Exception as e:
        logger.error(f"Pipeline error at turn {turn.turn_id}: {e}", exc_info=True)
        await manager.send_message({
            "type": "error",
            "message": f"Processing error: {str(e)}",
            "turn_id": turn.turn_id
        }, session_id)


# --- Demo Replay Route ---
@router.websocket("/demo/{session_id}")
async def demo_websocket(websocket: WebSocket, session_id: str):
    """
    Demo endpoint that replays a scripted conversation through the real pipeline.
    Does NOT require microphone, external ASR, or API credentials.
    """
    await manager.connect(websocket, session_id)
    
    session = session_manager.get_session(session_id)
    if not session:
        session = session_manager.create_session(session_id)
    
    import random
    
    # 1. Bank impersonation demo script
    demo_script_bank = [
        {"speaker": "SPEAKER_A", "text": "Hello, I'm calling from your bank's fraud department.", "delay": 2.0},
        {"speaker": "SPEAKER_B", "text": "Oh, hello. Is everything alright?", "delay": 1.5},
        {"speaker": "SPEAKER_A", "text": "We have detected suspicious activity on your account. Your account has been compromised and funds are being transferred out as we speak.", "delay": 2.5},
        {"speaker": "SPEAKER_B", "text": "Oh my god, that's terrible. What should I do?", "delay": 1.5},
        {"speaker": "SPEAKER_A", "text": "You need to act immediately. We only have about ten minutes before the transfers become irreversible. Time is running out.", "delay": 2.5},
        {"speaker": "SPEAKER_B", "text": "I should call my daughter. She helps me with these things.", "delay": 1.5},
        {"speaker": "SPEAKER_A", "text": "Please do not contact your daughter or anyone else. This is a confidential federal investigation. Involving outside parties could compromise the entire operation and put your family at risk.", "delay": 3.0},
        {"speaker": "SPEAKER_B", "text": "Okay, I won't tell anyone.", "delay": 1.5},
        {"speaker": "SPEAKER_A", "text": "Good. Now I need you to stay on this line. Do not hang up or call any other number. I will guide you through transferring your funds to a secure government-protected account.", "delay": 3.0},
        {"speaker": "SPEAKER_B", "text": "Alright, what do I need to do?", "delay": 1.5},
        {"speaker": "SPEAKER_A", "text": "First, I need your online banking login credentials so I can verify your identity against our federal database. This is standard procedure.", "delay": 2.5},
    ]

    # 2. IRS / Legal threat demo script
    demo_script_irs = [
        {"speaker": "CALLER", "text": "This is Officer Johnson with the IRS Criminal Investigation Division.", "delay": 2.0},
        {"speaker": "VICTIM", "text": "Hello? The IRS? Why are you calling me?", "delay": 1.5},
        {"speaker": "CALLER", "text": "We have issued a warrant for your arrest due to unpaid back taxes and tax fraud.", "delay": 2.0},
        {"speaker": "VICTIM", "text": "What? I pay my taxes every year!", "delay": 1.5},
        {"speaker": "CALLER", "text": "If you do not resolve this matter immediately, local authorities will be dispatched to your home to arrest you.", "delay": 2.5},
        {"speaker": "VICTIM", "text": "Please, there must be a mistake! Can I call my accountant?", "delay": 1.5},
        {"speaker": "CALLER", "text": "You are not allowed to contact your accountant. Do not speak to anyone about this. You are under federal investigation.", "delay": 2.5},
        {"speaker": "VICTIM", "text": "I'm so scared. How do I fix this?", "delay": 1.5},
        {"speaker": "CALLER", "text": "You must pay the penalty fine immediately using electronic gift cards to prevent your arrest. Stay on the phone while you go to the store.", "delay": 3.0},
    ]

    # 3. Benign / Safe call demo script
    demo_script_safe = [
        {"speaker": "CLINIC", "text": "Hi, I'm calling from Dr. Smith's office to remind you of your appointment tomorrow at 10 AM.", "delay": 2.0},
        {"speaker": "PATIENT", "text": "Oh, yes. I have it written down on my calendar.", "delay": 1.5},
        {"speaker": "CLINIC", "text": "Great. Please remember to bring your insurance card and an updated list of your medications.", "delay": 2.5},
        {"speaker": "PATIENT", "text": "Will do. Do I need to fill out any new paperwork?", "delay": 1.5},
        {"speaker": "CLINIC", "text": "No, you filled everything out last time. Just show up about 5 minutes early.", "delay": 2.0},
        {"speaker": "PATIENT", "text": "Okay, sounds perfect. I will see you tomorrow.", "delay": 1.5},
        {"speaker": "CLINIC", "text": "Have a wonderful day!", "delay": 1.0},
    ]

    # Randomly select one of the scenarios
    demo_script = random.choice([demo_script_bank, demo_script_irs, demo_script_safe])
    
    try:
        # Send session start
        await manager.send_message({
            "type": "session_start",
            "session_id": session_id,
            "mode": "demo_replay",
            "total_turns": len(demo_script)
        }, session_id)
        
        for turn_data in demo_script:
            if not manager.is_connected(session_id):
                break
            
            await asyncio.sleep(turn_data["delay"])
            
            # Process through the real pipeline
            await _process_turn_message(session_id, {
                "type": "turn",
                "text": turn_data["text"],
                "speaker": turn_data["speaker"]
            })
        
        # Send session end
        session = session_manager.get_session(session_id)
        await manager.send_message({
            "type": "session_end",
            "reason": "demo_complete",
            "session": session.to_dict() if session else {}
        }, session_id)
    
    except WebSocketDisconnect:
        logger.info(f"Demo client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
    finally:
        manager.disconnect(session_id)
        _cleanup_session(session_id)
