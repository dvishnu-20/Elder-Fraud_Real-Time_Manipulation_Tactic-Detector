"""
Phase 7 — Turn Segmenter
Constructs conversational turns from streaming ASR output.
Each finalized turn gets a turn_id, timestamp, speaker, transcript, and ASR confidence.
"""
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A finalized conversational turn."""
    turn_id: int
    speaker: str
    timestamp: str
    text: str
    asr_confidence: float = 1.0
    finalized_at: float = 0.0  # perf_counter timestamp for latency measurement
    
    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "text": self.text,
            "asr_confidence": round(self.asr_confidence, 4)
        }


class TurnSegmenter:
    """
    Streaming turn construction from ASR output.
    
    A turn is finalized when:
    - ASR endpoint detected
    - Sufficient silence (configurable)
    - Speaker change (if diarization available)
    - Maximum turn duration exceeded
    
    Speaker identities are NOT invented. If diarization is unavailable,
    speakers are labeled SPEAKER_A / SPEAKER_B or UNKNOWN.
    """
    
    def __init__(
        self,
        silence_threshold_ms: float = 1500.0,
        max_turn_duration_ms: float = 30000.0,
        min_turn_length_chars: int = 3
    ):
        self._silence_threshold_ms = silence_threshold_ms
        self._max_turn_duration_ms = max_turn_duration_ms
        self._min_turn_length_chars = min_turn_length_chars
        self._turn_counter = 0
        self._current_buffer = ""
        self._current_speaker = "UNKNOWN"
        self._turn_start_time = None
        self._finalized_turns: List[Turn] = []
        self._session_start = time.time()
    
    def reset(self):
        """Reset segmenter state for a new session."""
        self._turn_counter = 0
        self._current_buffer = ""
        self._current_speaker = "UNKNOWN"
        self._turn_start_time = None
        self._finalized_turns = []
        self._session_start = time.time()
    
    def add_text(self, text: str, speaker: str = None):
        """Add ASR-recognized text to the current turn buffer."""
        if not text or not text.strip():
            return
        
        if speaker and speaker != self._current_speaker and self._current_buffer:
            # Speaker change detected — finalize current turn first
            self._finalize_current()
        
        if speaker:
            self._current_speaker = speaker
        
        if self._turn_start_time is None:
            self._turn_start_time = time.time()
        
        if self._current_buffer:
            self._current_buffer += " " + text.strip()
        else:
            self._current_buffer = text.strip()
    
    def finalize_turn(self, speaker: str = None, text: str = None, 
                      asr_confidence: float = 1.0) -> Optional[Turn]:
        """
        Directly finalize a complete turn (used in replay mode).
        
        Args:
            speaker: Speaker identifier
            text: Complete turn text
            asr_confidence: ASR confidence score
            
        Returns:
            Finalized Turn object, or None if text is empty/too short
        """
        if text:
            self._current_buffer = text.strip()
        if speaker:
            self._current_speaker = speaker
        
        if not self._current_buffer or len(self._current_buffer) < self._min_turn_length_chars:
            self._current_buffer = ""
            return None
        
        self._turn_counter += 1
        elapsed = time.time() - self._session_start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        turn = Turn(
            turn_id=self._turn_counter,
            speaker=self._current_speaker,
            timestamp=f"{minutes:02d}:{seconds:02d}",
            text=self._current_buffer,
            asr_confidence=asr_confidence,
            finalized_at=time.perf_counter()
        )
        
        self._finalized_turns.append(turn)
        self._current_buffer = ""
        self._turn_start_time = None
        
        logger.debug(f"Turn {turn.turn_id} finalized: [{turn.speaker}] {turn.text[:60]}...")
        
        return turn
    
    def _finalize_current(self) -> Optional[Turn]:
        """Internal: finalize the current buffer as a turn."""
        return self.finalize_turn()
    
    def check_timeout(self) -> Optional[Turn]:
        """Check if current turn has exceeded max duration."""
        if self._turn_start_time and self._current_buffer:
            elapsed_ms = (time.time() - self._turn_start_time) * 1000
            if elapsed_ms >= self._max_turn_duration_ms:
                return self._finalize_current()
        return None
    
    @property
    def turn_count(self) -> int:
        return self._turn_counter
    
    @property
    def finalized_turns(self) -> List[Turn]:
        return self._finalized_turns.copy()
    
    @property
    def has_pending(self) -> bool:
        return bool(self._current_buffer)
