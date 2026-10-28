"""
Phase 7 — ASR Provider Abstraction
Replaceable ASR module with mock and optional Whisper implementations.
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class ASRProvider(ABC):
    """
    Abstract interface for speech-to-text providers.
    The rest of the application does NOT tightly couple to any specific ASR engine.
    """
    
    @abstractmethod
    def start(self) -> None:
        """Initialize the ASR provider."""
        pass
    
    @abstractmethod
    def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """
        Process a chunk of audio data.
        Returns partial transcript or None if no speech detected.
        """
        pass
    
    @abstractmethod
    def finalize_segment(self) -> Optional[str]:
        """
        Finalize the current speech segment.
        Returns the final transcript for the segment.
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the ASR provider and release resources."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name."""
        pass


class MockASRProvider(ASRProvider):
    """
    Mock ASR provider for development and testing.
    Does NOT fabricate live transcription.
    Instead, accepts pre-formed text and passes it through the same interface,
    allowing deterministic testing of the full pipeline.
    """
    
    def __init__(self):
        self._started = False
        self._buffer = ""
        self._confidence = 0.95  # mock confidence
    
    def start(self) -> None:
        self._started = True
        self._buffer = ""
        logger.info("MockASR started")
    
    def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """In mock mode, audio_data is expected to be UTF-8 encoded text."""
        if not self._started:
            return None
        try:
            text = audio_data.decode("utf-8", errors="ignore").strip()
            if text:
                self._buffer += " " + text if self._buffer else text
                return text
        except Exception:
            pass
        return None
    
    def finalize_segment(self) -> Optional[str]:
        """Return accumulated text and reset buffer."""
        if not self._buffer:
            return None
        result = self._buffer.strip()
        self._buffer = ""
        return result
    
    def stop(self) -> None:
        self._started = False
        self._buffer = ""
        logger.info("MockASR stopped")
    
    def get_name(self) -> str:
        return "MockASR"
    
    def get_confidence(self) -> float:
        return self._confidence


class ReplayASRProvider(ASRProvider):
    """
    ASR provider for offline replay mode.
    Feeds pre-recorded conversation turns through the pipeline
    as if they were arriving from live ASR.
    """
    
    def __init__(self):
        self._started = False
        self._current_text = None
    
    def start(self) -> None:
        self._started = True
        self._current_text = None
        logger.info("ReplayASR started")
    
    def inject_turn(self, text: str, speaker: str = "UNKNOWN", confidence: float = 1.0):
        """Inject a pre-recorded turn directly (bypasses audio processing)."""
        self._current_text = {
            "text": text,
            "speaker": speaker,
            "confidence": confidence
        }
    
    def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """Not used in replay mode."""
        return None
    
    def finalize_segment(self) -> Optional[str]:
        """Return the injected text."""
        if self._current_text:
            text = self._current_text["text"]
            self._current_text = None
            return text
        return None
    
    def stop(self) -> None:
        self._started = False
        self._current_text = None
        logger.info("ReplayASR stopped")
    
    def get_name(self) -> str:
        return "ReplayASR"


def create_asr_provider(provider_type: str = "mock") -> ASRProvider:
    """
    Factory function to create ASR providers.
    
    Supported providers:
    - 'mock': MockASRProvider (no external dependencies)
    - 'replay': ReplayASRProvider (offline replay)
    """
    providers = {
        "mock": MockASRProvider,
        "replay": ReplayASRProvider,
    }
    
    if provider_type not in providers:
        logger.warning(f"Unknown ASR provider '{provider_type}', falling back to mock")
        provider_type = "mock"
    
    provider = providers[provider_type]()
    logger.info(f"Created ASR provider: {provider.get_name()}")
    return provider
