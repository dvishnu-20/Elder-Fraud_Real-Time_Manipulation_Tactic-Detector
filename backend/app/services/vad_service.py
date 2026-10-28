"""
Phase 7 — Voice Activity Detection (VAD) Service
Abstraction layer for detecting speech segments in audio.
VAD is NOT a fraud detector — it only separates speech from silence.
"""
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VADProvider(ABC):
    """Abstract VAD interface."""
    
    @abstractmethod
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Return True if the audio chunk contains speech."""
        pass
    
    @abstractmethod
    def get_silence_duration_ms(self) -> float:
        """Return how long silence has been continuous (in ms)."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset VAD state."""
        pass


class EnergyVAD(VADProvider):
    """
    Simple energy-based VAD for development.
    Checks if the RMS energy of an audio chunk exceeds a threshold.
    """
    
    def __init__(self, energy_threshold: float = 0.01, sample_rate: int = 16000):
        self._energy_threshold = energy_threshold
        self._sample_rate = sample_rate
        self._silence_start = None
        self._last_speech_time = time.time()
    
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk has sufficient energy to be speech."""
        if len(audio_chunk) < 2:
            return False
        
        # Compute RMS energy from PCM16 samples
        import struct
        n_samples = len(audio_chunk) // 2
        if n_samples == 0:
            return False
        
        samples = struct.unpack(f"<{n_samples}h", audio_chunk[:n_samples * 2])
        rms = (sum(s * s for s in samples) / n_samples) ** 0.5
        normalized_rms = rms / 32768.0  # Normalize to [0, 1]
        
        is_speech = normalized_rms > self._energy_threshold
        
        if is_speech:
            self._last_speech_time = time.time()
            self._silence_start = None
        else:
            if self._silence_start is None:
                self._silence_start = time.time()
        
        return is_speech
    
    def get_silence_duration_ms(self) -> float:
        """Return duration of continuous silence in milliseconds."""
        if self._silence_start is None:
            return 0.0
        return (time.time() - self._silence_start) * 1000.0
    
    def reset(self) -> None:
        self._silence_start = None
        self._last_speech_time = time.time()


class MockVAD(VADProvider):
    """
    Mock VAD that always reports speech detected.
    Used in replay/demo mode where we already have text.
    """
    
    def __init__(self):
        self._silence_duration = 0.0
    
    def is_speech(self, audio_chunk: bytes) -> bool:
        return True
    
    def get_silence_duration_ms(self) -> float:
        return self._silence_duration
    
    def set_silence_duration(self, ms: float):
        """Manually set silence duration for testing."""
        self._silence_duration = ms
    
    def reset(self) -> None:
        self._silence_duration = 0.0


def create_vad(vad_type: str = "mock", **kwargs) -> VADProvider:
    """Factory function for VAD providers."""
    if vad_type == "energy":
        return EnergyVAD(**kwargs)
    return MockVAD()
