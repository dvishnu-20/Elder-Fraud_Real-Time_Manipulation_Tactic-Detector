"""
Phase 7 — Backend Configuration
All thresholds and model paths drawn from frozen Phase 5/6 config.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Frozen model paths (Phase 5/6 validated)
    DISTILBERT_PATH: str = "ml/models/tactic_classifier/v2_temporal_validity/"
    LSTM_PATH: str = "ml/models/temporal_risk/lstm_v2/"
    
    # Frozen thresholds — do NOT change without documented experiment
    LSTM_DECISION_THRESHOLD: float = 0.20  # tau = 0.20
    
    # UI severity bands (presentation only, NOT scientifically validated)
    RISK_BAND_WATCH: float = 0.10
    RISK_BAND_ELEVATED: float = 0.20
    RISK_BAND_HIGH: float = 0.50
    RISK_BAND_CRITICAL: float = 0.80
    
    # ASR
    ASR_PROVIDER: str = "mock"  # mock, replay, whisper_local
    ASR_SAMPLE_RATE: int = 16000
    ASR_CHUNK_DURATION_MS: int = 500
    
    # VAD
    VAD_SILENCE_THRESHOLD_MS: float = 1500.0
    VAD_MAX_TURN_DURATION_MS: float = 30000.0
    VAD_ENERGY_THRESHOLD: float = 0.01
    
    # Turn segmentation
    MIN_TURN_LENGTH_CHARS: int = 3
    
    # Session
    SESSION_MAX_DURATION_MIN: int = 60
    SESSION_IDLE_TIMEOUT_MIN: int = 10
    
    # Privacy
    STORE_AUDIO: bool = False
    STORE_TRANSCRIPT: bool = True
    REDACT_PII: bool = True
    
    # Model integrity
    VERIFY_CHECKSUMS: bool = True
    
    class Config:
        env_file = ".env"


settings = Settings()
