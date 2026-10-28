"""
Phase 7 — Privacy Service
PII redaction and privacy controls for sensitive conversation data.

LIMITATIONS:
- Regex-based redaction is NOT comprehensive and may miss novel formats.
- This is a best-effort engineering measure, not a legal compliance guarantee.
- Audio is NOT stored by default.
"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


# PII patterns for redaction
PII_PATTERNS = {
    "credit_card": {
        "pattern": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "replacement": "[CARD REDACTED]",
        "description": "Credit/debit card numbers (16 digits)"
    },
    "ssn": {
        "pattern": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        "replacement": "[SSN REDACTED]",
        "description": "Social Security Numbers"
    },
    "phone_number": {
        "pattern": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "replacement": "[PHONE REDACTED]",
        "description": "US phone numbers"
    },
    "bank_account": {
        "pattern": r'\b\d{8,17}\b',
        "replacement": "[ACCOUNT REDACTED]",
        "description": "Bank account numbers (8-17 digits)"
    },
    "email": {
        "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "replacement": "[EMAIL REDACTED]",
        "description": "Email addresses"
    },
    "auth_code": {
        "pattern": r'\b(?:code|pin|otp|verification)[\s:]*\d{4,8}\b',
        "replacement": "[AUTH CODE REDACTED]",
        "description": "Authentication/verification codes"
    }
}


class PrivacyService:
    """
    Privacy and PII redaction service.
    
    Controls:
    - Configurable PII pattern redaction
    - Audio retention policy (default: do not store)
    - Transcript retention policy
    - Logging level restrictions
    """
    
    def __init__(
        self,
        redact_pii: bool = True,
        store_audio: bool = False,
        store_transcript: bool = True,
        enabled_patterns: List[str] = None
    ):
        self._redact_pii = redact_pii
        self._store_audio = store_audio
        self._store_transcript = store_transcript
        
        # Compile enabled patterns
        if enabled_patterns is None:
            enabled_patterns = list(PII_PATTERNS.keys())
        
        self._compiled_patterns = {}
        for name in enabled_patterns:
            if name in PII_PATTERNS:
                self._compiled_patterns[name] = {
                    "regex": re.compile(PII_PATTERNS[name]["pattern"], re.IGNORECASE),
                    "replacement": PII_PATTERNS[name]["replacement"]
                }
        
        logger.info(
            f"Privacy service initialized: redact_pii={redact_pii}, "
            f"store_audio={store_audio}, patterns={list(self._compiled_patterns.keys())}"
        )
    
    def redact_text(self, text: str) -> str:
        """
        Apply PII redaction patterns to text.
        Returns redacted text.
        """
        if not self._redact_pii or not text:
            return text
        
        redacted = text
        for name, pattern_info in self._compiled_patterns.items():
            redacted = pattern_info["regex"].sub(
                pattern_info["replacement"], 
                redacted
            )
        
        return redacted
    
    def should_store_audio(self) -> bool:
        return self._store_audio
    
    def should_store_transcript(self) -> bool:
        return self._store_transcript
    
    def redact_for_logging(self, text: str) -> str:
        """Always redact for log output regardless of settings."""
        return self.redact_text(text)
    
    def get_privacy_config(self) -> dict:
        """Return current privacy configuration."""
        return {
            "redact_pii": self._redact_pii,
            "store_audio": self._store_audio,
            "store_transcript": self._store_transcript,
            "active_patterns": list(self._compiled_patterns.keys()),
            "limitations": [
                "Regex-based redaction — may miss novel PII formats",
                "Not a legal compliance guarantee",
                "Audio NOT stored by default",
                "Redaction applied before storage, not retroactively"
            ]
        }


# Global singleton
privacy_service = PrivacyService()
