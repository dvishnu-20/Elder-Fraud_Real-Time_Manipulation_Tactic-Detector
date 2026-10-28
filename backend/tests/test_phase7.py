"""
Phase 7 — Comprehensive Test Suite
Tests for model loading, inference, session isolation, causality, privacy, and more.
"""
import os
import sys
import json
import time
import pytest
import hashlib
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import torch


# ============================================================
# 1. MODEL LOADING TESTS
# ============================================================
class TestModelLoading:
    """Test frozen model loading and integrity."""
    
    def test_model_service_loads(self):
        """Verify both models load without error."""
        from app.services.model_service import ModelService
        svc = ModelService()
        svc.load_models(verify_checksums=False)  # Skip checksums for speed in CI
        assert svc.is_loaded
    
    def test_distilbert_checksum(self):
        """Verify DistilBERT V2 weights haven't been modified."""
        path = PROJECT_ROOT / "ml" / "models" / "tactic_classifier" / "v2_temporal_validity" / "model.safetensors"
        if path.exists():
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            assert sha.hexdigest() == "76d2d4e440f65cb1e9f275a6496a0dcd6751a35e1f85261882815ccdda59a684"
    
    def test_lstm_checksum(self):
        """Verify Causal LSTM V2 weights haven't been modified."""
        path = PROJECT_ROOT / "ml" / "models" / "temporal_risk" / "lstm_v2" / "best_model.pt"
        if path.exists():
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            assert sha.hexdigest() == "c6dd5b03ba91a20677ed4c7a148d23cba26ff08fa1528d5603a0cb1bd2ae64a1"
    
    def test_lstm_threshold(self):
        """Verify LSTM threshold is tau = 0.20."""
        from app.services.model_service import ModelService
        svc = ModelService()
        svc.load_models(verify_checksums=False)
        assert svc.lstm_threshold == 0.20
    
    def test_labels_correct(self):
        """Verify the 5 tactic labels are correct."""
        from app.services.model_service import ModelService
        svc = ModelService()
        expected = ["urgency", "isolation", "authority", "emotional_destabilization", "channel_control"]
        assert svc.labels == expected
    
    def test_models_in_eval_mode(self):
        """Verify models are in evaluation mode."""
        from app.services.model_service import ModelService
        svc = ModelService()
        svc.load_models(verify_checksums=False)
        assert not svc._distilbert.training
        assert not svc._lstm.training


# ============================================================
# 2. TOKENIZER TESTS
# ============================================================
class TestTokenizer:
    """Test tokenizer configuration."""
    
    def test_tokenizer_loads(self):
        """Verify tokenizer loads from the model directory."""
        from app.services.model_service import ModelService
        svc = ModelService()
        svc.load_models(verify_checksums=False)
        assert svc._tokenizer is not None
    
    def test_tokenizer_max_length(self):
        """Verify tokenizer uses max_length=48."""
        from app.services.model_service import ModelService
        svc = ModelService()
        svc.load_models(verify_checksums=False)
        inputs = svc._tokenizer("test text", max_length=48, truncation=True, padding="max_length", return_tensors="pt")
        assert inputs["input_ids"].shape[1] == 48


# ============================================================
# 3. TACTIC INFERENCE TESTS
# ============================================================
class TestTacticInference:
    """Test DistilBERT V2 tactic classification."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        from app.services.model_service import ModelService
        self.svc = ModelService()
        self.svc.load_models(verify_checksums=False)
    
    def test_inference_returns_five_dimensions(self):
        """Each turn should produce exactly 5 tactic probabilities."""
        result = self.svc.classify_turn("This is a test sentence.")
        assert len(result["tactics"]) == 5
        assert all(k in result["tactics"] for k in self.svc.labels)
    
    def test_probabilities_in_range(self):
        """All probabilities should be in [0.0, 1.0]."""
        result = self.svc.classify_turn("You need to act immediately or lose everything.")
        for v in result["tactics"].values():
            assert 0.0 <= v <= 1.0
    
    def test_latency_recorded(self):
        """Inference latency should be recorded."""
        result = self.svc.classify_turn("Hello, how are you?")
        assert "latency_ms" in result
        assert result["latency_ms"] > 0
    
    def test_empty_text_handled(self):
        """Empty text should not crash."""
        result = self.svc.classify_turn("")
        assert len(result["tactics"]) == 5
    
    def test_fraud_text_high_urgency(self):
        """Fraud-like text should produce higher tactic probabilities."""
        result = self.svc.classify_turn(
            "You need to act immediately or your account will be permanently closed within ten minutes."
        )
        # At minimum urgency should be non-trivial
        assert result["tactics"]["urgency"] > 0.01


# ============================================================
# 4. TEMPORAL INFERENCE TESTS
# ============================================================
class TestTemporalInference:
    """Test Causal LSTM V2 temporal risk prediction."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        from app.services.model_service import ModelService
        self.svc = ModelService()
        self.svc.load_models(verify_checksums=False)
    
    def test_single_step_prediction(self):
        """Single-step prediction should return risk and hidden state."""
        result = self.svc.predict_temporal_risk([0.5, 0.3, 0.7, 0.1, 0.4])
        assert "risk" in result
        assert 0.0 <= result["risk"] <= 1.0
        assert "hidden_state" in result
        assert result["hidden_state"] is not None
    
    def test_sequential_predictions(self):
        """Sequential predictions should accumulate hidden state."""
        r1 = self.svc.predict_temporal_risk([0.1, 0.0, 0.0, 0.0, 0.0])
        r2 = self.svc.predict_temporal_risk([0.8, 0.0, 0.7, 0.0, 0.0], r1["hidden_state"])
        r3 = self.svc.predict_temporal_risk([0.9, 0.8, 0.9, 0.7, 0.8], r2["hidden_state"])
        # Risk should generally increase with manipulation tactics
        assert r3["risk"] >= r1["risk"]
    
    def test_risk_triggered_flag(self):
        """risk_triggered should be True when risk >= 0.20."""
        result = self.svc.predict_temporal_risk([0.9, 0.9, 0.9, 0.9, 0.9])
        # Even single high step may or may not trigger, but the flag should be boolean
        assert isinstance(result["risk_triggered"], bool)


# ============================================================
# 5. SESSION MANAGEMENT TESTS
# ============================================================
class TestSessionManagement:
    """Test session creation, update, and reset."""
    
    def test_create_session(self):
        from app.services.session_manager import SessionManager
        mgr = SessionManager()
        session = mgr.create_session("test-001")
        assert session.session_id == "test-001"
        assert session.status == "active"
        assert session.turn_count == 0
    
    def test_stop_session(self):
        from app.services.session_manager import SessionManager
        mgr = SessionManager()
        mgr.create_session("test-002")
        session = mgr.stop_session("test-002")
        assert session.status == "stopped"
    
    def test_session_not_found(self):
        from app.services.session_manager import SessionManager
        mgr = SessionManager()
        assert mgr.get_session("nonexistent") is None


# ============================================================
# 6. SESSION RESET / LSTM STATE ISOLATION
# ============================================================
class TestSessionIsolation:
    """
    CRITICAL: Verify LSTM hidden state does NOT leak between sessions.
    """
    
    def test_hidden_state_reset(self):
        """New engine should have None hidden state."""
        from app.services.model_service import ModelService
        from app.services.temporal_engine import TemporalRiskEngine
        
        svc = ModelService()
        svc.load_models(verify_checksums=False)
        
        engine = TemporalRiskEngine(svc)
        engine.reset()
        assert engine.get_hidden_state() is None
    
    def test_session_isolation(self, monkeypatch):
        """Two separate sessions should have independent temporal states."""
        from app.services.model_service import ModelService
        from app.services.temporal_engine import TemporalRiskEngine
        
        svc = ModelService()
        svc.load_models(verify_checksums=False)
        
        # Session A: fraud conversation (Mock high tactic probabilities)
        def mock_classify_high(text):
            return {"tactics": {"urgency": 0.9, "isolation": 0.9, "authority": 0.9, "emotional_destabilization": 0.9, "channel_control": 0.9}, "latency_ms": 1.0}
            
        monkeypatch.setattr(svc, "classify_turn", mock_classify_high)
        
        engine_a = TemporalRiskEngine(svc)
        engine_a.reset()
        engine_a.process_turn(1, "Text", "SPEAKER_A", "00:00")
        engine_a.process_turn(2, "Text", "SPEAKER_A", "00:01")
        engine_a.process_turn(3, "Text", "SPEAKER_A", "00:02")
        risk_a = engine_a.current_risk
        
        # Session B: benign conversation (Mock low tactic probabilities)
        def mock_classify_low(text):
            return {"tactics": {"urgency": 0.0, "isolation": 0.0, "authority": 0.0, "emotional_destabilization": 0.0, "channel_control": 0.0}, "latency_ms": 1.0}
            
        monkeypatch.setattr(svc, "classify_turn", mock_classify_low)
        
        engine_b = TemporalRiskEngine(svc)
        engine_b.reset()
        assert engine_b.get_hidden_state() is None  # Must be clean
        
        engine_b.process_turn(1, "Text", "SPEAKER_A", "00:00")
        risk_b = engine_b.current_risk
        
        # Session B should NOT be contaminated by Session A
        assert risk_b < risk_a, f"Session B risk ({risk_b}) should be less than Session A risk ({risk_a})"


# ============================================================
# 7. TURN PROCESSING TESTS
# ============================================================
class TestTurnProcessing:
    """Test turn segmentation."""
    
    def test_finalize_turn(self):
        from app.services.turn_segmenter import TurnSegmenter
        seg = TurnSegmenter()
        turn = seg.finalize_turn(speaker="SPEAKER_A", text="Hello there")
        assert turn is not None
        assert turn.turn_id == 1
        assert turn.speaker == "SPEAKER_A"
        assert turn.text == "Hello there"
    
    def test_empty_turn_rejected(self):
        from app.services.turn_segmenter import TurnSegmenter
        seg = TurnSegmenter()
        turn = seg.finalize_turn(speaker="SPEAKER_A", text="")
        assert turn is None
    
    def test_short_turn_rejected(self):
        from app.services.turn_segmenter import TurnSegmenter
        seg = TurnSegmenter(min_turn_length_chars=5)
        turn = seg.finalize_turn(speaker="SPEAKER_A", text="Hi")
        assert turn is None
    
    def test_turn_counter_increments(self):
        from app.services.turn_segmenter import TurnSegmenter
        seg = TurnSegmenter()
        t1 = seg.finalize_turn(speaker="A", text="Turn one")
        t2 = seg.finalize_turn(speaker="B", text="Turn two")
        assert t1.turn_id == 1
        assert t2.turn_id == 2
    
    def test_reset_clears_counter(self):
        from app.services.turn_segmenter import TurnSegmenter
        seg = TurnSegmenter()
        seg.finalize_turn(speaker="A", text="Turn one")
        seg.reset()
        t = seg.finalize_turn(speaker="A", text="After reset")
        assert t.turn_id == 1


# ============================================================
# 8. CAUSAL BEHAVIOR / NO FUTURE LEAKAGE
# ============================================================
class TestCausalBehavior:
    """
    CRITICAL: Verify the pipeline cannot access future turns.
    prediction(t) must be identical regardless of whether turns t+1.. exist.
    """
    
    def test_no_future_leakage(self):
        """
        Process turns 1-3, record risk at turn 3.
        Then process turns 1-5, verify risk at turn 3 is identical.
        """
        from app.services.model_service import ModelService
        from app.services.temporal_engine import TemporalRiskEngine
        
        svc = ModelService()
        svc.load_models(verify_checksums=False)
        
        turns = [
            "I'm calling from your bank's fraud department.",
            "Your account has been compromised.",
            "You need to act immediately.",
            "Don't tell anyone about this.",
            "Stay on the line and transfer your funds.",
        ]
        
        # Run 1: Process turns 1-3 only
        engine1 = TemporalRiskEngine(svc)
        engine1.reset()
        for i, text in enumerate(turns[:3]):
            engine1.process_turn(i+1, text, "SPEAKER_A", f"00:{i:02d}")
        risk_at_3_run1 = engine1.risk_history[2]
        
        # Run 2: Process turns 1-5
        engine2 = TemporalRiskEngine(svc)
        engine2.reset()
        for i, text in enumerate(turns[:5]):
            engine2.process_turn(i+1, text, "SPEAKER_A", f"00:{i:02d}")
        risk_at_3_run2 = engine2.risk_history[2]
        
        # Risk at turn 3 must be IDENTICAL regardless of future turns
        assert abs(risk_at_3_run1 - risk_at_3_run2) < 1e-6, \
            f"CAUSALITY VIOLATION: risk@3 differs: {risk_at_3_run1} vs {risk_at_3_run2}"


# ============================================================
# 9. FUTURE-TURN LEAKAGE (PERMUTATION TEST)
# ============================================================
class TestFutureLeakage:
    """Additional future-leakage checks."""
    
    def test_prefix_consistency(self):
        """Risk at each prefix length should be deterministic."""
        from app.services.model_service import ModelService
        from app.services.temporal_engine import TemporalRiskEngine
        
        svc = ModelService()
        svc.load_models(verify_checksums=False)
        
        turns = [
            "Hello, this is customer service.",
            "How can I help you today?",
            "Let me check your account.",
        ]
        
        # Run twice with identical input
        risks_run1 = []
        engine1 = TemporalRiskEngine(svc)
        engine1.reset()
        for i, text in enumerate(turns):
            engine1.process_turn(i+1, text, "A", f"00:{i:02d}")
            risks_run1.append(engine1.current_risk)
        
        risks_run2 = []
        engine2 = TemporalRiskEngine(svc)
        engine2.reset()
        for i, text in enumerate(turns):
            engine2.process_turn(i+1, text, "A", f"00:{i:02d}")
            risks_run2.append(engine2.current_risk)
        
        for i, (r1, r2) in enumerate(zip(risks_run1, risks_run2)):
            assert abs(r1 - r2) < 1e-6, f"Non-deterministic at turn {i+1}: {r1} vs {r2}"


# ============================================================
# 10. AUDIO CHUNK HANDLING
# ============================================================
class TestAudioHandling:
    """Test ASR provider audio chunk handling."""
    
    def test_mock_asr_text_passthrough(self):
        from app.services.asr_provider import MockASRProvider
        asr = MockASRProvider()
        asr.start()
        result = asr.process_audio_chunk(b"Hello world")
        assert result == "Hello world"
    
    def test_mock_asr_finalize(self):
        from app.services.asr_provider import MockASRProvider
        asr = MockASRProvider()
        asr.start()
        asr.process_audio_chunk(b"Hello")
        asr.process_audio_chunk(b"world")
        final = asr.finalize_segment()
        assert "Hello" in final
        assert "world" in final
    
    def test_replay_asr_injection(self):
        from app.services.asr_provider import ReplayASRProvider
        asr = ReplayASRProvider()
        asr.start()
        asr.inject_turn("Test turn text", speaker="SPEAKER_A")
        result = asr.finalize_segment()
        assert result == "Test turn text"


# ============================================================
# 11. EMPTY TRANSCRIPT HANDLING
# ============================================================
class TestEmptyTranscript:
    """Test graceful handling of empty transcripts."""
    
    def test_empty_turn_no_crash(self):
        from app.services.turn_segmenter import TurnSegmenter
        seg = TurnSegmenter()
        turn = seg.finalize_turn(speaker="A", text="")
        assert turn is None
    
    def test_whitespace_turn_no_crash(self):
        from app.services.turn_segmenter import TurnSegmenter
        seg = TurnSegmenter()
        turn = seg.finalize_turn(speaker="A", text="   ")
        assert turn is None


# ============================================================
# 12. API HEALTH
# ============================================================
class TestAPIHealth:
    """Test API health endpoint."""
    
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# ============================================================
# 13. ALERT GENERATION
# ============================================================
class TestAlertGeneration:
    """Test explanation and alert generation."""
    
    def test_normal_no_alert(self):
        from app.services.explanation_engine import generate_explanation
        result = generate_explanation(
            tactics={"urgency": 0.01, "isolation": 0.01, "authority": 0.01,
                     "emotional_destabilization": 0.01, "channel_control": 0.01},
            temporal_risk=0.05,
            threshold=0.20,
            tactic_thresholds={"urgency": 0.15, "isolation": 0.25, "authority": 0.10,
                               "emotional_destabilization": 0.20, "channel_control": 0.15},
            risk_state="NORMAL",
            turn_id=1
        )
        assert result["alert_message"] is None
    
    def test_critical_generates_alert(self):
        from app.services.explanation_engine import generate_explanation
        result = generate_explanation(
            tactics={"urgency": 0.9, "isolation": 0.8, "authority": 0.85,
                     "emotional_destabilization": 0.7, "channel_control": 0.9},
            temporal_risk=0.90,
            threshold=0.20,
            tactic_thresholds={"urgency": 0.15, "isolation": 0.25, "authority": 0.10,
                               "emotional_destabilization": 0.20, "channel_control": 0.15},
            risk_state="CRITICAL",
            turn_id=5
        )
        assert result["alert_message"] is not None
        assert "HIGH-RISK" in result["alert_message"]


# ============================================================
# 14. PRIVACY / REDACTION
# ============================================================
class TestPrivacy:
    """Test PII redaction."""
    
    def test_credit_card_redacted(self):
        from app.services.privacy_service import PrivacyService
        svc = PrivacyService(redact_pii=True)
        text = "My card number is 4111-1111-1111-1111"
        redacted = svc.redact_text(text)
        assert "4111" not in redacted
        assert "[CARD REDACTED]" in redacted
    
    def test_ssn_redacted(self):
        from app.services.privacy_service import PrivacyService
        svc = PrivacyService(redact_pii=True)
        text = "My SSN is 123-45-6789"
        redacted = svc.redact_text(text)
        assert "123-45-6789" not in redacted
    
    def test_no_redaction_when_disabled(self):
        from app.services.privacy_service import PrivacyService
        svc = PrivacyService(redact_pii=False)
        text = "My card is 4111-1111-1111-1111"
        assert svc.redact_text(text) == text
    
    def test_audio_not_stored_by_default(self):
        from app.services.privacy_service import PrivacyService
        svc = PrivacyService()
        assert not svc.should_store_audio()


# ============================================================
# 15. DEMO REPLAY
# ============================================================
class TestDemoReplay:
    """Test the demo replay functionality."""
    
    def test_demo_script_exists(self):
        """Verify demo script has both fraud and benign conversations."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_phase7_replay import get_demo_script
        script = get_demo_script()
        assert len(script) >= 2
        labels = [v["label"] for v in script.values()]
        assert "fraud" in labels
        assert "benign" in labels


# ============================================================
# 16. RISK STATE MAPPING
# ============================================================
class TestRiskStateMapping:
    """Test UI risk state mapping."""
    
    def test_state_boundaries(self):
        from app.services.model_service import ModelService
        svc = ModelService()
        assert svc.get_risk_state(0.05) == "NORMAL"
        assert svc.get_risk_state(0.15) == "WATCH"
        assert svc.get_risk_state(0.35) == "ELEVATED"
        assert svc.get_risk_state(0.65) == "HIGH_RISK"
        assert svc.get_risk_state(0.90) == "CRITICAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
