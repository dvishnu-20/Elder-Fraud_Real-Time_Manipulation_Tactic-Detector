"""
Phase 7 — Model Service
Loads frozen DistilBERT V2 and Causal LSTM V2 once at startup.
Models are placed in eval mode with no gradient computation.
NO weights are modified. NO retraining occurs.
"""
import os
import sys
import json
import hashlib
import time
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# Add project root to path for ml.temporal_model import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.temporal_model import CausalLSTMRiskModel


def _sha256_file(filepath: str) -> str:
    """Compute SHA256 of a file for integrity verification."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


class ModelService:
    """
    Singleton-style service that loads and holds the frozen Phase 5/6 models.
    
    Models loaded:
    - DistilBERT V2 turn-level tactic classifier (5 binary heads)
    - Causal LSTM V2 temporal risk model (input_dim=5, hidden_dim=64)
    
    Both are placed in eval() mode with torch.no_grad() used at inference time.
    """
    
    def __init__(self):
        self._distilbert = None
        self._tokenizer = None
        self._lstm = None
        self._device = None
        self._labels = ["urgency", "isolation", "authority", "emotional_destabilization", "channel_control"]
        self._tactic_thresholds = {}
        self._lstm_threshold = 0.20  # tau = 0.20, frozen
        self._loaded = False
        self._load_times = {}
        
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def labels(self) -> list:
        return self._labels
    
    @property
    def lstm_threshold(self) -> float:
        return self._lstm_threshold
    
    @property
    def tactic_thresholds(self) -> dict:
        return self._tactic_thresholds
    
    def load_models(self, 
                    distilbert_path: str = None,
                    lstm_path: str = None,
                    verify_checksums: bool = True):
        """
        Load both frozen models into memory.
        
        Args:
            distilbert_path: Path to DistilBERT V2 directory
            lstm_path: Path to LSTM V2 directory
            verify_checksums: If True, verify SHA256 against Phase 6 manifest
        """
        if distilbert_path is None:
            distilbert_path = str(PROJECT_ROOT / "ml" / "models" / "tactic_classifier" / "v2_temporal_validity")
        if lstm_path is None:
            lstm_path = str(PROJECT_ROOT / "ml" / "models" / "temporal_risk" / "lstm_v2")
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self._device}")
        
        # --- Load DistilBERT V2 ---
        t0 = time.perf_counter()
        logger.info(f"Loading DistilBERT V2 from: {distilbert_path}")
        
        if verify_checksums:
            safetensors_path = os.path.join(distilbert_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                actual_sha = _sha256_file(safetensors_path)
                expected_sha = "76d2d4e440f65cb1e9f275a6496a0dcd6751a35e1f85261882815ccdda59a684"
                if actual_sha != expected_sha:
                    raise RuntimeError(
                        f"DistilBERT V2 checksum mismatch!\n"
                        f"Expected: {expected_sha}\n"
                        f"Actual:   {actual_sha}\n"
                        f"The model weights may have been modified. STOPPING."
                    )
                logger.info("DistilBERT V2 checksum verified ✓")
        
        self._tokenizer = AutoTokenizer.from_pretrained(distilbert_path)
        self._distilbert = AutoModelForSequenceClassification.from_pretrained(distilbert_path)
        self._distilbert.to(self._device)
        self._distilbert.eval()  # FROZEN — no training
        
        self._load_times["distilbert"] = time.perf_counter() - t0
        logger.info(f"DistilBERT V2 loaded in {self._load_times['distilbert']:.2f}s")
        
        # Load tactic thresholds
        threshold_path = os.path.join(distilbert_path, "threshold_config.json")
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                tc = json.load(f)
            self._tactic_thresholds = tc.get("threshold_mapping", {})
            logger.info(f"Tactic thresholds loaded: {self._tactic_thresholds}")
        
        # --- Load Causal LSTM V2 ---
        t0 = time.perf_counter()
        logger.info(f"Loading Causal LSTM V2 from: {lstm_path}")
        
        lstm_config_path = os.path.join(lstm_path, "config.json")
        with open(lstm_config_path) as f:
            lstm_config = json.load(f)
        
        lstm_weights_path = os.path.join(lstm_path, "best_model.pt")
        
        if verify_checksums:
            actual_sha = _sha256_file(lstm_weights_path)
            expected_sha = "c6dd5b03ba91a20677ed4c7a148d23cba26ff08fa1528d5603a0cb1bd2ae64a1"
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Causal LSTM V2 checksum mismatch!\n"
                    f"Expected: {expected_sha}\n"
                    f"Actual:   {actual_sha}\n"
                    f"The model weights may have been modified. STOPPING."
                )
            logger.info("Causal LSTM V2 checksum verified ✓")
        
        self._lstm = CausalLSTMRiskModel(
            input_dim=lstm_config["input_dim"],
            hidden_dim=lstm_config["hidden_dim"],
            num_layers=lstm_config["num_layers"],
            dropout=lstm_config["dropout"],
            pooling_head=lstm_config["pooling_head"]
        )
        
        state_dict = torch.load(lstm_weights_path, map_location=self._device, weights_only=True)
        self._lstm.load_state_dict(state_dict)
        self._lstm.to(self._device)
        self._lstm.eval()  # FROZEN — no training
        
        # Load LSTM threshold
        lstm_threshold_path = os.path.join(lstm_path, "threshold_config.json")
        if os.path.exists(lstm_threshold_path):
            with open(lstm_threshold_path) as f:
                lt = json.load(f)
            self._lstm_threshold = lt.get("calibrated_threshold", 0.20)
        
        self._load_times["lstm"] = time.perf_counter() - t0
        logger.info(f"Causal LSTM V2 loaded in {self._load_times['lstm']:.2f}s")
        logger.info(f"LSTM decision threshold: tau = {self._lstm_threshold}")
        
        self._loaded = True
        logger.info("All models loaded successfully. Phase 7 model service ready.")
    
    @torch.no_grad()
    def classify_turn(self, text: str) -> dict:
        """
        Run frozen DistilBERT V2 on a single turn of text.
        
        Returns dict with 5 continuous tactic probabilities in [0.0, 1.0].
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        t0 = time.perf_counter()
        
        inputs = self._tokenizer(
            text,
            max_length=48,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self._device)
        
        outputs = self._distilbert(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze(0).cpu().tolist()
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        result = {}
        for i, label in enumerate(self._labels):
            result[label] = round(probs[i], 6)
        
        return {
            "tactics": result,
            "latency_ms": round(latency_ms, 2)
        }
    
    @torch.no_grad()
    def predict_temporal_risk(self, tactic_probs: list, prev_hidden_state=None):
        """
        Run one causal LSTM step using predict_causal_step.
        
        Args:
            tactic_probs: list of 5 floats [urg, iso, aut, emo, chn]
            prev_hidden_state: tuple (h, c) from previous step, or None for first turn
            
        Returns:
            risk: float in [0.0, 1.0]
            risk_triggered: bool (risk >= tau)
            hidden_state: tuple (h, c) for next step
            latency_ms: float
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        t0 = time.perf_counter()
        
        # Shape: (1, 1, 5) — batch=1, seq_len=1, features=5
        step_tensor = torch.tensor([[tactic_probs]], dtype=torch.float32, device=self._device)
        
        risk, new_hidden = self._lstm.predict_causal_step(step_tensor, prev_hidden_state)
        risk_val = risk.item()
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return {
            "risk": round(risk_val, 6),
            "risk_triggered": risk_val >= self._lstm_threshold,
            "hidden_state": new_hidden,
            "latency_ms": round(latency_ms, 2)
        }
    
    def get_risk_state(self, risk: float) -> str:
        """Map continuous risk to UI severity band."""
        if risk >= 0.80:
            return "CRITICAL"
        elif risk >= 0.50:
            return "HIGH_RISK"
        elif risk >= 0.20:
            return "ELEVATED"
        elif risk >= 0.10:
            return "WATCH"
        else:
            return "NORMAL"
    
    def get_load_times(self) -> dict:
        return self._load_times.copy()


# Global singleton
model_service = ModelService()
