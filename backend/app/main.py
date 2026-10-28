"""
Phase 7 — FastAPI Application
Elder-Fraud Real-Time Manipulation-Tactic Detector

Loads frozen DistilBERT V2 and Causal LSTM V2 at startup.
No model weights are modified during runtime.
"""
import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api import routes, websocket
from app.services.model_service import model_service
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, cleanup on shutdown."""
    logger.info("=" * 60)
    logger.info("ELDER-FRAUD DETECTOR V2 — Phase 7 Real-Time Pipeline")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Device: CPU/CUDA auto-detect")
    logger.info(f"LSTM threshold: tau = {settings.LSTM_DECISION_THRESHOLD}")
    
    # Load frozen models
    try:
        distilbert_path = str(PROJECT_ROOT / settings.DISTILBERT_PATH)
        lstm_path = str(PROJECT_ROOT / settings.LSTM_PATH)
        
        model_service.load_models(
            distilbert_path=distilbert_path,
            lstm_path=lstm_path,
            verify_checksums=settings.VERIFY_CHECKSUMS
        )
        logger.info("Models loaded successfully ✓")
        logger.info(f"DistilBERT load time: {model_service.get_load_times().get('distilbert', 0):.2f}s")
        logger.info(f"LSTM load time: {model_service.get_load_times().get('lstm', 0):.2f}s")
    except Exception as e:
        logger.error(f"FATAL: Model loading failed: {e}")
        logger.error("The application will start but model inference will not work.")
        logger.error("Check model paths and integrity.")
    
    logger.info("=" * 60)
    logger.info("Server ready. Phase 7 pipeline operational.")
    logger.info("=" * 60)
    
    yield  # Application runs
    
    logger.info("Shutting down Phase 7 pipeline...")


app = FastAPI(
    title="Elder-Fraud Real-Time Manipulation-Tactic Detector",
    description="Phase 7: Real-time streaming pipeline using frozen DistilBERT V2 + Causal LSTM V2",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api")
app.include_router(websocket.router, prefix="/ws")
