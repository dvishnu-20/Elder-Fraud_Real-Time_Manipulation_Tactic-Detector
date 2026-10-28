# Elder-Fraud Real-Time Manipulation-Tactic Detector

## 1. Project Overview
A software-based AI system that detects psychological manipulation tactics during live or near-real-time voice conversations, with a focus on elder-fraud and scam calls.

## 2. Problem Statement
Current scam detection relies heavily on static blacklists, phone numbers, and keyword spotting, which are easily evaded. This system focuses on structural manipulation patterns independent of specific topics.

## 3. Novel Approach
We track the **progression of manipulation tactics over time** (e.g., Authority → Emotional Destabilization → Urgency → Isolation → Channel Control) to calculate a cumulative risk score rather than treating tactics in isolation.

## 4. Architecture
The MVP consists of:
- **Backend**: FastAPI for orchestration, WebSockets for streaming.
- **Frontend**: React (Vite) dashboard for live monitoring.
- **State**: In-memory dictionary (Mocked for MVP, easily replaceable with DB/Redis later).
- **ML Engine**: (Phase 2+) Modular pipeline supporting streaming ASR, diarization, a tactic classifier, and a sequence model.

## 5. Technology Stack
- **Python 3.11+, FastAPI, Pydantic, Uvicorn, WebSockets**
- **React, TypeScript, Vite, TailwindCSS**
- **Redis, Docker, Docker Compose**

## 6. Installation & 7. Running Locally (Without Docker)
Requires Node.js and Python 3.11+.
```bash
# Terminal 1 - Backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

## 8. Running with Docker (Recommended)
```bash
docker compose up --build
```

## 9. Demo Instructions
1. Navigate to `http://localhost:5173`
2. Click "Start Demo"
3. Observe the simulated conversation as the tactic classifications update, the composite risk score increases, and the intervention warning appears.

## 14. Privacy Considerations
- `STORE_AUDIO=false` and `STORE_TRANSCRIPT=false` by default.
- Data is kept in ephemeral memory during active calls.

## 15. Limitations
- MVP relies on mocked ML models for early validation.
- Does not have real telecom integration yet.
- Audio consent and wiretap laws apply.

## 16. Future Work
- Integration with real Transformer-based sequence models.
- Better streaming ASR integration.
- Carrier-level telecom support.
