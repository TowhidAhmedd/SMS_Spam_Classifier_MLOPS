"""
Email Spam Classifier — Production FastAPI Application
Full MLOps: FastAPI + PostgreSQL + MLflow + Prometheus + Sentry + Feedback Loop
"""
import os
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import sentry_sdk
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from src.config import get_settings
from src.database import init_db
from src.model import load_model, predict, predict_single
from src.metrics import (
    prometheus_middleware, metrics_response,
    track_prediction, track_feedback, set_model_loaded
)
from src.feedback import save_feedback, get_feedback_stats, log_prediction


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

# ── Sentry ────────────────────────────────────────────────────────────────────

if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        traces_sample_rate=0.1,
        environment="production",
    )
    logger.info("Sentry initialized.")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Spam Classifier API...")
    # Init PostgreSQL tables
    try:
        init_db()
    except Exception as e:
        logger.error(f"DB init failed: {e}")

    # Load model into memory
    try:
        load_model()
        set_model_loaded(True)
        logger.info("Model loaded successfully.")
    except FileNotFoundError as e:
        set_model_loaded(False)
        logger.error(str(e))

    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Email Spam Classifier API",
    description="""
## 📧 Production Email Spam Classifier

End-to-end MLOps project with:
- **ML Model**: TF-IDF + Logistic Regression (scikit-learn)
- **Experiment Tracking**: MLflow + DagsHub
- **Database**: PostgreSQL (Render)
- **Monitoring**: Prometheus + Sentry
- **Feedback Loop**: Auto-retraining on user corrections
- **Deployment**: Docker + Render

### Quick Start
1. POST `/predict` with email text
2. Submit feedback via POST `/feedback`
3. Monitor at `/metrics` (Prometheus)
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(prometheus_middleware)


# ── Schemas ───────────────────────────────────────────────────────────────────

class EmailRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Email body text")

    @field_validator("text")
    @classmethod
    def strip_and_validate(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty or whitespace only")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {"text": "Congratulations! You've won a $1000 gift card. Click here to claim!"}
        }
    }


class BatchEmailRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        cleaned = [t.strip() for t in v]
        if any(len(t) == 0 for t in cleaned):
            raise ValueError("All texts must be non-empty")
        return cleaned


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1)
    predicted: str = Field(..., pattern="^(spam|ham)$")
    correct: bool
    true_label: Optional[str] = Field(None, pattern="^(spam|ham)$")
    confidence: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Meeting at 3pm tomorrow",
                "predicted": "spam",
                "correct": False,
                "true_label": "ham",
                "confidence": 0.87
            }
        }
    }


class PredictionResult(BaseModel):
    label: str
    is_spam: bool
    spam_probability: float
    ham_probability: float
    confidence: float


class SinglePredictionResponse(BaseModel):
    success: bool
    prediction: PredictionResult
    text_preview: str
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    success: bool
    count: int
    predictions: List[PredictionResult]
    processing_time_ms: float


# ── System Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(path):
        # with open(path) as f:
        #     return HTMLResponse(content=f.read())
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Spam Classifier API</h1><p><a href='/docs'>Docs</a></p>")


@app.get("/health", tags=["System"])
async def health():
    """Health check — used by Render and load balancers."""
    try:
        load_model()
        model_loaded = True
        set_model_loaded(True)
    except Exception:
        model_loaded = False
        set_model_loaded(False)

    return {
        "status":       "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version":      settings.app_version,
        "database":     "PostgreSQL",
    }


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return metrics_response()


@app.get("/model/info", tags=["System"])
async def model_info():
    """Return last training metrics from MLflow run."""
    if not os.path.exists("models/metadata.json"):
        raise HTTPException(status_code=404, detail="No model metadata found. Train the model first.")
    with open("models/metadata.json") as f:
        return JSONResponse(content=json.load(f))


# ── Prediction Routes ─────────────────────────────────────────────────────────

@app.post("/predict", response_model=SinglePredictionResponse, tags=["Prediction"])
async def predict_email(request: EmailRequest):
    """Classify a single email as spam or ham."""
    start = time.time()
    try:
        result = predict_single(request.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.time() - start) * 1000, 2)

    # Track in Prometheus
    track_prediction(result["label"], result["spam_probability"])

    # Log to PostgreSQL
    log_prediction(
        text_preview=request.text[:120],
        predicted_label=result["label"],
        spam_probability=result["spam_probability"],
        confidence=result["confidence"],
        latency_ms=latency,
    )

    return SinglePredictionResponse(
        success=True,
        prediction=PredictionResult(**result),
        text_preview=request.text[:100] + ("…" if len(request.text) > 100 else ""),
        processing_time_ms=latency,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchEmailRequest):
    """Classify up to 50 emails in one request."""
    start = time.time()
    try:
        results = predict(request.texts)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    for r in results:
        track_prediction(r["label"], r["spam_probability"])

    return BatchPredictionResponse(
        success=True,
        count=len(results),
        predictions=[PredictionResult(**r) for r in results],
        processing_time_ms=round((time.time() - start) * 1000, 2),
    )


# ── Feedback Routes ───────────────────────────────────────────────────────────

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a prediction result.
    Wrong predictions accumulate and trigger auto-retraining.
    """
    if not request.correct and request.true_label is None:
        raise HTTPException(
            status_code=422,
            detail="'true_label' is required when 'correct' is false."
        )

    total = save_feedback(
        text=request.text,
        predicted=request.predicted,
        correct=request.correct,
        true_label=request.true_label,
        confidence=request.confidence,
    )
    track_feedback(request.correct)

    return {
        "success":        True,
        "message":        "Feedback saved. Thank you!",
        "total_feedback": total,
    }


@app.get("/feedback/stats", tags=["Feedback"])
async def feedback_statistics():
    """Return feedback statistics and retraining progress."""
    return get_feedback_stats()


# ── Examples ──────────────────────────────────────────────────────────────────

@app.get("/examples", tags=["Prediction"])
async def examples():
    """Sample spam and ham emails for testing."""
    return {
        "spam_examples": [
            "WINNER!! You've been selected for a $1,000 gift card. Call now!",
            "Free entry to win FA Cup Final tickets! Text FA to 87121",
            "URGENT: Your account has been suspended. Verify now: bit.ly/xxxx",
            "Congratulations! You've won a Nokia 6610. Claim at www.prize.com",
        ],
        "ham_examples": [
            "Hey, are we still on for lunch tomorrow at 1pm?",
            "Please review the attached Q3 report before Thursday's meeting.",
            "Your Amazon order has shipped and arrives by Friday.",
            "Don't forget — dentist appointment at 3pm today.",
        ],
    }
