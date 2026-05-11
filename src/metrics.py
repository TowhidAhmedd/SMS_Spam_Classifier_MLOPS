"""
Prometheus metrics for monitoring the spam classifier API.
Exposes /metrics endpoint for Prometheus scraping.
"""
import time

from fastapi import Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# ── Metrics definitions ───────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "spam_api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "spam_api_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

PREDICTION_COUNT = Counter(
    "spam_predictions_total",
    "Total predictions",
    ["label"]
)

SPAM_PROBABILITY_HIST = Histogram(
    "spam_probability_distribution",
    "Distribution of spam probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

FEEDBACK_COUNT = Counter(
    "spam_feedback_total",
    "Total feedback submissions",
    ["correct"]
)

MODEL_LOADED_GAUGE = Gauge(
    "spam_model_loaded",
    "1 if model is loaded, 0 otherwise"
)

RETRAIN_COUNT = Counter(
    "spam_retrain_total",
    "Total retraining runs triggered"
)


# ── Middleware ────────────────────────────────────────────────────────────────

async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    return response


# ── Helper functions ──────────────────────────────────────────────────────────

def track_prediction(label: str, spam_probability: float):
    PREDICTION_COUNT.labels(label=label).inc()
    SPAM_PROBABILITY_HIST.observe(spam_probability)


def track_feedback(correct: bool):
    FEEDBACK_COUNT.labels(correct=str(correct).lower()).inc()


def set_model_loaded(loaded: bool):
    MODEL_LOADED_GAUGE.set(1 if loaded else 0)


def track_retrain():
    RETRAIN_COUNT.inc()


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
