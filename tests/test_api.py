"""API integration tests."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

MOCK_SPAM = {"label": "spam", "is_spam": True,  "spam_probability": 0.95, "ham_probability": 0.05, "confidence": 0.95}
MOCK_HAM  = {"label": "ham",  "is_spam": False, "spam_probability": 0.03, "ham_probability": 0.97, "confidence": 0.97}


@pytest.fixture
def client():
    with patch("src.model.load_model") as ml, \
         patch("src.model.predict_single", return_value=MOCK_SPAM), \
         patch("src.model.predict", return_value=[MOCK_SPAM, MOCK_HAM]), \
         patch("src.feedback.save_feedback", return_value=1), \
         patch("src.feedback.get_feedback_stats", return_value={"total_feedback": 0}), \
         patch("src.feedback.log_prediction"), \
         patch("src.database.init_db"):
        ml.return_value = MagicMock()
        from app import app
        yield TestClient(app)


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert "status" in d
        assert "model_loaded" in d

    def test_root_html(self, client):
        r = client.get("/")
        assert r.status_code == 200


class TestPredict:
    def test_predict_success(self, client):
        r = client.post("/predict", json={"text": "Win free money now!"})
        assert r.status_code == 200
        d = r.json()
        assert d["success"] is True
        p = d["prediction"]
        assert p["label"] in ("spam", "ham")
        assert 0 <= p["spam_probability"] <= 1
        assert "processing_time_ms" in d

    def test_predict_empty_rejected(self, client):
        r = client.post("/predict", json={"text": ""})
        assert r.status_code == 422

    def test_predict_too_long_rejected(self, client):
        r = client.post("/predict", json={"text": "a" * 10001})
        assert r.status_code == 422


class TestBatch:
    def test_batch_success(self, client):
        r = client.post("/predict/batch", json={"texts": ["spam text", "normal text"]})
        assert r.status_code == 200
        d = r.json()
        assert d["count"] == 2

    def test_batch_empty_rejected(self, client):
        r = client.post("/predict/batch", json={"texts": []})
        assert r.status_code == 422


class TestFeedback:
    def test_feedback_correct(self, client):
        r = client.post("/feedback", json={"text": "hello", "predicted": "ham", "correct": True})
        assert r.status_code == 200

    def test_feedback_incorrect_needs_true_label(self, client):
        r = client.post("/feedback", json={"text": "hello", "predicted": "ham", "correct": False})
        assert r.status_code == 422

    def test_feedback_incorrect_with_label(self, client):
        r = client.post("/feedback", json={"text": "hello", "predicted": "ham", "correct": False, "true_label": "spam"})
        assert r.status_code == 200

    def test_feedback_stats(self, client):
        r = client.get("/feedback/stats")
        assert r.status_code == 200


class TestExamples:
    def test_examples(self, client):
        r = client.get("/examples")
        assert r.status_code == 200
        d = r.json()
        assert len(d["spam_examples"]) > 0
        assert len(d["ham_examples"]) > 0
