"""
Model inference module.
Loads trained pipeline once and caches it in memory.
"""
import os
import logging
from functools import lru_cache
from typing import List

import joblib

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def load_model():
    path = settings.model_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run `python src/train.py` first."
        )
    logger.info(f"Loading model from {path}...")
    model = joblib.load(path)
    logger.info("Model loaded and cached.")
    return model


def predict(texts: List[str]) -> List[dict]:
    model = load_model()
    preds = model.predict(texts)
    probs = model.predict_proba(texts)

    return [
        {
            "label":             "spam" if pred == 1 else "ham",
            "is_spam":           bool(pred == 1),
            "spam_probability":  round(float(prob[1]), 4),
            "ham_probability":   round(float(prob[0]), 4),
            "confidence":        round(float(max(prob)), 4),
        }
        for pred, prob in zip(preds, probs)
    ]


def predict_single(text: str) -> dict:
    return predict([text])[0]
