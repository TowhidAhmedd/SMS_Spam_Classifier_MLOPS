"""
Feedback loop + auto-retraining pipeline.
Uses PostgreSQL for persistent storage on Render.
"""
import os
import json
import logging
import threading
from typing import Optional

from src.config import get_settings
from src.database import (
    get_db, save_feedback_db, get_wrong_count,
    get_feedback_stats_db, save_retrain_log_db,
    get_wrong_samples, log_prediction_db
)
from src.metrics import track_retrain

logger = logging.getLogger(__name__)
settings = get_settings()


def save_feedback(
    text: str,
    predicted: str,
    correct: bool,
    true_label: Optional[str],
    confidence: Optional[float] = None,
) -> int:
    with get_db() as db:
        total = save_feedback_db(db, text, predicted, correct, true_label, confidence)
        wrong = get_wrong_count(db)

    logger.info(f"Feedback saved | total={total} | wrong={wrong}")

    if wrong >= settings.retrain_threshold:
        logger.info(f"Retrain threshold reached ({wrong}). Triggering background retrain...")
        thread = threading.Thread(target=retrain_from_feedback, daemon=True)
        thread.start()

    return total


def get_feedback_stats() -> dict:
    with get_db() as db:
        return get_feedback_stats_db(db)


def log_prediction(
    text_preview: str,
    predicted_label: str,
    spam_probability: float,
    confidence: float,
    latency_ms: float,
):
    """Log every prediction to PostgreSQL for monitoring."""
    try:
        with get_db() as db:
            log_prediction_db(db, text_preview, predicted_label, spam_probability, confidence, latency_ms)
    except Exception as e:
        logger.warning(f"Could not log prediction: {e}")


def retrain_from_feedback():
    """
    Retrain model using original data + user-corrected samples.
    Runs in background thread — does not block the API.
    """
    track_retrain()
    try:
        import pandas as pd
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split

        logger.info("Retraining started...")

        # Load original dataset
        original = pd.read_csv("data/spam.csv", encoding="latin-1")
        if "v1" in original.columns:
            original = original[["v1", "v2"]]
            original.columns = ["label", "text"]
        else:
            original = original[["Category", "Message"]]
            original.columns = ["label", "text"]
        original["label_num"] = original["label"].map({"ham": 0, "spam": 1})

        # Load corrected samples from PostgreSQL
        with get_db() as db:
            wrong_samples = get_wrong_samples(db)
            wrong_count   = len(wrong_samples)

        if wrong_samples:
            feedback_rows = [
                {"text": s.text, "label_num": 0 if s.true_label == "ham" else 1}
                for s in wrong_samples
            ]
            feedback_df = pd.DataFrame(feedback_rows)
            # Weight feedback samples 3x
            combined = pd.concat([
                original[["text", "label_num"]],
                pd.concat([feedback_df] * 3)
            ], ignore_index=True)
        else:
            combined = original[["text", "label_num"]]

        combined.dropna(inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(
            combined["text"], combined["label_num"],
            test_size=0.2, random_state=42, stratify=combined["label_num"]
        )

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
        pipeline.fit(X_train, y_train)
        new_f1 = round(f1_score(y_test, pipeline.predict(X_test)), 4)

        # Load current best F1
        old_f1 = 0.0
        if os.path.exists("models/metadata.json"):
            with open("models/metadata.json") as f:
                old_f1 = json.load(f).get("best_f1", 0.0)

        if new_f1 >= old_f1 - 0.01:
            joblib.dump(pipeline, settings.model_path)
            from src.model import load_model
            load_model.cache_clear()
            result = f"success | F1: {old_f1} → {new_f1}"
            logger.info(f"Retrain complete: {result}")
        else:
            result = f"skipped | new model worse: {new_f1} < {old_f1}"
            logger.warning(result)

        # Save retrain log to PostgreSQL
        with get_db() as db:
            save_retrain_log_db(db, wrong_count, old_f1, new_f1, result)

    except Exception as e:
        logger.exception(f"Retraining failed: {e}")
