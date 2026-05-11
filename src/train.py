"""
Training pipeline with MLflow experiment tracking.
Supports both local mlruns/ and remote DagsHub tracking.
Run: python src/train.py
"""
import json
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import get_settings

load_dotenv()



logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

DATA_PATH    = "data/spam.csv"
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "spam_classifier.joblib")
META_PATH    = os.path.join(MODEL_DIR, "metadata.json")


def setup_mlflow():
    """Configure MLflow tracking — local or DagsHub."""
    if settings.mlflow_tracking_username and settings.mlflow_tracking_password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
        logger.info(f"MLflow → DagsHub: {settings.mlflow_tracking_uri}")
    else:
        logger.info("MLflow → local mlruns/")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")

    # Auto-detect columns
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "text"]
    elif "Category" in df.columns and "Message" in df.columns:
        df = df[["Category", "Message"]]
        df.columns = ["label", "text"]
    else:
        raise ValueError(f"Unknown CSV format. Columns found: {df.columns.tolist()}")

    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    spam_count = df["label_num"].sum()
    ham_count  = (df["label_num"] == 0).sum()
    logger.info(f"Dataset: {len(df)} samples | spam={spam_count} | ham={ham_count}")
    return df


def build_pipeline(model_type: str) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        stop_words="english",
    )
    clf = (
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        if model_type == "lr"
        else MultinomialNB(alpha=0.1)
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
    }
    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    setup_mlflow()

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_num"],
        test_size=0.2, random_state=42, stratify=df["label_num"]
    )

    best_f1       = 0
    best_pipeline = None
    best_metrics  = {}
    best_run_id   = None

    experiments = [
        ("lr", "LogisticRegression"),
        ("nb", "NaiveBayes"),
    ]

    for model_type, run_name in experiments:
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Training {run_name}...")

            mlflow.log_params({
                "model_type":         model_type,
                "train_size":         len(X_train),
                "test_size":          len(X_test),
                "tfidf_max_features": 10000,
                "tfidf_ngram_range":  "(1,2)",
            })

            pipeline = build_pipeline(model_type)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = (
                pipeline.predict_proba(X_test)[:, 1]
                if hasattr(pipeline.named_steps["clf"], "predict_proba")
                else None
            )

            metrics = compute_metrics(y_test, y_pred, y_prob)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(
                {"confusion_matrix": confusion_matrix(y_test, y_pred).tolist()},
                "confusion_matrix.json"
            )
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name="spam-classifier-prod",
                input_example=["Free entry to win $1000 now!!!"],
            )

            logger.info(f"\n{classification_report(y_test, y_pred, target_names=['ham','spam'])}")

            if metrics["f1"] > best_f1:
                best_f1       = metrics["f1"]
                best_pipeline = pipeline
                best_metrics  = metrics
                best_run_id   = run.info.run_id

    # Save best model
    joblib.dump(best_pipeline, MODEL_PATH)
    metadata = {
        "best_run_id": best_run_id,
        "best_f1":     best_f1,
        **best_metrics
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Best model saved → {MODEL_PATH} | F1={best_f1:.4f}")
    return best_pipeline, best_metrics


if __name__ == "__main__":
    train()
