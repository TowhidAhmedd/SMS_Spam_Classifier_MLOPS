"""
Centralized configuration — reads from .env file.
All settings in one place for easy management.
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── App ─────────────────────────────────────────────
    app_name: str = "Email Spam Classifier"
    app_version: str = "1.0.0"
    debug: bool = False
    port: int = 8000

    # ── Model ───────────────────────────────────────────
    model_path: str = "models/spam_classifier.joblib"
    retrain_threshold: int = 50

    # ── Database ────────────────────────────────────────
    database_url: str = Field(...)

    # ── MLflow ──────────────────────────────────────────
    mlflow_tracking_uri: str = Field(...)
    mlflow_tracking_username: Optional[str] = None
    mlflow_tracking_password: Optional[str] = None
    mlflow_experiment_name: str = "spam-classifier"

    # ── Sentry ──────────────────────────────────────────
    sentry_dsn: Optional[str] = None

    # ── Pydantic v2 config ──────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


# """
# Centralized configuration — reads from .env file.
# All settings in one place for easy management.
# """
# from functools import lru_cache

# from pydantic import Field
# from pydantic_settings import BaseSettings


# class Settings(BaseSettings):
#     # ── App ───────────────────────────────────────────────────────
#     app_name: str = "Email Spam Classifier"
#     app_version: str = "1.0.0"
#     debug: bool = False
#     port: int = 8000

#     # ── Model ─────────────────────────────────────────────────────
#     model_path: str = "models/spam_classifier.joblib"
#     retrain_threshold: int = 50

#     # ── Database (PostgreSQL) ─────────────────────────────────────
#     database_url: str = Field(
#         default="postgresql://postgres:postgres@localhost:5432/spam_db",
#         description="PostgreSQL connection string"
#     )

#     # ── MLflow / DagsHub ──────────────────────────────────────────
#     mlflow_tracking_uri: str = Field(
#         default="mlruns",
#         description="MLflow tracking URI (local or DagsHub)"
#     )
#     mlflow_tracking_username: str = ""
#     mlflow_tracking_password: str = ""
#     mlflow_experiment_name: str = "spam-classifier"

#     # ── Sentry ────────────────────────────────────────────────────
#     sentry_dsn: str = ""

#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"
#         case_sensitive = False


# @lru_cache(maxsize=1)
# def get_settings() -> Settings:
#     return Settings()
