"""
PostgreSQL database setup using SQLAlchemy.
Handles connection pooling, table creation, and session management.
"""
import logging
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Text, Boolean,
    DateTime, String, Float, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

from src.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,       # auto-reconnect if connection drops
    pool_recycle=300,         # recycle connections every 5 min
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class Feedback(Base):
    __tablename__ = "feedback"

    id          = Column(Integer, primary_key=True, index=True)
    text        = Column(Text, nullable=False)
    predicted   = Column(String(10), nullable=False)   # spam / ham
    correct     = Column(Boolean, nullable=False)
    true_label  = Column(String(10), nullable=True)    # user correction
    confidence  = Column(Float, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)


class RetrainLog(Base):
    __tablename__ = "retrain_log"

    id             = Column(Integer, primary_key=True, index=True)
    timestamp      = Column(DateTime, default=datetime.utcnow, nullable=False)
    reason         = Column(String(100))
    feedback_count = Column(Integer)
    old_f1         = Column(Float, nullable=True)
    new_f1         = Column(Float, nullable=True)
    result         = Column(Text)


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id               = Column(Integer, primary_key=True, index=True)
    text_preview     = Column(Text)
    predicted_label  = Column(String(10))
    spam_probability = Column(Float)
    confidence       = Column(Float)
    latency_ms       = Column(Float)
    created_at       = Column(DateTime, default=datetime.utcnow, nullable=False)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("PostgreSQL tables created/verified successfully.")
    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {e}")
        raise


# ── Session ───────────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    """Context manager for database sessions."""
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Queries ───────────────────────────────────────────────────────────────────

def save_feedback_db(
    db: Session,
    text: str,
    predicted: str,
    correct: bool,
    true_label: str | None,
    confidence: float | None
) -> int:
    feedback = Feedback(
        text=text,
        predicted=predicted,
        correct=correct,
        true_label=true_label,
        confidence=confidence,
    )
    db.add(feedback)
    db.flush()
    total = db.query(func.count(Feedback.id)).scalar()
    return total


def get_wrong_count(db: Session) -> int:
    return db.query(func.count(Feedback.id)).filter(Feedback.correct == False).scalar()


def get_feedback_stats_db(db: Session) -> dict:
    total   = db.query(func.count(Feedback.id)).scalar() or 0
    correct = db.query(func.count(Feedback.id)).filter(Feedback.correct == True).scalar() or 0
    wrong   = db.query(func.count(Feedback.id)).filter(Feedback.correct == False).scalar() or 0
    last    = db.query(RetrainLog).order_by(RetrainLog.id.desc()).first()

    return {
        "database": "PostgreSQL",
        "total_feedback": total,
        "correct_predictions": correct,
        "wrong_predictions": wrong,
        "accuracy_from_feedback": round(correct / total, 4) if total > 0 else None,
        "retrain_threshold": settings.retrain_threshold,
        "retraining_in": max(0, settings.retrain_threshold - wrong),
        "last_retrain": str(last.timestamp) if last else None,
        "last_retrain_result": last.result if last else None,
    }


def log_prediction_db(
    db: Session,
    text_preview: str,
    predicted_label: str,
    spam_probability: float,
    confidence: float,
    latency_ms: float,
):
    log = PredictionLog(
        text_preview=text_preview,
        predicted_label=predicted_label,
        spam_probability=spam_probability,
        confidence=confidence,
        latency_ms=latency_ms,
    )
    db.add(log)


def save_retrain_log_db(
    db: Session,
    feedback_count: int,
    old_f1: float,
    new_f1: float,
    result: str,
):
    log = RetrainLog(
        reason="threshold_reached",
        feedback_count=feedback_count,
        old_f1=old_f1,
        new_f1=new_f1,
        result=result,
    )
    db.add(log)


def get_wrong_samples(db: Session):
    """Return wrong predictions with user corrections for retraining."""
    return db.query(Feedback).filter(
        Feedback.correct == False,
        Feedback.true_label.isnot(None)
    ).all()
