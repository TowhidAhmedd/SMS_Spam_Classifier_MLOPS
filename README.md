# 📧 End-to-End MLOps Spam Detection System

> Production-grade email spam classifier with complete MLOps pipeline deployed on Render.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.17-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Render](https://img.shields.io/badge/Deployed-Render-purple)

---

## 🏗 MLOps Architecture

```
Data Source (UCI SMS Spam Collection)
    ↓
Data Pipeline / ETL  (src/download_data.py)
    ↓
Feature Engineering  (TF-IDF, n-grams)
    ↓
Model Training       (Logistic Regression + NaiveBayes)
    ↓
Experiment Tracking  (MLflow + DagsHub)
    ↓
Model Registry       (MLflow Model Registry)
    ↓
API Service          (FastAPI)
    ↓
Docker + CI/CD       (Dockerfile + GitHub Actions)
    ↓
Render Deployment    (Docker + PostgreSQL)
    ↓
Monitoring           (Prometheus + Grafana + Sentry)
    ↓
Feedback Loop        (PostgreSQL) → Auto-Retraining
```

---

## 📁 Project Structure

```
Spam_prediction/
├── src/
│   ├── config.py          # Centralized settings (pydantic-settings)
│   ├── database.py        # PostgreSQL models + queries (SQLAlchemy)
│   ├── train.py           # Training pipeline + MLflow tracking
│   ├── model.py           # Inference (cached model loading)
│   ├── metrics.py         # Prometheus metrics
│   ├── feedback.py        # Feedback loop + auto-retraining
│   └── download_data.py   # Dataset downloader
├── static/
│   └── index.html         # Production Web UI
├── tests/
│   ├── conftest.py
│   └── test_api.py
├── models/                # Trained model artifacts
├── data/                  # Dataset (gitignored)
├── monitoring/
│   └── prometheus.yml     # Prometheus scrape config
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI/CD
├── app.py                 # FastAPI application
├── Dockerfile             # Multi-stage production build
├── docker-compose.yml     # Local full stack
├── render.yaml            # Render deployment blueprint
├── Makefile               # Developer shortcuts
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Local Setup

### 1. Clone and setup
```bash
git clone https://github.com/YOUR_USERNAME/Spam_prediction
cd Spam_prediction
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements-dev.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Download data and train
```bash
make data    # download dataset
make train   # train model + MLflow tracking
```

### 4. Run API only (SQLite-free, PostgreSQL needed)
```bash
# Option A: Docker Compose (recommended — includes PostgreSQL)
make docker-up
# API: http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)

# Option B: Manual (need local PostgreSQL running)
make run
```

### 5. Run tests
```bash
make test
```

---

## ☁️ Deploy to Render

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2: Deploy via Blueprint
1. Go to [render.com](https://render.com) → **New → Blueprint**
2. Connect your GitHub repo
3. Render reads `render.yaml` → creates PostgreSQL + Web Service automatically

### Step 3: Set environment variables in Render Dashboard
```
MLFLOW_TRACKING_URI      = https://dagshub.com/YOUR_USER/YOUR_REPO.mlflow
MLFLOW_TRACKING_USERNAME = your_dagshub_username
MLFLOW_TRACKING_PASSWORD = your_dagshub_PAT
SENTRY_DSN               = your_sentry_dsn
```

### Step 4: Commit trained model
```bash
# Remove model from .gitignore first, then:
git add models/spam_classifier.joblib models/metadata.json
git commit -m "Add trained model"
git push
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/model/info` | Training metrics |
| `POST` | `/predict` | Classify single email |
| `POST` | `/predict/batch` | Classify up to 50 emails |
| `POST` | `/feedback` | Submit prediction feedback |
| `GET` | `/feedback/stats` | Feedback statistics |
| `GET` | `/examples` | Sample emails |
| `GET` | `/docs` | Swagger UI |

### Example Request
```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won a free iPhone!"}'
```

### Example Response
```json
{
  "success": true,
  "prediction": {
    "label": "spam",
    "is_spam": true,
    "spam_probability": 0.9823,
    "ham_probability": 0.0177,
    "confidence": 0.9823
  },
  "text_preview": "Congratulations! You have won a free iPhone!",
  "processing_time_ms": 3.8
}
```

---

## 🗄 Database Schema (PostgreSQL)

| Table | Purpose |
|-------|---------|
| `feedback` | User feedback on predictions |
| `prediction_log` | Every prediction logged |
| `retrain_log` | Retraining history |

---

## 🔄 Feedback & Auto-Retraining

1. User classifies an email → result shown in UI
2. User clicks ✅ Correct or ❌ Wrong
3. Feedback saved to **PostgreSQL**
4. When **50 wrong predictions** accumulate → auto-retrain triggers in background
5. New model replaces old one only if F1 score improves
6. `GET /feedback/stats` shows progress

---

## 📊 Monitoring Stack

| Tool | What it monitors |
|------|-----------------|
| **Prometheus** | Request count, latency, prediction distribution |
| **Grafana** | Visual dashboards |
| **Sentry** | Errors and exceptions |
| **Render Dashboard** | CPU, memory, logs |
| **PostgreSQL** | Prediction logs, feedback history |

---

## 🧠 Model Details

| Component | Choice |
|-----------|--------|
| Vectorizer | TF-IDF (10k features, 1-2 ngrams, sublinear TF) |
| Classifier | Logistic Regression (class_weight=balanced) |
| Tracking | MLflow → DagsHub |
| Serialization | joblib |

**Typical Performance:**

| Metric | Score |
|--------|-------|
| Accuracy | ~98.5% |
| Precision | ~97% |
| Recall | ~95% |
| F1 | ~96% |
| ROC AUC | ~99% |

---

## 🛠 Tech Stack

`Python 3.11` · `FastAPI` · `scikit-learn` · `MLflow` · `DagsHub` · `PostgreSQL` · `SQLAlchemy` · `Prometheus` · `Grafana` · `Sentry` · `Docker` · `GitHub Actions` · `Render`
