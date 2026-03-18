# HealthAI Suite — Intelligent Analytics for Patient Care

## Overview
HealthAI Suite is an end-to-end healthcare analytics project built to demonstrate:
- disease risk classification
- length of stay regression
- patient clustering
- association rule mining
- sentiment analysis on patient feedback
- REST API deployment with FastAPI
- interactive dashboard with Streamlit

## Project Structure
- `data/raw/` → input datasets
- `data/processed/` → cleaned datasets
- `artifacts/` → trained model files and association outputs
- `src/data/` → preprocessing scripts
- `src/models/` → model training scripts
- `src/api/` → FastAPI backend
- `src/dashboard/` → Streamlit frontend
- `reports/` → metrics summary and model card

## Features
### 1. Risk Classification
Predicts disease risk category from patient vitals and history.

### 2. Length of Stay Prediction
Forecasts expected hospitalization duration.

### 3. Patient Segmentation
Groups patients into clusters based on feature similarity.

### 4. Association Rule Mining
Finds interpretable health patterns among risk factors.

### 5. Sentiment Analysis
Classifies patient reviews as positive or negative.

### 6. API
FastAPI endpoints:
- `/health`
- `/predict/risk`
- `/predict/los`
- `/cluster/patient`
- `/predict/sentiment`
- `/association/rules`

### 7. Dashboard
Interactive Streamlit UI for model inference and output exploration.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.preprocess
python -m src.models.train_classifier
python -m src.models.train_regressor
python -m src.models.train_cluster
python -m src.models.train_association
python -m src.models.train_sentiment
python -m src.utils.generate_reports
uvicorn src.api.main:app --reload
streamlit run src/dashboard/app.py

## Step 22 — add a tiny smoke test

The brief mentions coding standards and unit tests. We do not need to build a cathedral here; a small smoke test is enough to avoid looking asleep at the wheel. :contentReference[oaicite:3]{index=3}

```bash
cat > tests/test_artifacts.py << 'EOF'
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

def test_artifacts_exist():
    required = [
        BASE_DIR / "artifacts" / "tabular" / "risk_classifier.pkl",
        BASE_DIR / "artifacts" / "tabular" / "los_regressor.pkl",
        BASE_DIR / "artifacts" / "cluster" / "kmeans.pkl",
        BASE_DIR / "artifacts" / "association" / "association_rules.csv",
        BASE_DIR / "artifacts" / "nlp" / "sentiment_model.pkl",
        BASE_DIR / "artifacts" / "nlp" / "tfidf_vectorizer.pkl",
    ]
    for path in required:
        assert path.exists(), f"Missing artifact: {path}"
