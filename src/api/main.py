from pathlib import Path
from typing import Optional, List, Dict
import json
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.utils.sentiment_engine import HybridSentimentEngine
from src.rag.chatbot import HealthRAGChatbot
from src.translation.translator_utils import EnHiTranslator

TAB_DIR = BASE_DIR / "artifacts" / "tabular"
CLUSTER_DIR = BASE_DIR / "artifacts" / "cluster"
ASSOC_DIR = BASE_DIR / "artifacts" / "association"

risk_model = joblib.load(TAB_DIR / "risk_classifier.pkl")
risk_encoder = joblib.load(TAB_DIR / "risk_label_encoder.pkl")
los_model = joblib.load(TAB_DIR / "los_regressor.pkl")
cluster_model = joblib.load(CLUSTER_DIR / "kmeans_pipeline.pkl")
sentiment_engine = HybridSentimentEngine()
chatbot = HealthRAGChatbot()
translator_en_hi = EnHiTranslator()

cluster_names = {}
cluster_names_path = CLUSTER_DIR / "cluster_names.json"
if cluster_names_path.exists():
    with open(cluster_names_path, "r") as f:
        cluster_names = json.load(f)

app = FastAPI(title="HealthAI API", version="3.3.0")

class PatientInput(BaseModel):
    age: int
    gender: int
    bmi: float
    blood_pressure: float
    glucose: float
    cholesterol: float
    heart_rate: float
    smoker: int
    diabetes_history: int

class TextInput(BaseModel):
    review: str

class ChatInput(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = None

class TranslateInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "HealthAI API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/risk")
def predict_risk(data: PatientInput):
    df = pd.DataFrame([data.model_dump()])
    pred = risk_model.predict(df)[0]
    label = risk_encoder.inverse_transform([pred])[0]
    return {"risk_prediction": str(label)}

@app.post("/predict/los")
def predict_los(data: PatientInput):
    df = pd.DataFrame([data.model_dump()])
    pred = los_model.predict(df)[0]
    return {"predicted_length_of_stay": round(float(pred), 2)}

@app.post("/cluster/patient")
def cluster_patient(data: PatientInput):
    df = pd.DataFrame([data.model_dump()])
    cluster_df = df[["age","bmi","blood_pressure","glucose","cholesterol","heart_rate","smoker","diabetes_history"]]
    cluster = int(cluster_model.predict(cluster_df)[0])
    cluster_name = cluster_names.get(str(cluster), cluster_names.get(cluster, f"Cluster {cluster}"))
    return {"cluster_id": cluster, "cluster_name": cluster_name}

@app.post("/predict/sentiment")
def predict_sentiment(data: TextInput):
    return sentiment_engine.predict(data.review)

@app.get("/association/rules")
def association_rules(limit: Optional[int] = 10):
    csv_path = ASSOC_DIR / "association_rules.csv"
    df = pd.read_csv(csv_path).head(limit)
    return {"rules": df.to_dict(orient="records")}

@app.post("/chat/ask")
def ask_chatbot(data: ChatInput):
    return chatbot.ask(data.question, history=data.history)

@app.post("/translate/en-hi")
def translate_en_hi(data: TranslateInput):
    return {"translated_text": translator_en_hi.translate(data.text)}
