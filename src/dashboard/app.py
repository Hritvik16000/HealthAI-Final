from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
import streamlit as st

from src.utils.sentiment_engine import HybridSentimentEngine
from src.rag.chatbot import HealthRAGChatbot
from src.translation.translator_utils import EnHiTranslator

TAB_DIR = BASE_DIR / "artifacts" / "tabular"
CLUSTER_DIR = BASE_DIR / "artifacts" / "cluster"
ASSOC_DIR = BASE_DIR / "artifacts" / "association"
DATA_DIR = BASE_DIR / "data" / "processed"

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

st.set_page_config(page_title="HealthAI Dashboard", layout="wide")
st.title("HealthAI Suite Dashboard")
st.caption("Full-scope demonstrator: analytics, NLP, chatbot, translation, and advanced modules")

tabs = st.tabs([
    "Risk Prediction",
    "Length of Stay",
    "Patient Clustering",
    "Association Rules",
    "Sentiment Analysis",
    "Live Chatbot",
    "Translation"
])

with st.sidebar:
    st.header("Patient Input")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5)
    blood_pressure = st.number_input("Blood Pressure", min_value=50.0, max_value=250.0, value=130.0)
    glucose = st.number_input("Glucose", min_value=40.0, max_value=400.0, value=110.0)
    cholesterol = st.number_input("Cholesterol", min_value=50.0, max_value=500.0, value=200.0)
    heart_rate = st.number_input("Heart Rate", min_value=30.0, max_value=220.0, value=80.0)
    smoker = st.selectbox("Smoker", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    diabetes_history = st.selectbox("Diabetes History", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

patient_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "blood_pressure": blood_pressure,
    "glucose": glucose,
    "cholesterol": cholesterol,
    "heart_rate": heart_rate,
    "smoker": smoker,
    "diabetes_history": diabetes_history
}])

cluster_df = patient_df[["age","bmi","blood_pressure","glucose","cholesterol","heart_rate","smoker","diabetes_history"]]

with tabs[0]:
    if st.button("Predict Risk"):
        pred = risk_model.predict(patient_df)[0]
        label = risk_encoder.inverse_transform([pred])[0]
        st.success(f"Predicted Risk Category: {label}")

with tabs[1]:
    if st.button("Predict Length of Stay"):
        pred = los_model.predict(patient_df)[0]
        st.info(f"Predicted Length of Stay: {pred:.2f} days")

with tabs[2]:
    if st.button("Assign Cluster"):
        cluster = int(cluster_model.predict(cluster_df)[0])
        cluster_name = cluster_names.get(str(cluster), cluster_names.get(cluster, f"Cluster {cluster}"))
        st.warning(f"Assigned Cluster ID: {cluster}")
        st.success(f"Cluster Meaning: {cluster_name}")

with tabs[3]:
    rules_path = ASSOC_DIR / "association_rules.csv"
    if rules_path.exists():
        st.dataframe(pd.read_csv(rules_path), use_container_width=True)

with tabs[4]:
    review = st.text_area("Enter patient review", "The nursing staff were caring and attentive.")
    if st.button("Analyze Sentiment"):
        result = sentiment_engine.predict(review)
        st.success(f"Predicted Sentiment: {result['label']}")
        st.json(result)

with tabs[5]:
    st.subheader("Real-Time Healthcare Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask a healthcare question...")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chatbot.ask(user_prompt, history=st.session_state.chat_history)
                answer = result["answer"]
                st.markdown(answer)

                with st.expander("Retrieved Context"):
                    for i, ctx in enumerate(result["contexts"], 1):
                        st.markdown(f"**Context {i}:** {ctx}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

with tabs[6]:
    text = st.text_input("English to Hindi", "Please take your medicines on time.")
    if st.button("Translate"):
        out = translator_en_hi.translate(text)
        st.success(out)

st.markdown("---")
patients_path = DATA_DIR / "patients_clean.csv"
if patients_path.exists():
    st.dataframe(pd.read_csv(patients_path).head(), use_container_width=True)
