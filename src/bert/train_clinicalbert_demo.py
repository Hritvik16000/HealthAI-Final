from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "reviews_clean.csv"
OUT_DIR = BASE_DIR / "artifacts" / "bert"
REPORTS = BASE_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna(subset=["review", "sentiment"]).copy()
df["review"] = df["review"].astype(str)
df["sentiment"] = df["sentiment"].astype(str)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model.eval()

def embed_texts(texts, batch_size=16):
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = model(**enc).last_hidden_state
            emb = out.mean(dim=1).cpu().numpy()
            all_emb.append(emb)
    return np.vstack(all_emb)

X = embed_texts(df["review"].tolist())
y = df["sentiment"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="weighted")

joblib.dump(clf, OUT_DIR / "clinicalbert_logreg.pkl")
pd.DataFrame([{"module": "clinicalbert_demo", "accuracy": round(float(acc), 4), "f1_weighted": round(float(f1), 4)}]).to_csv(
    REPORTS / "clinicalbert_metrics.csv", index=False
)

print("[OK] Saved clinicalbert_logreg.pkl")
print("Accuracy:", round(float(acc), 4))
print("F1:", round(float(f1), 4))
