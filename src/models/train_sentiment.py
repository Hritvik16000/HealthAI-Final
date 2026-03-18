from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "reviews_clean.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "nlp"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# keep only valid rows
df = df.dropna(subset=["review", "sentiment"]).copy()
df["review"] = df["review"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

X = df["review"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_vec, y)
preds = model.predict(X_vec)

print(classification_report(y, preds, zero_division=0))

joblib.dump(model, ARTIFACT_DIR / "sentiment_model.pkl")
joblib.dump(vectorizer, ARTIFACT_DIR / "tfidf_vectorizer.pkl")
print("[OK] Sentiment ML model saved")
