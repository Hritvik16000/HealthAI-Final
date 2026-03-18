from pathlib import Path
import math
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

patients = pd.read_csv(BASE_DIR / "data" / "processed" / "patients_clean.csv")
reviews = pd.read_csv(BASE_DIR / "data" / "processed" / "reviews_clean.csv")

# Classification
clf_X = patients.drop(columns=["risk_label", "length_of_stay"], errors="ignore")
clf_y = patients["risk_label"]
le = LabelEncoder()
clf_y_enc = le.fit_transform(clf_y)
X_train, X_test, y_train, y_test = train_test_split(clf_X, clf_y_enc, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)

# Regression
reg_X = patients.drop(columns=["length_of_stay", "risk_label"], errors="ignore")
reg_y = patients["length_of_stay"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(reg_X, reg_y, test_size=0.2, random_state=42)
reg = RandomForestRegressor(random_state=42)
reg.fit(Xr_train, yr_train)
reg_pred = reg.predict(Xr_test)

# Clustering
clus_X = patients.drop(columns=["risk_label", "length_of_stay"], errors="ignore")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(clus_X)

# Sentiment
txt_X = reviews["review"].astype(str)
txt_y = reviews["sentiment"].astype(str)
Xt_train, Xt_test, yt_train, yt_test = train_test_split(txt_X, txt_y, test_size=0.2, random_state=42)
vec = TfidfVectorizer(stop_words="english")
Xt_train_vec = vec.fit_transform(Xt_train)
Xt_test_vec = vec.transform(Xt_test)
sent_model = LogisticRegression(max_iter=1000)
sent_model.fit(Xt_train_vec, yt_train)
sent_pred = sent_model.predict(Xt_test_vec)
prec, rec, f1, _ = precision_recall_fscore_support(yt_test, sent_pred, average="weighted", zero_division=0)

metrics = {
    "classification_accuracy": round(float(accuracy_score(y_test, clf_pred)), 4),
    "classification_f1_weighted": round(float(f1_score(y_test, clf_pred, average="weighted")), 4),
    "regression_mae": round(float(mean_absolute_error(yr_test, reg_pred)), 4),
    "regression_rmse": round(float(math.sqrt(mean_squared_error(yr_test, reg_pred))), 4),
    "regression_r2": round(float(r2_score(yr_test, reg_pred)), 4),
    "clustering_silhouette": round(float(silhouette_score(clus_X, clusters)), 4),
    "sentiment_precision_weighted": round(float(prec), 4),
    "sentiment_recall_weighted": round(float(rec), 4),
    "sentiment_f1_weighted": round(float(f1), 4),
}

pd.DataFrame([metrics]).to_csv(REPORT_DIR / "metrics_summary.csv", index=False)

with open(REPORT_DIR / "model_card.md", "w") as f:
    f.write("# HealthAI Model Card\n\n")
    f.write("## Scope\n")
    f.write("This project provides a modular healthcare analytics demo including classification, regression, clustering, association rules, and sentiment analysis.\n\n")
    f.write("## Data\n")
    f.write("Synthetic demo datasets were used for tabular patient data and patient feedback text.\n\n")
    f.write("## Models\n")
    f.write("- RandomForestClassifier for risk prediction\n")
    f.write("- RandomForestRegressor for length of stay prediction\n")
    f.write("- KMeans for patient clustering\n")
    f.write("- Apriori association rules for comorbidity pattern discovery\n")
    f.write("- TF-IDF + Logistic Regression for sentiment analysis\n\n")
    f.write("## Ethics\n")
    f.write("No real patient PII is used. This is a learning/demo system and must not be used for clinical decision-making.\n\n")
    f.write("## Metrics\n")
    for k, v in metrics.items():
        f.write(f"- {k}: {v}\n")

print("[OK] Reports generated")
