from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "patients_clean.csv"
ART_DIR = BASE_DIR / "artifacts" / "tabular"
REPORTS = BASE_DIR / "reports"
ART_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["risk_label", "length_of_stay"], errors="ignore")
y = df["risk_label"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="weighted")

joblib.dump(model, ART_DIR / "mlp_classifier.pkl")
joblib.dump(le, ART_DIR / "mlp_classifier_label_encoder.pkl")

out = pd.DataFrame([{
    "module": "mlp_classifier",
    "metric_1_name": "accuracy",
    "metric_1_value": round(float(acc), 4),
    "metric_2_name": "f1_weighted",
    "metric_2_value": round(float(f1), 4)
}])

path = REPORTS / "deep_learning_metrics.csv"
if path.exists():
    old = pd.read_csv(path)
    old = old[old["module"] != "mlp_classifier"]
    out = pd.concat([old, out], ignore_index=True)
out.to_csv(path, index=False)

print("[OK] Saved mlp_classifier.pkl")
print("Accuracy:", round(float(acc), 4))
print("F1:", round(float(f1), 4))
