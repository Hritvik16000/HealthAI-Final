from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "patients_clean.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "tabular"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

target = "risk_label"
X = df.drop(columns=[target, "length_of_stay"], errors="ignore")
y = df[target]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds, average="weighted"))

joblib.dump(model, ARTIFACT_DIR / "risk_classifier.pkl")
joblib.dump(label_encoder, ARTIFACT_DIR / "risk_label_encoder.pkl")
print("[OK] Classifier saved")
