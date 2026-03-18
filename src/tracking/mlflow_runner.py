from pathlib import Path
import math
import mlflow
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "patients_clean.csv"
ART_DIR = BASE_DIR / "artifacts" / "tabular"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("HealthAI")

df = pd.read_csv(DATA_PATH)

with mlflow.start_run(run_name="tabular_baselines"):
    # classifier
    clf = joblib.load(ART_DIR / "risk_classifier.pkl")
    Xc = df.drop(columns=["risk_label", "length_of_stay"], errors="ignore")
    yc = df["risk_label"]
    le = LabelEncoder()
    yc_enc = le.fit_transform(yc)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc_enc, test_size=0.2, random_state=42)
    clf_pred = clf.predict(Xc_test)

    mlflow.log_metric("classifier_accuracy", float(accuracy_score(yc_test, clf_pred)))
    mlflow.log_metric("classifier_f1_weighted", float(f1_score(yc_test, clf_pred, average="weighted")))

    # regressor
    reg = joblib.load(ART_DIR / "los_regressor.pkl")
    Xr = df.drop(columns=["length_of_stay", "risk_label"], errors="ignore")
    yr = df["length_of_stay"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg_pred = reg.predict(Xr_test)

    mlflow.log_metric("regression_mae", float(mean_absolute_error(yr_test, reg_pred)))
    mlflow.log_metric("regression_rmse", float(math.sqrt(mean_squared_error(yr_test, reg_pred))))
    mlflow.log_metric("regression_r2", float(r2_score(yr_test, reg_pred)))

    mlflow.log_artifact(str(ART_DIR / "risk_classifier.pkl"))
    mlflow.log_artifact(str(ART_DIR / "los_regressor.pkl"))

print("[OK] Logged run to MLflow")
