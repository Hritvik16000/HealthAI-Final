from pathlib import Path
import math
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "patients_clean.csv"
ART_DIR = BASE_DIR / "artifacts" / "tabular"
REPORTS = BASE_DIR / "reports"
ART_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["length_of_stay", "risk_label"], errors="ignore")
y = df["length_of_stay"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = math.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

joblib.dump(model, ART_DIR / "mlp_regressor.pkl")

out = pd.DataFrame([{
    "module": "mlp_regressor",
    "metric_1_name": "mae",
    "metric_1_value": round(float(mae), 4),
    "metric_2_name": "rmse",
    "metric_2_value": round(float(rmse), 4),
    "metric_3_name": "r2",
    "metric_3_value": round(float(r2), 4)
}])

path = REPORTS / "deep_learning_metrics.csv"
if path.exists():
    old = pd.read_csv(path)
    old = old[old["module"] != "mlp_regressor"]
    out = pd.concat([old, out], ignore_index=True)
out.to_csv(path, index=False)

print("[OK] Saved mlp_regressor.pkl")
print("MAE:", round(float(mae), 4))
print("RMSE:", round(float(rmse), 4))
print("R2:", round(float(r2), 4))
