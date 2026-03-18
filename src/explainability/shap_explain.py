from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

BASE_DIR = Path(__file__).resolve().parents[2]
ART_DIR = BASE_DIR / "artifacts" / "tabular"
OUT_DIR = BASE_DIR / "artifacts" / "explainability"
DATA_PATH = BASE_DIR / "data" / "processed" / "patients_clean.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load(ART_DIR / "risk_classifier.pkl")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["risk_label", "length_of_stay"], errors="ignore").head(100)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure()
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[0], X, show=False)
else:
    shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "risk_classifier_shap_summary.png", dpi=180, bbox_inches="tight")
print("[OK] Saved SHAP summary plot")
