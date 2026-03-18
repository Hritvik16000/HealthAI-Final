from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "patients_clean.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "association"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

bins_df = pd.DataFrame({
    "high_bp": df["blood_pressure"] > 135,
    "high_glucose": df["glucose"] > 120,
    "high_cholesterol": df["cholesterol"] > 220,
    "smoker_yes": df["smoker"] == 1 if df["smoker"].dtype != "object" else df["smoker"].astype(str).str.lower().eq("yes"),
    "diabetes_history_yes": df["diabetes_history"] == 1 if df["diabetes_history"].dtype != "object" else df["diabetes_history"].astype(str).str.lower().eq("yes"),
})

freq = apriori(bins_df, min_support=0.2, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.5)

rules.to_csv(ARTIFACT_DIR / "association_rules.csv", index=False)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
print("[OK] Association rules saved")
