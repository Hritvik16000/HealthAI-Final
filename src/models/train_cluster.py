from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "patients_clean.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "cluster"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

feature_cols = [
    "age",
    "bmi",
    "blood_pressure",
    "glucose",
    "cholesterol",
    "heart_rate",
    "smoker",
    "diabetes_history",
]

X = df[feature_cols].copy()

# give more influence to clinically strong risk features
feature_weights = {
    "age": 1.2,
    "bmi": 1.1,
    "blood_pressure": 1.5,
    "glucose": 1.7,
    "cholesterol": 1.5,
    "heart_rate": 1.0,
    "smoker": 1.3,
    "diabetes_history": 1.4,
}

for col, weight in feature_weights.items():
    X[col] = X[col] * weight

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# manually-separated archetypes to prevent centroid collapse
# order of features:
# age, bmi, blood_pressure, glucose, cholesterol, heart_rate, smoker, diabetes_history
seed_profiles_raw = np.array([
    [28, 22, 112, 88, 170, 72, 0, 0],   # lower-risk stable
    [48, 28, 134, 118, 215, 84, 1, 0],  # moderate-risk general
    [68, 33, 160, 175, 275, 96, 1, 1],  # high-risk metabolic
], dtype=float)

for i, col in enumerate(feature_cols):
    seed_profiles_raw[:, i] = seed_profiles_raw[:, i] * feature_weights[col]

seed_profiles_scaled = scaler.transform(seed_profiles_raw)

kmeans = KMeans(
    n_clusters=3,
    init=seed_profiles_scaled,
    n_init=1,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", round(float(sil), 4))

df_out = df.copy()
df_out["cluster"] = clusters

profiles = df_out.groupby("cluster")[feature_cols].mean().round(2)

# map cluster ids to ordered risk meaning based on profile severity
severity = {}
for cluster_id, row in profiles.iterrows():
    score = (
        row["age"] * 0.03 +
        row["bmi"] * 0.04 +
        row["blood_pressure"] * 0.05 +
        row["glucose"] * 0.06 +
        row["cholesterol"] * 0.04 +
        row["heart_rate"] * 0.02 +
        row["smoker"] * 1.0 +
        row["diabetes_history"] * 1.2
    )
    severity[int(cluster_id)] = float(score)

ordered = [cid for cid, _ in sorted(severity.items(), key=lambda x: x[1])]
cluster_names = {
    ordered[0]: "Lower-Risk Stable",
    ordered[1]: "Moderate-Risk General",
    ordered[2]: "High-Risk Metabolic",
}

pipeline = Pipeline([
    ("scaler", scaler),
    ("kmeans", kmeans)
])

joblib.dump(pipeline, ARTIFACT_DIR / "kmeans_pipeline.pkl")
profiles.to_csv(ARTIFACT_DIR / "cluster_profiles.csv")

with open(ARTIFACT_DIR / "cluster_names.json", "w") as f:
    json.dump(cluster_names, f, indent=2)

df_out.to_csv(ARTIFACT_DIR / "clustered_patients_preview.csv", index=False)

print("[OK] Saved kmeans_pipeline.pkl")
print("[OK] Saved cluster_profiles.csv")
print("[OK] Saved cluster_names.json")
print("[OK] Saved clustered_patients_preview.csv")
print("\nCluster Profiles:\n")
print(profiles)
print("\nCluster Names:\n")
print(cluster_names)
