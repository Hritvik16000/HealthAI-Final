from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TABULAR_FILE = RAW_DIR / "patients.csv"
TEXT_FILE = RAW_DIR / "reviews.csv"

def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Use stratified split only when valid.
    Falls back to normal split for very small datasets.
    """
    n_classes = y.nunique()
    n_samples = len(y)
    n_test = max(1, int(round(n_samples * test_size)))
    min_class_count = y.value_counts().min()

    can_stratify = (
        n_classes > 1 and
        n_test >= n_classes and
        min_class_count >= 2
    )

    if can_stratify:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        print("[INFO] Dataset too small for stratified split. Using normal split instead.")
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

def preprocess_tabular():
    if not TABULAR_FILE.exists():
        print(f"[WARN] Missing {TABULAR_FILE}")
        return

    df = pd.read_csv(TABULAR_FILE)
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            mode_vals = df[col].mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
        else:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        if col != "risk_label":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df.to_csv(PROCESSED_DIR / "patients_clean.csv", index=False)

    if "risk_label" in df.columns:
        X = df.drop(columns=["risk_label"])
        y = df["risk_label"]

        X_train, X_test, y_train, y_test = safe_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
        X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
        y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
        y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("[OK] Tabular preprocessing complete")

def preprocess_text():
    if not TEXT_FILE.exists():
        print(f"[WARN] Missing {TEXT_FILE}")
        return

    df = pd.read_csv(TEXT_FILE)
    df = df.drop_duplicates()

    if "review" in df.columns:
        df["review"] = df["review"].fillna("").astype(str).str.strip()

    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].fillna("neutral").astype(str)

    df.to_csv(PROCESSED_DIR / "reviews_clean.csv", index=False)
    print("[OK] Text preprocessing complete")

if __name__ == "__main__":
    preprocess_tabular()
    preprocess_text()
