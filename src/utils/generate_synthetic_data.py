from pathlib import Path
import random
import pandas as pd

random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS_PATH = RAW_DIR / "patients.csv"
REVIEWS_PATH = RAW_DIR / "reviews.csv"

def generate_patients(n=250):
    rows = []

    for _ in range(n):
        profile_type = random.choices(
            ["low", "medium", "high"],
            weights=[0.35, 0.35, 0.30],
            k=1
        )[0]

        if profile_type == "low":
            age = random.randint(20, 45)
            gender = random.choice(["M", "F"])
            bmi = round(random.uniform(18.0, 25.5), 1)
            bp = random.randint(105, 128)
            glucose = random.randint(75, 108)
            cholesterol = random.randint(150, 205)
            heart_rate = random.randint(65, 82)
            smoker = random.choice(["No", "No", "No", "Yes"])
            diabetes = random.choice(["No", "No", "No", "Yes"])
            risk = "Low"
            los = random.randint(1, 4)

        elif profile_type == "medium":
            age = random.randint(35, 60)
            gender = random.choice(["M", "F"])
            bmi = round(random.uniform(24.5, 30.5), 1)
            bp = random.randint(125, 142)
            glucose = random.randint(100, 132)
            cholesterol = random.randint(190, 230)
            heart_rate = random.randint(74, 90)
            smoker = random.choice(["No", "Yes"])
            diabetes = random.choice(["No", "Yes"])
            risk = "Medium"
            los = random.randint(3, 7)

        else:
            age = random.randint(50, 80)
            gender = random.choice(["M", "F"])
            bmi = round(random.uniform(28.0, 37.5), 1)
            bp = random.randint(138, 175)
            glucose = random.randint(126, 220)
            cholesterol = random.randint(220, 310)
            heart_rate = random.randint(82, 105)
            smoker = random.choice(["Yes", "Yes", "No"])
            diabetes = random.choice(["Yes", "Yes", "No"])
            risk = "High"
            los = random.randint(6, 12)

        # add controlled noise so it is not too cartoonish
        if random.random() < 0.08:
            risk = random.choice(["Low", "Medium", "High"])

        rows.append({
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "blood_pressure": bp,
            "glucose": glucose,
            "cholesterol": cholesterol,
            "heart_rate": heart_rate,
            "smoker": smoker,
            "diabetes_history": diabetes,
            "risk_label": risk,
            "length_of_stay": los
        })

    df = pd.DataFrame(rows)
    df.to_csv(PATIENTS_PATH, index=False)
    print(f"[OK] Wrote {len(df)} patient rows to {PATIENTS_PATH}")

def generate_reviews():
    positive_templates = [
        "The doctor was very helpful and explained everything clearly.",
        "The nursing staff were caring, attentive, and professional.",
        "Clean room, quick service, and a smooth admission process.",
        "I was satisfied with the treatment and overall hospital support.",
        "The staff were friendly and the discharge process was clear.",
        "Excellent coordination between doctors and nurses.",
        "The room was clean and the service was prompt.",
        "Very supportive team and effective treatment.",
        "The reception staff were polite and helpful.",
        "Overall a comfortable and reassuring experience."
    ]

    negative_templates = [
        "Long waiting time and poor coordination at reception.",
        "The room was dirty and the service was delayed.",
        "I am unhappy with the discharge process and communication.",
        "The staff were rude and the instructions were unclear.",
        "Very slow service and confusing billing process.",
        "The doctor did not explain the treatment properly.",
        "The ward was crowded and not clean.",
        "Poor hygiene and delayed response from staff.",
        "The treatment was not helpful and the service was bad.",
        "I had a frustrating experience due to waiting and neglect."
    ]

    mixed_positive = [
        "Treatment was good but waiting time was a bit long.",
        "The doctor was helpful although the billing process was slow.",
        "The room was clean and staff were caring, but discharge took time.",
        "Good medical support, though reception coordination could improve.",
        "Overall satisfactory care with minor delays."
    ]

    mixed_negative = [
        "The nurses were polite but the room was not clean.",
        "The doctor explained things well, but service was delayed.",
        "Treatment was satisfactory, however waiting time was terrible.",
        "The hospital was helpful but billing support was confusing.",
        "The staff were caring, yet the discharge process was poor."
    ]

    rows = []

    for _ in range(120):
        text = random.choice(positive_templates)
        rows.append({"review": text, "sentiment": "positive"})

    for _ in range(120):
        text = random.choice(negative_templates)
        rows.append({"review": text, "sentiment": "negative"})

    for _ in range(30):
        text = random.choice(mixed_positive)
        rows.append({"review": text, "sentiment": "positive"})

    for _ in range(30):
        text = random.choice(mixed_negative)
        rows.append({"review": text, "sentiment": "negative"})

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(REVIEWS_PATH, index=False)
    print(f"[OK] Wrote {len(df)} review rows to {REVIEWS_PATH}")

if __name__ == "__main__":
    generate_patients(n=250)
    generate_reviews()
