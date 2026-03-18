import re
from pathlib import Path

import joblib
from nltk.sentiment import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parents[2]
NLP_DIR = BASE_DIR / "artifacts" / "nlp"

POSITIVE_WORDS = {
    "helpful", "caring", "attentive", "clean", "excellent", "good", "great",
    "friendly", "professional", "quick", "supportive", "satisfactory",
    "smooth", "clear", "effective", "comfortable"
}

NEGATIVE_WORDS = {
    "delay", "delayed", "late", "poor", "bad", "dirty", "unclean", "rude",
    "unhappy", "slow", "confusing", "painful", "worst", "terrible", "long waiting",
    "waiting", "crowded", "neglect", "ignored", "unclear", "problem", "issues"
}

NEGATION_WORDS = {"not", "no", "never", "hardly", "barely", "without"}

class HybridSentimentEngine:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.model = None
        self.vectorizer = None

        model_path = NLP_DIR / "sentiment_model.pkl"
        vec_path = NLP_DIR / "tfidf_vectorizer.pkl"

        if model_path.exists() and vec_path.exists():
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def keyword_score(self, text: str) -> int:
        tokens = text.split()
        score = 0

        for phrase in POSITIVE_WORDS:
            if phrase in text:
                score += 1

        for phrase in NEGATIVE_WORDS:
            if phrase in text:
                score -= 1

        # crude negation handling
        for i, token in enumerate(tokens[:-1]):
            if token in NEGATION_WORDS:
                nxt = tokens[i + 1]
                if nxt in POSITIVE_WORDS:
                    score -= 2
                if nxt in NEGATIVE_WORDS:
                    score += 2

        return score

    def predict(self, text: str):
        text = self.normalize(text)

        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores["compound"]
        keyword = self.keyword_score(text)

        ml_pred = None
        ml_conf = None
        if self.model is not None and self.vectorizer is not None:
            X = self.vectorizer.transform([text])
            ml_pred = self.model.predict(X)[0]
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                ml_conf = float(max(probs))

        # combine signals
        if compound >= 0.25 and keyword >= 0:
            final = "positive"
        elif compound <= -0.25 and keyword <= 0:
            final = "negative"
        elif keyword >= 2:
            final = "positive"
        elif keyword <= -2:
            final = "negative"
        elif ml_pred is not None:
            final = str(ml_pred)
        else:
            final = "neutral"

        return {
            "label": final,
            "vader_compound": round(compound, 4),
            "keyword_score": keyword,
            "ml_prediction": ml_pred,
            "ml_confidence": round(ml_conf, 4) if ml_conf is not None else None
        }
