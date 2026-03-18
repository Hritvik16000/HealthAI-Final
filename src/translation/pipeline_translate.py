from pathlib import Path
import pandas as pd
from sacrebleu import corpus_bleu

from src.translation.translator_utils import EnHiTranslator

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "artifacts" / "translation"
REPORTS = BASE_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

translator = EnHiTranslator()

samples_en = [
    "Please take your medicines on time.",
    "Seek urgent care if chest pain worsens.",
    "Drink enough water to avoid dehydration."
]
refs_hi = [
    "कृपया अपनी दवाइयाँ समय पर लें।",
    "यदि छाती का दर्द बढ़े तो तुरंत चिकित्सा लें।",
    "डिहाइड्रेशन से बचने के लिए पर्याप्त पानी पिएँ।"
]

pred_hi = [translator.translate(x) for x in samples_en]
bleu = corpus_bleu(pred_hi, [refs_hi]).score

pd.DataFrame({
    "english": samples_en,
    "predicted_hindi": pred_hi,
    "reference_hindi": refs_hi
}).to_csv(OUT_DIR / "translation_samples.csv", index=False)

pd.DataFrame([{
    "module": "translation_en_hi",
    "bleu": round(float(bleu), 4)
}]).to_csv(REPORTS / "translation_metrics.csv", index=False)

print("[OK] Saved translation artifacts")
print("BLEU:", round(float(bleu), 4))
