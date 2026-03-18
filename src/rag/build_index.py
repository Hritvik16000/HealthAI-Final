from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
KB_PATH = BASE_DIR / "data" / "knowledge" / "medical_faq.txt"
OUT_DIR = BASE_DIR / "artifacts" / "rag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

text = KB_PATH.read_text(encoding="utf-8").strip()
chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_tensor=False, normalize_embeddings=True)

joblib.dump({"chunks": chunks, "embeddings": embeddings}, OUT_DIR / "rag_index.pkl")
print("[OK] Saved rag_index.pkl with", len(chunks), "chunks")
