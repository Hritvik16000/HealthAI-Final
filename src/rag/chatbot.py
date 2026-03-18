from pathlib import Path
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_PATH = BASE_DIR / "artifacts" / "rag" / "rag_index.pkl"

class HealthRAGChatbot:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        payload = joblib.load(INDEX_PATH)
        self.chunks = payload["chunks"]
        self.embeddings = np.array(payload["embeddings"])
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, question: str, top_k: int = 3):
        q_emb = self.embedder.encode([question], convert_to_tensor=False, normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb
        idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in idx]

    def build_messages(self, question: str, contexts: list[str], history=None):
        system_prompt = (
            "You are a helpful healthcare assistant for an academic project demo. "
            "Answer the user's actual question directly and naturally. "
            "Use retrieved context only when relevant. "
            "Do not simply repeat the context. "
            "Do not claim to diagnose with certainty. "
            "If symptoms sound urgent, advise contacting a doctor or emergency care."
        )

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for turn in history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ["user", "assistant"]:
                    messages.append({"role": role, "content": content})

        context_block = "\n\n".join(contexts)
        messages.append({
            "role": "user",
            "content": f"""Relevant medical context:
{context_block}

User question:
{question}

Answer the user's question directly. Use the context if helpful, but do not copy it."""
        })
        return messages

    def ask(self, question: str, history=None, top_k: int = 3):
        contexts = self.retrieve(question, top_k=top_k)
        messages = self.build_messages(question, contexts, history=history)

        response = ollama.chat(
            model=self.model_name,
            messages=messages
        )

        answer = response["message"]["content"].strip()

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts
        }

if __name__ == "__main__":
    bot = HealthRAGChatbot()
    print(bot.ask("What should I do if chest pain becomes severe?")["answer"])
