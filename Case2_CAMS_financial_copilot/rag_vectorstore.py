
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from transformers import pipeline

EXTRACTED_JSON = "Case2_CAMS_financial_copilot/extracted/pdf_texts.json"
TABLE_DIR = "Case2_CAMS_financial_copilot/extracted/tables"
INDEX_PATH = "Case2_CAMS_financial_copilot/extracted/faiss_index.bin"
META_PATH = "Case2_CAMS_financial_copilot/extracted/faiss_meta.json"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast

class RAGStore:
    def __init__(self, emb_model_name=EMB_MODEL_NAME):
        self.model = SentenceTransformer(emb_model_name)
        self.index = None
        self.meta = []

    def build(self):
        with open(EXTRACTED_JSON, "r", encoding="utf-8") as f:
            passages = json.load(f)

        texts = [p["text"] for p in passages]
        print(f"Encoding {len(texts)} passages...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype(np.float32))

        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False, indent=2)
        print("FAISS index and meta saved.")

    def load(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError("Index or meta missing. Run build() first.")
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model = SentenceTransformer(EMB_MODEL_NAME)

    def retrieve(self, query, top_k=4):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for idx in I[0]:
            results.append(self.meta[int(idx)])
        return results

# Simple LLM answerer: use HuggingFace pipeline (you can swap for OpenAI)
def generate_answer(question, passages):
    # Compose prompt
    context = "\n\n".join([f"Page {p['page']}: {p['text']}" for p in passages])
    prompt = f"Use the context below to answer the question concisely and cite page numbers.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    # small model pipeline (change model_name to larger LLM if available)
    gen = pipeline("text-generation", model="google/flan-t5-small", device=-1)  # CPU
    output = gen(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return output

# Utility function for consumption by chatbot
def answer_query(question, top_k=4):
    rag = RAGStore()
    rag.load()
    passages = rag.retrieve(question, top_k=top_k)
    answer = generate_answer(question, passages)
    return answer, passages

if __name__ == "__main__":
    # build index (one-time)
    rag = RAGStore()
    rag.build()
    print("Built index. Now you can call answer_query()")
