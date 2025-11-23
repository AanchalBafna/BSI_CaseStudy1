
import pdfplumber
import camelot
import os
import json
import pandas as pd
from tqdm import tqdm

PDF_PATH = "Case2_CAMS_financial_copilot/pdf/CAMS_Result.pdf"
OUT_DIR = "Case2_CAMS_financial_copilot/extracted"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def extract_text_pages(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i+1, "text": text})
    return pages

def extract_tables(pdf_path, pages="all"):
    
    try:
        tables = camelot.read_pdf(pdf_path, pages=pages)
    except Exception as e:
        print("Camelot failed:", e)
        return []
    saved = []
    for i, table in enumerate(tables):
        csv_path = os.path.join(TABLE_DIR, f"table_{i+1}.csv")
        table.df.to_csv(csv_path, index=False, header=False)
        saved.append(csv_path)
    return saved

def cleanup_and_chunk_text(pages, chunk_size=600):
    
    chunks = []
    for p in pages:
        text = (p["text"] or "").strip()
        if not text:
            continue
        
        pos = 0
        while pos < len(text):
            chunk = text[pos:pos+chunk_size]
            chunks.append({
                "page": p["page"],
                "text": chunk
            })
            pos += chunk_size
    return chunks

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Put CAMS PDF at: {PDF_PATH}")

    print("Extracting pages...")
    pages = extract_text_pages(PDF_PATH)
    print(f"Extracted {len(pages)} pages")

    print("Extracting tables (pages 3-5 recommended)...")
    table_files = extract_tables(PDF_PATH, pages="3-5")
    print("Tables saved:", table_files)

    print("Chunking text for RAG")
    chunks = cleanup_and_chunk_text(pages)
    out_json = os.path.join(OUT_DIR, "pdf_texts.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Done. Outputs:")
    print(" - Text chunks:", out_json)
    print(" - Tables dir:", TABLE_DIR)
