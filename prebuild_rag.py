# prebuild_rag.py
# Precalcula y persiste un índice TF-IDF sobre assets/pdfs/*.pdf en assets/index/.

import os
import numpy as np

try:
    from pypdf import PdfReader
except Exception as e:
    raise SystemExit("Falta pypdf. Instala con: pip install pypdf") from e

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as e:
    raise SystemExit("Falta scikit-learn. Instala con: pip install scikit-learn") from e

try:
    from scipy.sparse import csr_matrix  # noqa: F401 (solo para asegurar dependencia)
except Exception as e:
    raise SystemExit("Falta scipy. Instala con: pip install scipy") from e

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(APP_DIR, "assets", "pdfs")
IDX_DIR = os.path.join(APP_DIR, "assets", "index")
os.makedirs(IDX_DIR, exist_ok=True)

INDEX_NPZ = os.path.join(IDX_DIR, "rag_tfidf.npz")
INDEX_TSV = os.path.join(IDX_DIR, "rag_meta.tsv")

def read_pdfs(pdf_dir, chunk_chars=1200, overlap=200):
    chunks = []
    pdfs = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise SystemExit(f"No se encontraron PDFs en {pdf_dir}. Añade PDFs y reintenta.")
    for p in pdfs:
        try:
            full = ""
            reader = PdfReader(p)
            for page in reader.pages:
                try:
                    full += page.extract_text() or ""
                except Exception:
                    continue
            full = " ".join(full.split())
            i = 0
            while i < len(full):
                j = min(len(full), i + chunk_chars)
                txt = full[i:j]
                chunks.append({"src": os.path.basename(p), "text": txt})
                i = max(0, j - overlap)
        except Exception as e:
            print(f"[WARN] Error leyendo {p}: {e}")
    if not chunks:
        raise SystemExit("No se extrajo texto de los PDFs. Revisa que no estén escaneados sin OCR.")
    return chunks

def build_and_save(chunks):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(texts)  # CSR
    vocab = vectorizer.vocabulary_
    idf = getattr(vectorizer, "idf_", None)

    np.savez_compressed(
        INDEX_NPZ,
        indptr=matrix.indptr.astype(np.int64),
        indices=matrix.indices.astype(np.int32),
        data=matrix.data.astype(np.float32),
        vocab_tokens=np.array(list(vocab.keys()), dtype=object),
        vocab_ids=np.array(list(vocab.values()), dtype=np.int32),
        idf=(idf.astype(np.float32) if idf is not None else None),
    )
    with open(INDEX_TSV, "w", encoding="utf-8") as f:
        for c in chunks:
            txt = c["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{c['src']}\t{txt}\n")

def main():
    print(f"[RAG] Leyendo PDFs desde {PDF_DIR} ...")
    chunks = read_pdfs(PDF_DIR)
    print(f"[RAG] Chunks: {len(chunks)}")
    print("[RAG] Construyendo TF-IDF ...")
    build_and_save(chunks)
    print(f"[RAG] Guardado índice en:\n  {INDEX_NPZ}\n  {INDEX_TSV}")
    print("[OK] Listo para compilar la app con el índice embebido.")

if __name__ == "__main__":
    main()
