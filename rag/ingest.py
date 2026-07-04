import json
import logging
import re
from pathlib import Path

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

MIMIC_CSV   = Path("data/NOTEEVENTS.csv")
SYNTHEA_CSV = Path("data/synthea_notes.csv")
INDEX_DIR   = Path("data/faiss_index")
META_PATH   = Path("data/notes_metadata.json")

KEEP_CATEGORIES = {"Discharge summary", "Nursing", "Physician", "Radiology"}
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _resolve_notes_csv() -> Path:
    if MIMIC_CSV.exists() and MIMIC_CSV.stat().st_size > 500:
        logger.info("Using MIMIC-III notes: %s", MIMIC_CSV)
        return MIMIC_CSV
    if SYNTHEA_CSV.exists():
        logger.info("MIMIC notes not available — using Synthea-generated notes: %s", SYNTHEA_CSV)
        return SYNTHEA_CSV
    raise FileNotFoundError(
        "No notes CSV found. Either:\n"
        "  a) Place NOTEEVENTS.csv in data/ (MIMIC-III)\n"
        "  b) Run `python scripts/generate_synthea_notes.py` to generate from Synthea data"
    )


def clean_note(text: str) -> str:
    text = re.sub(r"\[\*\*.*?\*\*\]", "[REDACTED]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_notes(csv_path: Path | None = None) -> list[dict]:
    if csv_path is None:
        csv_path = _resolve_notes_csv()
    df = pd.read_csv(csv_path)
    df.columns = [c.upper() for c in df.columns]
    df = df[["SUBJECT_ID", "HADM_ID", "CHARTDATE", "CATEGORY", "DESCRIPTION", "TEXT"]]
    df = df[df["CATEGORY"].isin(KEEP_CATEGORIES)].dropna(subset=["TEXT"])
    df["TEXT"] = df["TEXT"].apply(clean_note)
    df = df[df["TEXT"].str.len() > 50]
    logger.info(
        "Loaded %d notes across %d patients (categories: %s)",
        len(df), df["SUBJECT_ID"].nunique(), ", ".join(sorted(df["CATEGORY"].unique())),
    )
    return df.rename(columns={
        "SUBJECT_ID": "patient_id", "HADM_ID": "admission_id",
        "CHARTDATE": "note_date", "CATEGORY": "note_type",
        "DESCRIPTION": "note_description", "TEXT": "text",
    }).to_dict("records")


def build_index(notes: list[dict]) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs, metadata = [], []
    for note in notes:
        for i, chunk in enumerate(splitter.split_text(note["text"])):
            docs.append(chunk)
            metadata.append({
                "patient_id":       str(note["patient_id"]),
                "admission_id":     str(note.get("admission_id", "")),
                "note_type":        note["note_type"],
                "note_date":        str(note.get("note_date", "")),
                "chunk_index":      i,
                "note_description": note.get("note_description", ""),
            })

    logger.info("Chunked into %d text segments; building embeddings...", len(docs))
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(docs, embeddings, metadatas=metadata)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    META_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("Index saved: %d chunks from %d notes → %s", len(docs), len(notes), INDEX_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    notes = load_notes()
    build_index(notes)
