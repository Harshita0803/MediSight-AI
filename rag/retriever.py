import json
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

INDEX_DIR       = Path("data/faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K           = 5


class _FAISSRetriever(BaseRetriever):
    vectorstore: Any = Field(...)
    embeddings_model: Any = Field(...)
    patient_id: str | None = None
    k: int = TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if not self.patient_id:
            return self.vectorstore.similarity_search(query, k=self.k)

        query_vec = np.array(
            [self.embeddings_model.embed_query(query)], dtype=np.float32
        )
        total = self.vectorstore.index.ntotal
        scores, raw_indices = self.vectorstore.index.search(query_vec, total)

        results: list[Document] = []
        for raw_idx in raw_indices[0]:
            if raw_idx == -1:
                continue
            doc_id = self.vectorstore.index_to_docstore_id[int(raw_idx)]
            doc = self.vectorstore.docstore.search(doc_id)
            if isinstance(doc, Document) and doc.metadata.get("patient_id") == self.patient_id:
                results.append(doc)
                if len(results) >= self.k:
                    break

        return results


def load_retriever(patient_id: str | None = None) -> _FAISSRetriever:
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_DIR}. "
            "Run `python -m rag.ingest` first to build the index."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    return _FAISSRetriever(
        vectorstore=vectorstore,
        embeddings_model=embeddings,
        patient_id=patient_id,
    )
