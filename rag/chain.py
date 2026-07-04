import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag.retriever import load_retriever

load_dotenv()

SYSTEM_PROMPT = """You are a clinical decision support assistant.
Answer the clinician's question using ONLY the provided clinical note excerpts.
For each fact you state, cite the source as [Note: {{note_type}}, {{note_date}}].
If the notes do not contain enough information to answer, say so clearly.
Never fabricate clinical information.

Clinical note excerpts:
{context}
"""


def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        meta = doc.metadata
        header = (
            f"[Note type: {meta.get('note_type', '?')} | "
            f"Date: {meta.get('note_date', '?')} | "
            f"Patient: {meta.get('patient_id', '?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(
    patient_id: str | None = None,
    vectorstore=None,
    embeddings_model=None,
):
    if vectorstore is not None and embeddings_model is not None:
        from rag.retriever import _FAISSRetriever
        retriever = _FAISSRetriever(
            vectorstore=vectorstore,
            embeddings_model=embeddings_model,
            patient_id=patient_id,
        )
    else:
        retriever = load_retriever(patient_id=patient_id)
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, patient_id: str | None = None) -> str:
    chain = build_rag_chain(patient_id=patient_id)
    return chain.invoke(question)
