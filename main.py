from typing import Any, Mapping, Protocol, cast

from fastapi import FastAPI
from pydantic import BaseModel
from src.embedder import embeddings
from src.vector_store import get_vector_store
from langchain_community.document_loaders import PyPDFLoader
from src.prompts import build_prompt
from src.llm import llm


app = FastAPI(title="RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}


class QuestionRequest(BaseModel):
    question: str


class LLMResponse(Protocol):
    content: str | list[str | Mapping[str, Any]]

@app.post("/ingest")
def ingest_data():

    file_path = "./nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()
    from src.chunker import text_splitter
    all_splits = text_splitter.split_documents(docs)

    print(f"Total Chunks: {len(all_splits)}\n")
    for i, chunk in enumerate(all_splits):
        print(f"Chunk {i+1}: {chunk}")

    vector1 = embeddings.embed_query(all_splits[0].page_content)
    print(f"Vector 1 Length: {len(vector1)}")
    print(f"Vector 1 Sample: {vector1[:10]}")

    


@app.post("/ask")
def ask_question(request: QuestionRequest):
    
    vector_store = get_vector_store(embeddings)
    retriever = vector_store.as_retriever(search_kwargs={
            "k": 4,
            # "filter": qdrant_filter
        })
    query = request.question
    docs = retriever.invoke(query)
    
    prompt_text = build_prompt(docs=docs, question=request.question)

    response = cast(LLMResponse, llm.invoke(prompt_text))
    content = response.content

    if isinstance(content, str):
        # Here: content is guaranteed to be str
        return content
    else:
        # Here: content is guaranteed to be list[str | dict[str, str]]
        return  "something went wrong"
    
