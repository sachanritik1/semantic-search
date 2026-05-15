
# app/main.py

from app.services.self_consistency import generate_with_self_consistency
from app.services.tokenizer import tokenize
from app.schemas.tokens import TokenCountRequest, TokenCountResponse
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.services.embedder import embeddings
from app.services.vector_store import get_vector_store, upsert_documents
from langchain_community.document_loaders import PyPDFLoader
from app.services.prompts import build_prompt
from app.services.re_ranker import re_rank_docs
from app.services.chunker import text_splitter
from fastapi import Depends
from app.dependencies import get_llm_service
from app.services.llm_service import LLMService
from app.services.prompt_loader import load_prompt, render_prompt
from app.dependencies import get_llm_service
from fastapi import Depends
from app.services.llm_service import LLMService
from app.services.sparse_retriever import SparseRetriever
from app.services.document_store import init_db, list_chunks, save_documents

app = FastAPI(title="RAG API")


@app.on_event("startup")
def startup() -> None:
    init_db()

@app.get("/health")
def health():
    return {"status": "ok"}


class QuestionRequest(BaseModel):
    question: str

@app.post("/llm/test")
def test_llm(
    request: QuestionRequest,
    llm_service: LLMService = Depends(get_llm_service),
):
    response = llm_service.generate_text(request.question)
    
    return {"response": response}

@app.post("/ingest")
async def ingest_data(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    all_splits = text_splitter.split_documents(docs)

    print(f"Total Chunks: {len(all_splits)}\n")
    for i, chunk in enumerate(all_splits):
        print(f"Chunk {i+1}: {chunk}")

    upsert_documents(embeddings, all_splits)
    saved = save_documents(all_splits)

    return {
        "chunks_total": len(all_splits),
        "chunks_saved": saved,
    }

    

@app.post("/ask")
async def ask_question(
    request: QuestionRequest,
    llm_service: LLMService = Depends(get_llm_service),
):
    
    vector_store = get_vector_store(embeddings)
    retriever = vector_store.as_retriever(search_kwargs={
            "k": 20,
            # "filter": qdrant_filter
        })
    query = request.question
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents.")
    re_ranked_docs = await re_rank_docs(query, docs, llm_service=llm_service)
    print(f"Re-ranked to {len(re_ranked_docs)} documents.")
    print("Top documents after re-ranking:")
    for i, doc in enumerate(re_ranked_docs):
        print(f"Document {i+1}: {doc.page_content}")

    prompt_text = build_prompt(docs=re_ranked_docs, question=request.question)

    response = llm_service.generate_text(prompt_text)
    content = response.content

    return {"response": content}


@app.post("/tokens/count", response_model=TokenCountResponse)
def count_tokens_api(request: TokenCountRequest):
    tokens = tokenize(request.text)
    return TokenCountResponse(token_count=len(tokens), tokens=tokens)




class PromptTestRequest(BaseModel):
    template: str
    variables: dict[str, str]

@app.post("/prompt/test")
async def test_prompt(
    request: PromptTestRequest,
    llm_service: LLMService = Depends(get_llm_service),
):  
    try:
        template = load_prompt(request.template)
        prompt = render_prompt(template, request.variables)

        response = await llm_service.generate_text_async(prompt)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}        


@app.post("/self-consistency")
async def self_consistency_test(
    request: QuestionRequest,
    llm_service: LLMService = Depends(get_llm_service),
):
    final_answer = await generate_with_self_consistency(
        llm_service=llm_service,
        prompt=request.question,
        runs=5,
    )
    return {"final_answer": final_answer}


class CompareRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/compare")
async def compare_retrievers(
    request: CompareRequest,
):
    """Compare dense (RAG) and sparse (BM25) retrievers.

    Approach:
    - Use the existing vector store retriever to fetch `top_k` dense documents.
    - Build a sparse BM25 index over ingested chunks stored in the document store.
    - Return both sets of retrieved items for comparison.
    """
    vector_store = get_vector_store(embeddings)
    dense_retriever = vector_store.as_retriever(search_kwargs={
            "k": request.top_k,
        })
    query = request.question
    dense_docs = list(dense_retriever.invoke(query))

    dense_results = []
    for i, doc in enumerate(dense_docs):
        metadata = getattr(doc, "metadata", None)
        dense_results.append({
            "index": i,
            "content": doc.page_content,
            "metadata": metadata if metadata is not None else {},
        })

    # Build sparse retriever over ingested chunks
    chunks = list_chunks()
    texts = [c.content for c in chunks]
    sparse = SparseRetriever()
    if texts:
        sparse.build_index(texts)
        sparse_res = sparse.query(query, top_k=request.top_k)
        sparse_results = [
            {
                "index": idx,
                "score": score,
                "content": text,
                "chunk_id": chunks[idx].id,
                "source": chunks[idx].source,
                "chunk_index": chunks[idx].chunk_index,
                "metadata": chunks[idx].meta,
            }
            for idx, score, text in sparse_res
        ]
    else:
        sparse_results = []

    return {"dense": dense_results, "sparse": sparse_results}