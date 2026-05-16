
# app/main.py

import os
import tempfile

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel

from app.config import settings
from app.db.document_store import init_db, list_chunks, save_documents
from app.db.vector_store import upsert_documents
from app.dependencies import get_llm_service
from app.schemas.tokens import TokenCountRequest, TokenCountResponse
from app.services.dense_retriever import DenseRetriever
from app.services.embedder import embeddings
from app.services.llm_service import LLMService
from app.services.re_ranker import re_rank_docs
from app.services.self_consistency import generate_with_self_consistency
from app.services.sparse_retriever import SparseRetriever
from app.utils.chunker import text_splitter
from app.utils.prompt_loader import load_prompt, render_prompt
from app.utils.prompts import build_prompt
from app.utils.tokenizer import tokenize

app = FastAPI(title="RAG API")


@app.on_event("startup")
def startup() -> None:
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ.setdefault("LANGCHAIN_PROJECT", "semantic-search")
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
    
    query = request.question
    dense = DenseRetriever(embeddings, default_k=10)
    dense_docs = dense.retrieve(query)
    print(f"Retrieved {len(dense_docs)} dense documents.")
    try:
        dense_ranked = await re_rank_docs(
            query,
            dense_docs,
            llm_service=llm_service,
            top_n=5,
            max_candidates=8,
            max_doc_chars=200,
            max_tokens=64,
            timeout_s=12.0,
            batch_count=2,
        )
    except (TimeoutError, ValueError) as exc:
        print(f"Dense rerank failed: {exc}")
        dense_ranked = dense_docs[:5]

    chunks = list_chunks()
    sparse_docs: list[Document] = []
    if chunks:
        texts = [c.content for c in chunks]
        sparse = SparseRetriever()
        sparse.build_index(texts)
        sparse_res = sparse.query(query, top_k=10)
        for idx, _, _ in sparse_res:
            chunk = chunks[idx]
            metadata = dict(chunk.meta or {})
            metadata.setdefault("source", chunk.source)
            metadata.setdefault("chunk_index", chunk.chunk_index)
            sparse_docs.append(Document(page_content=chunk.content, metadata=metadata))
    print(f"Retrieved {len(sparse_docs)} sparse documents.")
    if sparse_docs:
        try:
            sparse_ranked = await re_rank_docs(
                query,
                sparse_docs,
                llm_service=llm_service,
                top_n=5,
                max_candidates=8,
                max_doc_chars=200,
                max_tokens=64,
                timeout_s=12.0,
                batch_count=2,
            )
        except (TimeoutError, ValueError) as exc:
            print(f"Sparse rerank failed: {exc}")
            sparse_ranked = sparse_docs[:5]
    else:
        sparse_ranked = []

    combined_docs: list[Document] = []
    seen: set[str] = set()
    for doc in dense_ranked + sparse_ranked:
        key = doc.page_content.strip()
        if key in seen:
            continue
        seen.add(key)
        combined_docs.append(doc)

    prompt_text = build_prompt(docs=combined_docs, question=request.question)

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
    query = request.question
    dense = DenseRetriever(embeddings, default_k=request.top_k)
    dense_docs = dense.retrieve(query, k=request.top_k)

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