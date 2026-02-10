
# app/main.py

from app.services.tokenizer import tokenize
from app.schemas.tokens import TokenCountRequest, TokenCountResponse
from fastapi import FastAPI
from pydantic import BaseModel
from app.services.embedder import embeddings
from app.services.vector_store import get_vector_store
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

app = FastAPI(title="RAG API")

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
def ingest_data():

    file_path = "./nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)

    print(f"Total Chunks: {len(all_splits)}\n")
    for i, chunk in enumerate(all_splits):
        print(f"Chunk {i+1}: {chunk}")

    vector1 = embeddings.embed_query(all_splits[0].page_content)
    print(f"Vector 1 Length: {len(vector1)}")
    print(f"Vector 1 Sample: {vector1[:10]}")

    

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
