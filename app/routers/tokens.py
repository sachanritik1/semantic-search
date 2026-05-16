from fastapi import APIRouter

from app.schemas.tokens import TokenCountRequest, TokenCountResponse
from app.utils.tokenizer import tokenize

router = APIRouter(tags=["tokens"])


@router.post("/tokens/count", response_model=TokenCountResponse)
def count_tokens_api(request: TokenCountRequest):
    tokens = tokenize(request.text)
    return TokenCountResponse(token_count=len(tokens), tokens=tokens)
