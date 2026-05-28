from fastapi import APIRouter

from app.api.schemas.tokens import TokenCountRequest, TokenCountResponse
from app.infrastructure.utils.tokenizer import tokenize

router = APIRouter(tags=["tokens"])


@router.post("/tokens/count", response_model=TokenCountResponse)
def count_tokens_api(request: TokenCountRequest):
    tokens = tokenize(request.text)
    return TokenCountResponse(token_count=len(tokens), tokens=tokens)
