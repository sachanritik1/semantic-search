# app/schemas/tokens.py

from pydantic import BaseModel


class TokenCountRequest(BaseModel):
    text: str


class TokenCountResponse(BaseModel):
    token_count: int
    tokens: list[int]
