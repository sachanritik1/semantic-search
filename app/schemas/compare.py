from pydantic import BaseModel


class CompareRequest(BaseModel):
    question: str
    top_k: int = 5
