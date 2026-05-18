from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str


class AskRequest(BaseModel):
    question: str
    document_id: str = Field(..., min_length=1)


class EnhanceResponse(BaseModel):
    original: str
    enhanced: str
