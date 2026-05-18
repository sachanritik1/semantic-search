from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    document_id: str | None = None


class EnhanceResponse(BaseModel):
    original: str
    enhanced: str
