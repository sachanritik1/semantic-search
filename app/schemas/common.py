from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class EnhanceResponse(BaseModel):
    original: str
    enhanced: str
