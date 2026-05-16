from pydantic import BaseModel


class PromptTestRequest(BaseModel):
    template: str
    variables: dict[str, str]
