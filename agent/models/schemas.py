from pydantic import BaseModel, ConfigDict

class QuizRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    email: str
    secret: str
    url: str