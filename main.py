import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

SECRET_KEY = os.environ.get("SECRET_KEY")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


@app.get("/")
def read_root():
    """
    Root endpoint.
    """
    return {"message": "Hello World. LLM Analysis Quiz agent is standing by."}


@app.get("/health")
def read_health():
    """
    A simple health check endpoint for the uptime monitor.
    """
    return {"status": "ok"}


@app.post("/quiz")
async def handle_quiz_request(request: QuizRequest):
    """
    This is the main endpoint for receiving quiz tasks.
    It validates the secret and (later) will trigger the agent.
    """
    if not SECRET_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Server is not configured with a SECRET_KEY."
        )

    if request.secret != SECRET_KEY:
        raise HTTPException(
            status_code=403, 
            detail="Invalid secret."
        )

    return {"status": "Job received. Starting processing."}