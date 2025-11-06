import os
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()
SECRET_KEY = os.environ.get("SECRET_KEY")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

async def solve_quiz_task(url: str, email: str, secret: str):
    """
    This is our asynchronous worker.
    It runs in the background, independent of the API response.
    In future phases, this is where all the "agent" logic will go.
    """
    print("--------------------------------------------------")
    print(f"[WORKER] 🤖 Task accepted for URL: {url}")
    print(f"[WORKER] ⏳ Simulating 10 seconds of hard work...")
    await asyncio.sleep(10) 
    
    print(f"[WORKER] ✅ Finished processing task for {url}")
    print("--------------------------------------------------")

@app.get("/")
def read_root():
    return {"message": "Hello World. LLM Analysis Quiz agent is standing by."}

@app.get("/health")
def read_health():
    return {"status": "ok"}

@app.post("/quiz")
async def handle_quiz_request(request: QuizRequest, background_tasks: BackgroundTasks):
    """
    This endpoint now does two things:
    1. Validates the secret.
    2. Immediately returns a 200 OK response.
    3. Adds the *actual* work to a background task queue.
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

    background_tasks.add_task(
        solve_quiz_task, 
        url=request.url, 
        email=request.email, 
        secret=request.secret
    )

    return {"status": "Job accepted and processing in background."}