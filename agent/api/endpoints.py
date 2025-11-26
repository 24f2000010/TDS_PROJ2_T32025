import os
import asyncio
import os
import asyncio
from json import JSONDecodeError

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import ValidationError

from agent.models.schemas import QuizRequest
from agent.core.worker import solve_quiz_task

router = APIRouter()
SECRET_KEY = os.environ.get("SECRET_KEY")
TASK_TIMEOUT = 180.0  # 3 minutes as per requirements


@router.get("/")
def read_root():
    return {"message": "Gemini 2.5 Pro Generalist Agent is live."}


@router.get("/health")
def read_health():
    """For the uptime monitor."""
    return {"status": "ok"}


@router.post("/quiz")
async def handle_quiz_request(request: Request, background_tasks: BackgroundTasks):
    """
    Entry point for quiz tasks.
    - Returns 400 when payload is not valid JSON.
    - Returns 422 when JSON is valid but fails schema validation.
    """

    if not SECRET_KEY:
        raise HTTPException(status_code=500, detail="Server SECRET_KEY not configured.")

    try:
        payload = await request.json()
    except JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    try:
        quiz_request = QuizRequest(**payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(),
        ) from exc

    if quiz_request.secret != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid secret.")

    task_data = quiz_request.model_dump()

    async def run_with_timeout(data):
        try:
            print(f"[SUPERVISOR] Starting task chain {data.get('url')} with {TASK_TIMEOUT}s timeout.")
            await asyncio.wait_for(solve_quiz_task(data), timeout=TASK_TIMEOUT)
        except asyncio.TimeoutError:
            print(f"[SUPERVISOR] ❌ CRITICAL: Task chain timed out after {TASK_TIMEOUT}s!")

    background_tasks.add_task(run_with_timeout, data=task_data)

    return {"status": "Job accepted. Processing in background."}