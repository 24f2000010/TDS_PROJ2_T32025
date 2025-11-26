from fastapi import FastAPI
from dotenv import load_dotenv
from agent.api.endpoints import router as api_router

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="LLM Router Agent")

app.include_router(api_router)