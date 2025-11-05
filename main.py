from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI()

@app.get("/")
def read_root():
    """
    Root endpoint that returns a simple Hello World message.
    This is just to verify that the server is running.
    """
    return {"message": "Hello World. LLM Analysis Quiz agent is standing by."}

@app.get("/health")
def read_health():
    """
    A simple health check endpoint.
    Uptime monitors will ping this.
    It just needs to return a 200 OK.
    """
    # A simple, lightweight JSON response
    return {"status": "ok"}