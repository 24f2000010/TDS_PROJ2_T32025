# LLM Analysis Quiz Agent

An intelligent agent that solves data science quizzes using LLM capabilities.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **Configure environment variables** (via `.env`, Render dashboard, etc.)

   | Variable | Description |
   | --- | --- |
   | `SECRET_KEY` | Shared secret used to authenticate `/quiz` requests (must match Google Form submission). |
   | `STUDENT_EMAIL` | Email identifier propagated to quiz submissions. |
   | `AIPIPE_API_KEY` | API key for the LLM provider (Gemini via AIPipe). |
   | `AIPIPE_BASE_URL` | Base URL for the LLM provider. |

   Example `.env`:
   ```env
   SECRET_KEY=your_secret_key
   STUDENT_EMAIL=your_email@example.com
   AIPIPE_API_KEY=your_api_key
   AIPIPE_BASE_URL=https://your-api-url.com
   ```

## Usage

**Start server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

## API

- `GET /health` - Health check
- `POST /quiz` - Submit quiz request

Request format:
```json
{
  "email": "your_email@example.com",
  "secret": "your_secret",
  "url": "https://quiz-url.com"
}
```

## Architecture Overview

1. **API Layer (`main.py`, `agent/api/endpoints.py`)**
   - Accepts `POST /quiz` requests.
   - Validates JSON (returns 400 on malformed payloads, 422 on schema errors).
   - Verifies `secret`, then hands off work to a background task with a 180 s timeout.

2. **Supervisor (`agent/core/worker.py :: solve_quiz_task`)**
   - Launches a headless Chromium browser via Playwright.
   - Iterates through the quiz chain, loading each URL and delegating to the solver loop.
   - Captures submission responses and decides whether to continue, retry, or exit.

3. **Solver loop (`run_single_task_loop`)**
   - Performs a See → Think → Act cycle up to 15 times per quiz.
   - “See”: extracts rendered HTML (including base64-encoded instructions).
   - “Think”: prompts Gemini 2.5 Pro with the system prompt, maintaining conversation history.
   - “Act”: executes the requested tool (click, read_file, run_python_code, etc.), logs tool output, and feeds it back to the LLM.
   - Submits the final answer by POSTing `{email, secret, url, answer}` to the server-provided submission URL.

4. **Toolbox (`agent/core/tools.py`)**
   - Browser actions: click, fill text, screenshot + vision.
   - Retrieval: HTTP GET, file download (PDF/CSV/text).
   - Processing: ad‑hoc Python execution for data wrangling.
   - Submission: validates answer format, enforces the 1 MB payload limit, and POSTs the answer.

## Test Cases

See `tests/` folder for 24 test cases across 6 categories:
- Web Scraping (4 tests)
- API Sourcing (4 tests)
- Cleansing (4 tests)
- Processing (4 tests)
- Analysis (4 tests)
- Visualization (4 tests)

Each test includes `index.html` with instructions and `generate_data.py` to create test data.
