import asyncio
import httpx
import pdfplumber
import io
import pandas as pd
import numpy as np
import traceback
import base64
import os
import json
from typing import Any, Tuple
from playwright.async_api import Page
from openai import AsyncOpenAI

# Lazy initialization to avoid errors during import when API key is not set
_llm_client = None

def get_llm_client():
    """Get or create the LLM client instance"""
    global _llm_client
    if _llm_client is None:
        api_key = os.environ.get("AIPIPE_API_KEY")
        base_url = os.environ.get("AIPIPE_BASE_URL")
        if api_key:
            _llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            # Return a mock client for testing when API key is not set
            # This allows tests to run without API credentials
            class MockCompletions:
                async def create(self, *args, **kwargs):
                    raise RuntimeError("AIPIPE_API_KEY not set")
            class MockChat:
                def __init__(self):
                    self.completions = MockCompletions()
            class MockClient:
                def __init__(self):
                    self.chat = MockChat()
            _llm_client = MockClient()
    return _llm_client

async def tool_click(page: Page, selector: str):
    """Uses Playwright to click an element based on its CSS selector."""
    print(f"[TOOL] 🦾 CLICK: {selector}")
    if not selector:
        raise ValueError("No selector provided for click tool")
    await page.locator(selector).first.click(timeout=5000)
    return f"Clicked element '{selector}'."

async def tool_fill_text(page: Page, selector: str, text: str):
    """Uses Playwright to fill a text field."""
    print(f"[TOOL] 🦾 FILL: {selector} with '{text}'")
    if not selector:
        raise ValueError("No selector provided for fill_text tool")
    await page.locator(selector).first.fill(text, timeout=5000)
    return f"Filled '{selector}'."

async def tool_call_api(url: str, headers: dict = None):
    """Makes a GET request to an API. FAIL FAST strategy."""
    print(f"[TOOL] 📡 Calling API: {url}")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            return f"Status: {response.status_code}\nBody: {response.text[:5000]}"
    except Exception as e:
        return f"Error calling API: {str(e)}"

async def tool_read_file(url: str):
    """Downloads a file (PDF, CSV, text) and extracts its content."""
    print(f"[TOOL] 📂 Downloading file: {url}")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return f"Error: Failed to download. Status: {response.status_code}"
            
            content_type = response.headers.get("content-type", "").lower()
            
            if "pdf" in content_type or url.endswith(".pdf"):
                print("[TOOL]  Processing PDF...")
                with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                    text = ""
                    for i, page in enumerate(pdf.pages[:10]):
                        text += f"--- Page {i+1} ---\n{page.extract_text()}\n"
                    return text
            
            elif "csv" in content_type or url.endswith(".csv"):
                print("[TOOL]  Processing CSV...")
                return response.text
            
            else:
                return response.text[:10000]
                
    except Exception as e:
        return f"Error reading file: {str(e)}"

async def tool_run_python_code(code: str):
    """
    Executes a snippet of Python code for data analysis.
    The code can import any library installed in the environment.
    """
    print(f"[TOOL] 🐍 RUN PYTHON: \n{code}")
    isolated_globals = {"__builtins__": __builtins__}
    
    code_out = io.StringIO()
    
    try:
        import sys
        original_stdout = sys.stdout
        sys.stdout = code_out
        
        exec(code, isolated_globals)
        
        sys.stdout = original_stdout
        
        output = code_out.getvalue()
        if not output:
            return "Code executed successfully (no print output)."
        return f"Python output:\n{output}"

    except Exception as e:
        sys.stdout = original_stdout
        tb = traceback.format_exc()
        return f"Error executing Python code: {e}\n{tb}"

async def tool_take_screenshot_and_analyze(page: Page, analysis_prompt: str):
    """Takes a screenshot, sends it to Gemini 2.5 Pro for analysis."""
    print(f"[TOOL] 📸 Taking screenshot for analysis...")
    
    screenshot_bytes = await page.screenshot()
    screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
    
    print(f"[TOOL]  Screenshot captured. Sending to Gemini for analysis...")
    
    try:
        llm_client = get_llm_client()
        response = await llm_client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "google/gemini-2.5-pro"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        analysis = response.choices[0].message.content
        print(f"[TOOL]  Vision analysis complete.")
        return analysis
    except Exception as e:
        print(f"[TOOL]  ❌ Vision analysis failed: {e}")
        return f"Error during vision analysis: {str(e)}"

def validate_answer_format(answer: Any) -> Tuple[bool, str]:
    """
    Validates that the answer is in one of the allowed formats:
    - boolean
    - number (int or float)
    - string
    - base64 URI (data:...;base64,...)
    - JSON object (dict) with combination of above
    
    Returns (is_valid, error_message)
    """
    if answer is None:
        return False, "Answer cannot be None"
    
    # Check for basic types
    if isinstance(answer, (bool, int, float, str)):
        return True, ""
    
    # Check for base64 URI
    if isinstance(answer, str) and answer.startswith("data:") and ";base64," in answer:
        return True, ""
    
    # Check for JSON object (dict)
    if isinstance(answer, dict):
        # Recursively validate all values in the dict
        for key, value in answer.items():
            is_valid, error = validate_answer_format(value)
            if not is_valid:
                return False, f"Invalid value for key '{key}': {error}"
        return True, ""
    
    # Check for list (which can be serialized as JSON)
    if isinstance(answer, list):
        for i, item in enumerate(answer):
            is_valid, error = validate_answer_format(item)
            if not is_valid:
                return False, f"Invalid item at index {i}: {error}"
        return True, ""
    
    return False, f"Answer must be boolean, number, string, base64 URI, or JSON object. Got {type(answer).__name__}"

def check_payload_size(payload: dict) -> Tuple[bool, str, int]:
    """
    Checks if the JSON payload is under 1MB.
    Returns (is_valid, error_message, size_in_bytes)
    """
    try:
        json_str = json.dumps(payload)
        size_bytes = len(json_str.encode('utf-8'))
        max_size = 1024 * 1024  # 1MB
        
        if size_bytes > max_size:
            return False, f"Payload size ({size_bytes} bytes) exceeds 1MB limit ({max_size} bytes)", size_bytes
        return True, "", size_bytes
    except Exception as e:
        return False, f"Error checking payload size: {e}", 0

async def tool_submit_answer(submission_url: str, answer_json: dict):
    """The FINAL tool. Submits the answer via HTTP POST and returns the response."""
    print(f"[TOOL] 📤 SUBMIT: {answer_json} to {submission_url}")
    
    # Validate answer format
    answer = answer_json.get("answer")
    if answer is not None:
        is_valid, error_msg = validate_answer_format(answer)
        if not is_valid:
            error = f"Invalid answer format: {error_msg}"
            print(f"[TOOL] ❌ {error}")
            return {
                "correct": False,
                "reason": error,
                "url": None
            }
    
    # Check payload size
    is_size_ok, size_error, size_bytes = check_payload_size(answer_json)
    if not is_size_ok:
        print(f"[TOOL] ❌ {size_error}")
        return {
            "correct": False,
            "reason": size_error,
            "url": None
        }
    print(f"[TOOL] ✓ Payload size: {size_bytes} bytes (under 1MB limit)")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                submission_url,
                json=answer_json,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"[TOOL] ✅ Submission successful. Response: {result}")
                    return result
                except Exception as e:
                    return {
                        "correct": False,
                        "reason": f"Failed to parse response JSON: {e}",
                        "url": None
                    }
            else:
                error_msg = f"Submission failed with status {response.status_code}: {response.text[:500]}"
                print(f"[TOOL] ❌ {error_msg}")
                return {
                    "correct": False,
                    "reason": error_msg,
                    "url": None
                }
    except Exception as e:
        error_msg = f"Error submitting answer: {str(e)}"
        print(f"[TOOL] ❌ {error_msg}")
        return {
            "correct": False,
            "reason": error_msg,
            "url": None
        }