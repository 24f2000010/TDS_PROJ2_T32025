import asyncio
import httpx
import pdfplumber
import io
import pandas as pd
import numpy as np
import traceback
import base64
import os
from playwright.async_api import Page
from openai import AsyncOpenAI

llm_client = AsyncOpenAI(
    api_key=os.environ.get("AIPIPE_API_KEY"),
    base_url=os.environ.get("AIPIPE_BASE_URL"),
)

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
        response = await llm_client.chat.completions.create(
            model="google/gemini-2.5-pro",
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

async def tool_submit_answer(submission_url: str, answer_json: dict):
    """The FINAL tool. Logs the answer and submission URL."""
    print(f"[TOOL] 📤 SUBMIT: {answer_json} to {submission_url}")
    return f"Task completed. Final answer logged: {answer_json}"