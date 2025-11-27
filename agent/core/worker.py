import os
import json
import asyncio
import re
import traceback
from playwright.async_api import async_playwright, Page
from openai import AsyncOpenAI
from agent.core.tools import *

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

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

SYSTEM_PROMPT = """
You are a "Generalist Agent," a highly intelligent AI controlling a web browser to solve a complex data science task.
Your goal is to solve the given task by navigating pages, reading files, analyzing data, and submitting the final answer.
You MUST operate in a step-by-step "See-Think-Act" loop.

**TASK:**
{task_hint}

**AVAILABLE TOOLS:**
You MUST respond with a single valid JSON object describing the *one* tool you want to use next.
(All curly braces in JSON examples must be escaped by doubling: {{ and }})

1.  **Web Navigation:**
    {{"tool": "click", "selector": "<css_selector>"}}
    {{"tool": "fill_text", "selector": "<css_selector>", "text": "<text_to_fill>"}}

2.  **Data Sourcing:**
    {{"tool": "call_api", "url": "<api_url>", "headers": {{}} }}
    {{"tool": "read_file", "url": "<file_url>"}}
       (Use this for PDFs, CSVs, or text files found on the page)

3.  **Data Analysis (Code):**
    {{"tool": "run_python_code", "code": "<python_code_snippet>"}}
       (CRITICAL: Use this for ALL math, parsing, filtering, or analysis.
       You MUST import any libraries you need (e.g., pandas, json).
       You MUST `print()` your final answer to get the output.)

4.  **Vision Analysis:**
    {{"tool": "take_screenshot_and_analyze", "analysis_prompt": "<what_to_look_for>"}}
       (Use this if the HTML is confusing or the task is visual, like a chart or image.)

5.  **Final Submission:**
    {{"tool": "submit_answer", "submission_url": "<url>", "answer_json": {{"answer": <value>}} }}
       (Use this *only* when you have the final answer for the current task.)

**CURRENT PAGE HTML:**
{html_content}
"""

def extract_submission_url(text: str) -> str:
    """
    Extracts the submission URL from quiz instructions text.
    Looks for patterns like "Post your answer to <url>" or "Post to <url>".
    Returns the URL if found, otherwise returns None.
    """
    if not text:
        return None
    
    # Pattern 1: "Post your answer to https://..."
    pattern1 = r'post\s+(?:your\s+)?answer\s+to\s+(https?://[^\s<>"\'\)]+)'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        url = match.group(1).rstrip('.,;:!?)')
        print(f"[EXTRACT] Found submission URL (pattern 1): {url}")
        return url
    
    # Pattern 2: "Post to https://..."
    pattern2 = r'post\s+to\s+(https?://[^\s<>"\'\)]+)'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        url = match.group(1).rstrip('.,;:!?)')
        print(f"[EXTRACT] Found submission URL (pattern 2): {url}")
        return url
    
    # Pattern 3: "Submit to https://..."
    pattern3 = r'submit\s+(?:to\s+)?(https?://[^\s<>"\'\)]+)'
    match = re.search(pattern3, text, re.IGNORECASE)
    if match:
        url = match.group(1).rstrip('.,;:!?)')
        print(f"[EXTRACT] Found submission URL (pattern 3): {url}")
        return url
    
    # Pattern 4: Any URL that appears after "submit" or "post" keywords
    pattern4 = r'(?:post|submit)[\s\S]{0,100}?(https?://[^\s<>"\'\)]+)'
    match = re.search(pattern4, text, re.IGNORECASE)
    if match:
        url = match.group(1).rstrip('.,;:!?)')
        print(f"[EXTRACT] Found submission URL (pattern 4): {url}")
        return url
    
    print("[EXTRACT] No submission URL found in text")
    return None

async def extract_rendered_quiz_content(page: Page) -> str:
    """
    Extracts the rendered quiz content from the DOM after JavaScript execution.
    Tries to get content from #result div first, then falls back to body text.
    """
    try:
        # Wait for JavaScript to execute and render content
        await page.wait_for_load_state("networkidle", timeout=5000)
        await asyncio.sleep(0.5)  # Give a bit more time for scripts
        
        # Try to get content from #result div (common pattern in quiz pages)
        result_div = await page.query_selector("#result")
        if result_div:
            rendered_text = await result_div.inner_text()
            if rendered_text and rendered_text.strip():
                print(f"[EXTRACT] Found quiz content in #result div ({len(rendered_text)} chars)")
                return rendered_text
        
        # Fallback: get all visible text from body
        body = await page.query_selector("body")
        if body:
            rendered_text = await body.inner_text()
            if rendered_text and rendered_text.strip():
                print(f"[EXTRACT] Extracted quiz content from body ({len(rendered_text)} chars)")
                return rendered_text
        
        # Last resort: return HTML content
        print("[EXTRACT] No rendered text found, using HTML content")
        return await page.content()
        
    except Exception as e:
        print(f"[EXTRACT] Error extracting rendered content: {e}, falling back to HTML")
        return await page.content()

async def run_single_task_loop(page: Page, task_hint: str):
    """
    This is the "Inner Loop" (Solver).
    It runs a "See-Think-Act" loop to solve a *single* task URL.
    It exits by calling "submit_answer" and returning the JSON response.
    """
    print(f"[SOLVER] 🤖 Starting new task. Hint: {task_hint[:100]}...")
    
    message_history = []
    
    for i in range(15):
        print(f"\n[SOLVER] --- Loop {i+1} / 15 ---")
        
        print("[SOLVER]  👀 Seeing (Extracting page content)...")
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
        
        # On first iteration, extract rendered quiz content; on subsequent iterations, get current page state
        if i == 0:
            page_content = await extract_rendered_quiz_content(page)
        else:
            await asyncio.sleep(0.5)
            page_content = await page.content()
        
        formatted_prompt = SYSTEM_PROMPT.format(task_hint=task_hint, html_content=page_content)
        
        if not message_history:
            message_history.append({"role": "system", "content": "You must respond with a single valid JSON tool command."})
            message_history.append({"role": "user", "content": formatted_prompt})
        else:
            message_history.append({"role": "user", "content": f"New Page Content:\n{page_content}"})

        print(f"[SOLVER]  🧠 Thinking (Calling Gemini 2.5 Pro)...")
        try:
            llm_client = get_llm_client()
            response = await llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "google/gemini-2.5-pro"),
                messages=message_history,
                response_format={"type": "json_object"}
            )
            llm_response_text = response.choices[0].message.content
            print(f"[SOLVER]  LLM response: {llm_response_text}")
            message_history.append({"role": "assistant", "content": llm_response_text})
        except Exception as e:
            print(f"[SOLVER] ❌ LLM call failed: {traceback.format_exc()}")
            message_history.append({"role": "user", "content": f"LLM Error: {e}. Please try again."})
            continue

        try:
            try:
                action_json = json.loads(llm_response_text)
            except json.JSONDecodeError:
                raise ValueError(f"LLM returned invalid JSON: {llm_response_text}")

            tool = action_json.get("tool")
            result = f"Error: Unknown tool '{tool}'."

            if tool == "click":
                result = await tool_click(page, action_json.get("selector"))
            elif tool == "fill_text":
                result = await tool_fill_text(page, action_json.get("selector"), action_json.get("text"))
            elif tool == "call_api":
                result = await tool_call_api(action_json.get("url"), action_json.get("headers"))
            elif tool == "read_file":
                result = await tool_read_file(action_json.get("url"))
            elif tool == "run_python_code":
                result = await tool_run_python_code(action_json.get("code"))
            elif tool == "take_screenshot_and_analyze":
                result = await tool_take_screenshot_and_analyze(page, action_json.get("analysis_prompt"))
            
            elif tool == "submit_answer":
                submission_payload = {
                    "email": os.environ.get("STUDENT_EMAIL", "default@email.com"),
                    "secret": os.environ.get("SECRET_KEY"),
                    "url": page.url,
                    "answer": action_json.get("answer_json", {}).get("answer")
                }
                result = await tool_submit_answer(
                    action_json.get("submission_url"), 
                    submission_payload
                )
                print(f"[SOLVER] ✅ Task submission complete.")
                return result
            else:
                result = f"Error: LLM returned an unknown tool: '{tool}'."

            print(f"[SOLVER]  Tool output: {result[:500]}...")
            message_history.append({"role": "user", "content": f"Tool Output: {result}"})
            
        except Exception as e:
            print(f"[SOLVER] ❌ Error in agent loop (ACT phase): {traceback.format_exc()}")
            message_history.append({"role": "user", "content": f"Error: {e}. Please try again."})
            
    return {"correct": False, "reason": "Solver reached max 15 loops.", "url": None}

async def solve_quiz_task(task_data: dict):
    """
    This is the "Outer Loop" (Supervisor).
    It manages the entire 180s session and task chain.
    """
    
    os.environ["STUDENT_EMAIL"] = task_data.get("email", "default@email.com")

    current_url = task_data.get("url")
    # task_hint will be extracted from the quiz page, not from request
    task_hint = None
    
    if not current_url:
        print("[SUPERVISOR] ❌ FAILED: No 'url' field in initial task_data.")
        return

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
            page = await browser.new_page(user_agent=USER_AGENT)
            
            while current_url:
                print(f"\n[SUPERVISOR] ➡️ Loading new task URL: {current_url}")
                await page.goto(current_url, wait_until="domcontentloaded", timeout=10000)
                
                # Extract task hint from the quiz page if not set
                if task_hint is None:
                    quiz_content = await extract_rendered_quiz_content(page)
                    # Use the quiz content as the task hint (first 500 chars for brevity)
                    task_hint = quiz_content[:500] if quiz_content else "Solve the task on the page."
                    print(f"[SUPERVISOR] Extracted task hint from quiz page ({len(task_hint)} chars)")

                submission_response = await run_single_task_loop(page, task_hint)
                
                print(f"[SUPERVISOR]  Submission response: {submission_response}")
                
                current_url = submission_response.get("url")
                
                if current_url:
                    print(f"[SUPERVISOR]  Chain continues. Next URL: {current_url}")
                    # Reset task_hint so it will be extracted from the new page
                    task_hint = None
                
                elif submission_response.get("correct") == True:
                    print("[SUPERVISOR] ✅ Chain complete. No new URL provided.")
                    break
                
                else:
                    print("[SUPERVISOR] ⚠️ Answer incorrect. Retrying same URL.")
                    task_hint = f"Previous attempt was wrong: {submission_response.get('reason')}. Please try again."
            
            await browser.close()
            print("[SUPERVISOR]  Browser closed. Session complete.")
            
    except Exception as e:
        print(f"[SUPERVISOR] ❌ CRITICAL FAILURE in task: {traceback.format_exc()}")
    finally:
        print(f"--------------------------------------------------")