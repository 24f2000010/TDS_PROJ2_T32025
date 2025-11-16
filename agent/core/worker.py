import os
import json
import asyncio
import re
import traceback
from playwright.async_api import async_playwright, Page
from openai import AsyncOpenAI
from agent.core.tools import *

llm_client = AsyncOpenAI(
    api_key=os.environ.get("AIPIPE_API_KEY"),
    base_url=os.environ.get("AIPIPE_BASE_URL"),
)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

SYSTEM_PROMPT = """
You are a "Generalist Agent," a highly intelligent AI controlling a web browser to solve a complex data science task.
Your goal is to solve the given task by navigating pages, reading files, analyzing data, and submitting the final answer.
You MUST operate in a step-by-step "See-Think-Act" loop.

**TASK:**
{task_hint}

**AVAILABLE TOOLS:**
You MUST respond with a single valid JSON object describing the *one* tool you want to use next.

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
       You have `pandas as pd`, `numpy as np`, and `io` available.
       Data from other tools is a string. Use `pd.read_csv(io.StringIO(data_string))`.
       You MUST `print()` your final answer to get the output.)

4.  **Vision Analysis:**
    {{"tool": "take_screenshot_and_analyze", "analysis_prompt": "<what_to_look_for>"}}
       (Use this if the HTML is confusing or the task is visual, like a chart or image.)

5.  **Final Submission:**
    {{"tool": "submit_answer", "submission_url": "<url>", "answer_json": {{"answer": <value>}} }}
       (Use this *only* when you have the final answer and are ready to end the task.)

**CURRENT PAGE HTML:**
{html_content}
"""

async def solve_quiz_task(task_data: dict):
    """
    This is the main "Generalist Agent" entrypoint.
    It contains the full "See-Think-Act" loop.
    """
    url = task_data.get("url")
    task_hint = task_data.get("task_hint", "Solve the task on the page.")
    
    if not url:
        print("[AGENT] ❌ FAILED: No 'url' field in task_data.")
        return

    print(f"[AGENT] 🤖 New Generalist Agent task accepted for URL: {url}")
    print(f"[AGENT]  Task Hint: {task_hint}")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
            page = await browser.new_page(user_agent=USER_AGENT)
            
            print(f"[AGENT]  Navigating to {url}...")
            await page.goto(url, wait_until="domcontentloaded", timeout=10000)
            await asyncio.sleep(0.5)
            
            message_history = []

            for i in range(3):
                print(f"\n[AGENT] --- Loop {i+1} / 15 ---")
                
                print("[AGENT]  👀 Seeing (Scraping page)...")
                html_content = await page.content()
                
                formatted_prompt = SYSTEM_PROMPT.format(task_hint=task_hint, html_content=html_content)
                
                if not message_history:
                    message_history.append({"role": "system", "content": "You must respond with a single valid JSON tool command."})
                    message_history.append({"role": "user", "content": formatted_prompt})
                else:
                    message_history.append({"role": "user", "content": f"New Page HTML:\n{html_content}"})

                print(f"[AGENT]  🧠 Thinking (Calling Gemini 2.5 Pro)...")
                try:
                    response = await llm_client.chat.completions.create(
                        model="google/gemini-2.5-pro",
                        messages=message_history,
                        response_format={"type": "json_object"}
                    )
                    llm_response_text = response.choices[0].message.content
                    print(f"[AGENT]  LLM response: {llm_response_text}")
                    message_history.append({"role": "assistant", "content": llm_response_text})
                except Exception as e:
                    print(f"[AGENT] ❌ LLM call failed: {traceback.format_exc()}")
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
                        result = await tool_submit_answer(action_json.get("submission_url"), action_json.get("answer_json"))
                        print(f"[AGENT] ✅ Task Complete!")
                        break
                    else:
                        result = f"Error: LLM returned an unknown tool: '{tool}'."

                    print(f"[AGENT]  Tool output: {result[:500]}...")
                    message_history.append({"role": "user", "content": f"Tool Output: {result}"})
                    
                    await asyncio.sleep(0.5)

                except Exception as e:
                    print(f"[AGENT] ❌ Error in agent loop (ACT phase): {traceback.format_exc()}")
                    message_history.append({"role": "user", "content": f"Error: {e}. Please try again."})
            
            await browser.close()
            print("[AGENT]  Browser closed.")
            
    except Exception as e:
        print(f"[AGENT] ❌ CRITICAL FAILURE in task: {traceback.format_exc()}")
    finally:
        print(f"--------------------------------------------------")