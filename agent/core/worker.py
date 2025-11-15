import os
import json
import asyncio
import re
from playwright.async_api import async_playwright, Page
from openai import AsyncOpenAI
from agent.core.tools import *

llm_client = AsyncOpenAI(
    api_key=os.environ.get("AIPIPE_API_KEY"),
    base_url=os.environ.get("AIPIPE_BASE_URL"),
)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

def clean_html_for_router(html: str) -> str:
    html = re.sub(r'<(script|style).*?>.*?</\1>', '', html, flags=re.DOTALL)
    html = re.sub(r'<svg.*?>.*?</svg>', '', html, flags=re.DOTALL)
    html = re.sub(r'', '', html, flags=re.DOTALL)
    html = re.sub(r'<.*?>', ' ', html)
    html = re.sub(r'\s+', ' ', html).strip()
    return html

ROUTER_PROMPT_TEMPLATE = """
You are an expert task router. Your job is to analyze a task and some
cleaned webpage text, then assign the *correct* specialist.
Respond with *one word only*: SIMPLE, CODE, or PRO.

- **CODE**: Use for tasks involving math, data, tables, PDFs, APIs, or Python.
- **PRO**: Use for tasks involving complex visual charts, images, or very hard reasoning.
- **SIMPLE**: Use for basic web clicking, text filling, and simple text scraping.

---
TASK DATA:
{task_data}

---
CLEANED PAGE TEXT (first 5000 chars):
{cleaned_html_snippet}
"""

SIMPLE_AGENT_PROMPT = """
You are the SIMPLE_AGENT, a web navigator.
Your job is to solve the task using *only* clicking and typing.
You will be given the HTML. You must respond with a single JSON command.
The user's task is: {task_hint}

Your available tools are:
1. {"tool": "click", "selector": "<css_selector>"}
2. {"tool": "fill_text", "selector": "<css_selector>", "text": "<text_to_fill>"}
3. {"tool": "submit_answer", "submission_url": "<url>", "answer_json": {"answer": <value>}}
   - Use this *only* when you have the final answer.

The current page HTML is:
{html_content}
"""

CODE_AGENT_PROMPT = """
You are the CODE_AGENT, a data science specialist.
Your job is to solve a task using data tools.
The user's task is: {task_hint}

Your available tools are:
1. {"tool": "read_file", "url": "<file_url>"}
2. {"tool": "call_api", "url": "<api_url>", "headers": {}}
3. {"tool": "run_python_code", "code": "<python_code_snippet>"}
   - You MUST use this for any math, data parsing (CSV/JSON), or analysis.
   - You have `pandas as pd`, `numpy as np`, and `io` available.
   - Data from tools will be strings. Use `pd.read_csv(io.StringIO(data_string))` for CSVs.
   - You MUST `print()` your final answer to get the output.
4. {"tool": "submit_answer", "submission_url": "<url>", "answer_json": {"answer": <value>}}
   - Use this *only* when you have the final answer.

The current page HTML (for context) is:
{html_content}
"""

PRO_AGENT_PROMPT = """
You are the PRO_AGENT, the most advanced reasoning agent.
Your job is to solve complex visual or logic puzzles.
The user's task is: {task_hint}

You have ALL tools available:
1. {"tool": "click", "selector": "<css_selector>"}
2. {"tool": "fill_text", "selector": "<css_selector>", "text": "<text_to_fill>"}
3. {"tool": "read_file", "url": "<file_url>"}
4. {"tool": "call_api", "url": "<api_url>", "headers": {}}
5. {"tool": "run_python_code", "code": "<python_code_snippet>"}
6. {"tool": "take_screenshot_and_analyze", "analysis_prompt": "<what_to_look_for>"}
   - Use this if the HTML is confusing or the task is visual (e.g., "What color is the bar chart?").
7. {"tool": "submit_answer", "submission_url": "<url>", "answer_json": {"answer": <value>}}

The current page HTML is:
{html_content}
"""

async def run_agent_loop(agent_name: str, model_name: str, system_prompt: str,
                         page: Page, task_data: dict, initial_html: str):
    """
    A generic "See-Think-Act" loop for a specialist agent.
    """
    print(f"[AGENT] 🤖 {agent_name} activated. Model: {model_name}")
    
    task_hint = task_data.get("task_hint", "Solve the task.")
    
    formatted_prompt = system_prompt.format(task_hint=task_hint, html_content=initial_html)
    message_history = [
        {"role": "system", "content": "You must respond with a single valid JSON tool command."},
        {"role": "user", "content": formatted_prompt}
    ]
    
    for i in range(10):
        print(f"\n[{agent_name}] --- Loop {i+1} / 10 ---")
        
        print(f"[{agent_name}]  🧠 Thinking (Calling {model_name})...")
        try:
            response = await llm_client.chat.completions.create(
                model=model_name,
                messages=message_history,
                response_format={"type": "json_object"}
            )
            llm_response_text = response.choices[0].message.content
            print(f"[{agent_name}]  LLM response: {llm_response_text}")
            message_history.append({"role": "assistant", "content": llm_response_text})
        except Exception as e:
            print(f"[{agent_name}] ❌ LLM call failed: {e}")
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
                print(f"[{agent_name}] ✅ Task Complete!")
                return result
            else:
                result = f"Error: LLM returned an unknown tool: '{tool}'."

            print(f"[{agent_name}]  Tool output: {result[:500]}...")
            await page.wait_for_load_state("networkidle", timeout=3000)
            new_html = await page.content()
            
            message_history.append({"role": "user", "content": f"Tool Output: {result}\n\nNew Page HTML:\n{new_html}"})

        except Exception as e:
            print(f"[{agent_name}] ❌ Error in loop: {e}")
            message_history.append({"role": "user", "content": f"Error: {e}. Please try again."})
            
    return f"{agent_name} finished (max loops reached)."

async def decide_specialist(task_data, raw_html: str):
    print("[ROUTER] 🧠 Deciding which specialist to use...")
    print("[ROUTER]  Cleaning HTML for analysis...")
    cleaned_html = clean_html_for_router(raw_html)
    
    prompt = ROUTER_PROMPT_TEMPLATE.format(
        task_data=json.dumps(task_data, indent=2),
        cleaned_html_snippet=cleaned_html[:5000]
    )
    
    try:
        response = await llm_client.chat.completions.create(
            model="openai/gpt-5-image-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        choice = response.choices[0].message.content.strip().upper()
        print(f"[ROUTER]  LLM choice: {choice}")
        
        if "CODE" in choice: return "CODE"
        if "PRO" in choice: return "PRO"
        return "SIMPLE"
        
    except Exception as e:
        print(f"[ROUTER] ❌ Error in router LLM, defaulting to CODE (safer): {e}")
        return "CODE"

async def solve_quiz_task(task_data: dict):
    url = task_data.get("url")
    if not url:
        print("[AGENT] ❌ FAILED: No 'url' field in task_data.")
        return

    print(f"[AGENT] 🤖 New task accepted for URL: {url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
            page = await browser.new_page(user_agent=USER_AGENT)
            
            print(f"[AGENT]  Navigating to {url}...")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            html_content = await page.content()
            print("[AGENT]  Page content scraped.")
            
            specialist = await decide_specialist(task_data, html_content)
            print(f"[ROUTER]  Decision: Routing to {specialist}_AGENT.")
            
            if specialist == "CODE":
                result = await run_agent_loop(
                    "CODE_AGENT", "openai/gpt-5-pro",
                    CODE_AGENT_PROMPT, page, task_data, html_content
                )
            elif specialist == "PRO":
                result = await run_agent_loop(
                    "PRO_AGENT", "openai/gpt-5-pro",
                    PRO_AGENT_PROMPT, page, task_data, html_content
                )
            else:
                result = await run_agent_loop(
                    "SIMPLE_AGENT", "openai/gpt-5-image-mini",
                    SIMPLE_AGENT_PROMPT, page, task_data, html_content
                )
            
            print(f"[AGENT]  Specialist finished with result: {result}")
            await browser.close()
            
    except Exception as e:
        print(f"[AGENT] ❌ CRITICAL FAILURE in task: {e}")
    finally:
        print(f"--------------------------------------------------")