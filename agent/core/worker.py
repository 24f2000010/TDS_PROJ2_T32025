import os
import json
import asyncio
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from agent.core.tools import *

llm_client = AsyncOpenAI(
    api_key=os.environ.get("AIPIPE_API_KEY"),
    base_url=os.environ.get("AIPIPE_BASE_URL"),
)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ... Chrome/119.0.0.0 Safari/537.36"

ROUTER_PROMPT_TEMPLATE = """
You are a task router. Given this task and webpage, which specialist is needed?
Respond with *one word only*: SIMPLE, CODE, or PRO.
- SIMPLE: For basic web clicking/typing.
- CODE: For math, data, PDF, API, or Python.
- PRO: For complex visual analysis (charts, images) or very hard reasoning.

Task:
{task_data}

Page Content (first 1000 chars):
{html_snippet}
"""

SIMPLE_AGENT_PROMPT = "You are a simple web agent. Your tools are click and fill_text..."
CODE_AGENT_PROMPT = "You are a data science agent. Your tools are call_api, read_file, run_python_code..."
PRO_AGENT_PROMPT = "You are a pro-level reasoning agent. You have all tools..."

async def run_simple_agent(page, task_data, message_history):
    print("[AGENT] 🤖 SIMPLE_AGENT activated.")
    await asyncio.sleep(1)
    return "SIMPLE_AGENT finished."

async def run_code_agent(page, task_data, message_history):
    print("[AGENT] 💻 CODE_AGENT activated.")
    await asyncio.sleep(1)
    return "CODE_AGENT finished."

async def run_pro_agent(page, task_data, message_history):
    print("[AGENT] 👑 PRO_AGENT activated.")
    await asyncio.sleep(1)
    return "PRO_AGENT finished."

async def decide_specialist(task_data, html_snippet):
    print("[ROUTER] 🧠 Deciding which specialist to use...")
    
    prompt = ROUTER_PROMPT_TEMPLATE.format(
        task_data=json.dumps(task_data, indent=2),
        html_snippet=html_snippet[:1000]
    )
    
    try:
        response = await llm_client.chat.completions.create(
            model="openai/gpt-5-image-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        choice = response.choices[0].message.content.strip().upper()
        
        if "CODE" in choice:
            return "CODE"
        if "PRO" in choice:
            return "PRO"
        
        return "SIMPLE"
        
    except Exception as e:
        print(f"[ROUTER] ❌ Error in router LLM, defaulting to SIMPLE: {e}")
        return "SIMPLE"

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
            
            message_history = [{"role": "user", "content": html_content}]
            if specialist == "CODE":
                result = await run_code_agent(page, task_data, message_history)
            elif specialist == "PRO":
                result = await run_pro_agent(page, task_data, message_history)
            else:
                result = await run_simple_agent(page, task_data, message_history)
            
            print(f"[AGENT]  Specialist finished with result: {result}")
            await browser.close()
            
    except Exception as e:
        print(f"[AGENT] ❌ CRITICAL FAILURE in task: {e}")
    finally:
        print(f"--------------------------------------------------")