import os
import json
import asyncio
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

llm_client = AsyncOpenAI(
    api_key=os.environ.get("AIPIPE_API_KEY"),
    base_url=os.environ.get("AIPIPE_BASE_URL"),
)

SYSTEM_PROMPT = """
You are a highly intelligent data analysis agent.
Your goal is to understand and solve a data quiz.
You will be given TWO pieces of information:
1.  A "Task Data" JSON object: This contains the URL to visit, but
    also *CRITICAL METADATA*. It may include session IDs, task IDs,
    or API keys that are *required* to solve the task.
2.  The "Page HTML" from that URL.

Analyze BOTH pieces of information and, in plain English, state:
1.  What is the task or question?
2.  Are there any hints (like API keys) in the "Task Data" JSON?
3.  What are the steps to solve it?
4.  What is the final submission URL?
"""

async def solve_quiz_task(task_data: dict):
    url = task_data.get("url")
    
    print("--------------------------------------------------")
    print(f"[WORKER] 🤖 Task accepted for URL: {url}")
    print(f"[WORKER]  Full task data: {task_data}")
    
    if not url:
        print("[WORKER] ❌ FAILED: No 'url' field in task_data.")
        return

    try:
        async with async_playwright() as p:
            print("[WORKER]  Launching browser...")
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
            page = await browser.new_page(user_agent=USER_AGENT)
            
            print(f"[WORKER]  Navigating to {url}...")
            await page.goto(url)
            await page.wait_for_load_state("networkidle", timeout=5000)
            
            print("[WORKER] 📸 Scraping page content...")
            scraped_html = await page.content()
            await browser.close()
            
            print(f"[WORKER] ✅ Successfully scraped page ({len(scraped_html)} bytes).")

        print("[WORKER] 🧠 Connecting to LLM to understand the task...")

        user_content = f"""
        --- Task Data (JSON) ---
        {json.dumps(task_data, indent=2)}

        --- Page HTML ---
        {scraped_html}
        """
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o", # MODEL
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )
        
        llm_response = response.choices[0].message.content
        
        print("[WORKER] ✅ LLM analysis complete.")
        print("\n--- LLM Response ---")
        print(llm_response)

    except Exception as e:
        print(f"[WORKER] ❌ FAILED to process task for {url}")
        print(f"[WORKER] Error: {e}")
    
    finally:
        print("--------------------------------------------------")