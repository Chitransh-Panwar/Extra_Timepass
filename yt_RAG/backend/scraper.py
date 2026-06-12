import asyncio
from playwright.async_api import async_playwright
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
api = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model="openrouter/owl-alpha",
    api_key=api,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.5
)

async def scrape_page(url: str) -> str:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            text = await page.inner_text("body")
            await browser.close()
            return text.strip()
    except Exception as e:
        return f"Error scraping the webpage: {str(e)}"

def scrape_page_basic(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""

async def ask_webpage(url: str, question: str) -> str:
    page_text = await scrape_page(url)
    
    if len(page_text) < 100:
        page_text = scrape_page_basic(url)
        
    if len(page_text) < 100 or page_text.startswith("Error"):
        return "Could not access this page. Try copying the text manually."

    trimmed_text = page_text[:8000]

    prompt = PromptTemplate(
        template="""
        you are a helpful assistant.
        Answer only from the provided webpage context.if the context is insufficient ,just say you dont know .
        {context}
        question:{question}
        """,
        input_variables=['context', 'question']
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser

    try:
        response_text = await chain.ainvoke({"context": trimmed_text, "question": question})
        return response_text
    except Exception as e:
        return f"Error executing pipeline generation: {str(e)}"