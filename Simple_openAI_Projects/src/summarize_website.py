# imports

import os
import sys
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI
from utils.modelutils import get_model, display_response
from utils.prompts import Prompts
from utils.web_scraping_beautifulsoup import Website


def main(args):
    print("********** Application Started **********")
    print(args)

    llm_model = args[0]
    model = args[1]
    website = args[2]

    if(args[0] == "openai"):
        llm_model = OpenAI()

    # llm_model = OpenAI()
    # model = "gpt-4o-mini"

    # llm_model = "ollama"
    # model = "llama3.2"

    ai_model = get_model(llm_model)

    system_prompt = """You are an assistant that analyzes the contents of a website \
    and provides a short summary, ignoring text that might be navigation related. \
    Respond in markdown."""

    # Scraping website
    print("website scraping Started...")
    website = Website(website)
    print("website scraping Ended...")

    user_prompt = f"You are looking at a website titled {website.title} "

    prompts = Prompts(system_prompt, user_prompt)
    prompts.user_prompt_add(f"""\nThe contents of this website is as follows; \
    please provide a short summary of this website in markdown. \
    If it includes news or announcements, then summarize these too.\n\n
    {website.text}
    """)

    print("LLM Thinking...")
    print("Getting summary and display in markdown...")
    # display the website summary using markdown
    display_response(llm_model=ai_model, model=model, messages=prompts.get_messages())

if __name__ == '__main__':
    main(sys.argv[1:])
