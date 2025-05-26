# imports

import os
import sys
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI
from utils.modelutils import get_model, get_llm_call, get_llm_json_call, get_llm_stream_call
from utils.prompts import Prompts
from utils.web_scraping_beautifulsoup import Website

def system_prompt_for_link():
    system_prompt = """You are an assistant help to build the company brochure. \
        You are provided the list of links found on a webpage. \
        You are able to decide which link is relevent to include in the company brochure. \
        Such as such as links to an About page, or a Company page, or Careers/Jobs pages.\n
        """
    system_prompt += "You should respond in JSON as in these examples:"
    system_prompt += """
    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page": "url": "https://another.full.url/careers"}
        ]
    }
    """
    system_prompt += """
    {
        "links": [
            {'type': 'company page', 'url': 'https://domain.com/companyname'},
            {"type": "blog page": "url": "https://domainname.co/blog"}
        ]
    }
    """
    return system_prompt

def user_prompt_for_link(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company,\
                    respond with the full https URL in JSON format. \
                    Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)

    return user_prompt

def system_prompt_for_brochure():
    system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
                    and creates a short brochure about the company for prospective customers, \
                    investors and recruits make sure of below pints. \n "
    
    system_prompt += "Create a creative and attractive content that will be little funny in professional way"
                    
    system_prompt += "Respond in markdown. \n\
                Include details of company culture, customers and careers/jobs if you have the information."

    return system_prompt

def user_prompt_for_brochure(company_name, company_details, no_truncate_char=5000):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; \
        use this information to build a short brochure of the company in markdown.\n"
    user_prompt += company_details
    user_prompt = user_prompt[:no_truncate_char] # Truncate if more than 5,000 characters

    return user_prompt

def system_prompt_for_translation():
    system_prompt = f"You are a translator that can translate contents from one language to another language"
    return system_prompt

def user_prompt_for_translation(language_from, language_to, company_brochure):
    user_prompt = f"You have the company brochure {company_brochure} which is in {language_from} language: \n \
        Now convert it into {language_to} language. \
            make sure the output is in markdown and the structure is same."

    return user_prompt

def get_all_company_details(website, relevant_links):
    result = "Landing Page:\n"
    result += website.get_contents()
    links = relevant_links
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link['url']).get_contents()
    return result

def main(args):
    print("********** Application Started **********")
    print(args)

    llm_model = args[0] # "ollama/openai"
    model = args[1] # "llama3.2/gpt-4o-mini"
    website = args[2] # "https://huggingface.co"
    company_name = args[3] # HuggingFace
    translate_to = "Spanish"

    if(args[0] == "openai"):
        llm_model = OpenAI()

    ai_model = get_model(llm_model)

    # Scraping website
    print("website scraping Started...")
    website = Website(website)
    print("website scraping Ended...")
    
    # Prepare prompts to read the links on a webpage, and respond in structured JSON.
    link_prompts = Prompts(system_prompt_for_link(), user_prompt_for_link(website))

    print("LLM Thinking...")
    print("Getting relevant links from the website")

    # get the relevant links from the website from LLM
    relevant_links = get_llm_json_call(ai_model, model, link_prompts.get_messages())
    #print(relevant_links)
    company_details = get_all_company_details(website=website, relevant_links=relevant_links)

    #print(company_details)

    truncate_char = 5000
    brochure_prompts = Prompts(system_prompt_for_brochure(), user_prompt_for_brochure(company_name, company_details, truncate_char))
    
    print("LLM Thinking...")
    print("Creating Company brochure from LLM")
    llm_company_brochure = get_llm_call(ai_model, model, brochure_prompts.get_messages())

    print(llm_company_brochure)

    translate_prompts = Prompts(system_prompt_for_translation(), user_prompt_for_translation("English",translate_to,llm_company_brochure))

    print("LLM Thinking...")
    print(f"Creating Company brochure from English to {translate_to}")
    llm_translated_brochure = get_llm_stream_call(ai_model, model, translate_prompts.get_messages())

    for chunk in llm_translated_brochure:
        print(chunk)

if __name__ == '__main__':
    main(sys.argv[1:])
