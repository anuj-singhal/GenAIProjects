from openai import OpenAI
import os
import requests
from dotenv import load_dotenv
from IPython.display import Markdown, display
import json

load_dotenv(override=True)

def check_openAI_key(api_key):
    # Check the key
    if not api_key:
        print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
        return False
    elif not api_key.startswith("sk-proj-"):
        print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
        return False
    elif api_key.strip() != api_key:
        print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
        return False
    else:
        print("API key found and looks good so far!")
    return True

def get_openAI_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if(not check_openAI_key(api_key)):
        return False
    return api_key

def get_openAI_model():
    print("You are using OpenAI model..")
    api_key = get_openAI_key()
    if(api_key):
        ai_model = OpenAI()
        return ai_model
    return None

def get_olamma_model(base_url="http://localhost:11434/v1", api_key="ollama"):
    print("You are using local Olamma model..")
    ai_model = OpenAI(base_url=base_url, api_key = api_key)
    return ai_model

def get_model(model, base_url=None, api_key=None):
    ai_model = None
    if(isinstance(model, OpenAI)):
        ai_model = get_openAI_model()
    
    if(model == "ollama"):
        if((base_url != None )& (api_key != None)):
            ai_model = get_olamma_model(base_url, api_key)
        else:
            ai_model = get_olamma_model()
    return ai_model

def get_llm_stream_call(llm_model, model, messages):
    stream = llm_model.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
    
    return response

def get_llm_stream_yield_call(llm_model, model, messages):
    stream = llm_model.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

def get_llm_call(llm_model, model, messages):
    response = llm_model.chat.completions.create(model=model, 
                                                 messages=messages,
                                                 )
    result = response.choices[0].message.content
    return result

def get_llm_json_call(llm_model, model, messages):
    response = llm_model.chat.completions.create(model=model, 
                                                 messages=messages,
                                                 response_format={"type": "json_object"}
                                                 )
    result = response.choices[0].message.content
    return json.loads(result)