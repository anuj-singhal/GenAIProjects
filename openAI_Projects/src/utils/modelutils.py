from openai import OpenAI
import os
import requests
from dotenv import load_dotenv
from IPython.display import Markdown, display
import json

load_dotenv(override=True)

def check_openAI_key(api_key: str) -> bool:
    """Checks the validity of the provided OpenAI API key.
    
    Parameters:
        api_key (str): The OpenAI API key to check.
        
    Returns:
        bool: True if the key is valid, False otherwise.
    """
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

def get_openAI_key() -> str or bool:
    """Retrieves OpenAI API key from environment variables and validates it.
    
    Returns:
        str or bool: The valid API key if found, False otherwise.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not check_openAI_key(api_key):
        return False
    return api_key

def get_openAI_model() -> OpenAI or None:
    """Initializes and returns an instance of the OpenAI model.
    
    Returns:
        OpenAI or None: The OpenAI model instance if the API key is valid, None otherwise.
    """
    print("You are using OpenAI model..")
    api_key = get_openAI_key()
    if api_key:
        ai_model = OpenAI()
        return ai_model
    return None

def get_olamma_model(base_url: str = "http://localhost:11434/v1", api_key: str = "ollama") -> OpenAI:
    """Creates and returns an instance of the local Olamma model.
    
    Parameters:
        base_url (str): The base URL for the Olamma model.
        api_key (str): The API key for Olamma.
        
    Returns:
        OpenAI: The Olamma model instance.
    """
    print("You are using local Olamma model..")
    ai_model = OpenAI(base_url=base_url, api_key=api_key)
    return ai_model

def get_model(model: str, base_url: str = None, api_key: str = None) -> OpenAI or None:
    """Retrieves the desired AI model based on the input identifier.
    
    Parameters:
        model (str): The model type to retrieve, either "ollama" or OpenAI instance.
        base_url (str): Optional; base URL for the Olamma model if applicable.
        api_key (str): Optional; API key for the Olamma model if applicable.
        
    Returns:
        OpenAI or None: The initialized model instance or None if not found.
    """
    ai_model = None
    if isinstance(model, OpenAI):
        ai_model = get_openAI_model()
    
    if model == "ollama":
        if base_url is not None and api_key is not None:
            ai_model = get_olamma_model(base_url, api_key)
        else:
            ai_model = get_olamma_model()
    return ai_model

def get_llm_stream_call(llm_model, model: str, messages: list) -> str:
    """Retrieves a streamed response from the language model based on given messages.
    
    Parameters:
        llm_model: The language model instance to use for the call.
        model (str): The specific model being queried.
        messages (list): A list of messages forming the conversation context.
        
    Returns:
        str: The accumulated response from the model stream.
    """
    stream = llm_model.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    
    response = ""
    # Iterate over the streamed response chunks
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
    
    return response

def get_llm_stream_yield_call(llm_model, model: str, messages: list):
    """Yields responses from the language model in a streamed manner based on messages.
    
    Parameters:
        llm_model: The language model instance to use for the call.
        model (str): The specific model being queried.
        messages (list): A list of messages forming the conversation context.
        
    Yields:
        str: The accumulated response from the model stream.
    """
    stream = llm_model.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    
    response = ""
    # Stream responses while yielding partial results
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

def get_llm_call(llm_model, model: str, messages: list) -> str:
    """Retrieves a single response from the language model based on given messages.
    
    Parameters:
        llm_model: The language model instance to use for the call.
        model (str): The specific model being queried.
        messages (list): A list of messages forming the conversation context.
        
    Returns:
        str: The complete response from the model.
    """
    response = llm_model.chat.completions.create(model=model, 
                                                 messages=messages,
                                                 )
    result = response.choices[0].message.content
    return result

def get_llm_json_call(llm_model, model: str, messages: list) -> dict:
    """Retrieves a JSON response from the language model based on given messages.
    
    Parameters:
        llm_model: The language model instance to use for the call.
        model (str): The specific model being queried.
        messages (list): A list of messages forming the conversation context.
        
    Returns:
        dict: The parsed JSON response from the model.
    """
    response = llm_model.chat.completions.create(model=model, 
                                                 messages=messages,
                                                 response_format={"type": "json_object"}
                                                 )
    result = response.choices[0].message.content
    return json.loads(result)