# imports
import gradio as gr
from openai import OpenAI
from utils.modelutils import get_model,  get_llm_stream_yield_call
from utils.prompts import Prompts
from pathlib import Path

def get_code_text(path):
    file_path = Path(path)
    code_text = file_path.read_text(encoding='utf-8')
    return code_text

def system_prompt():
    system_message = "You are an assistant that creates docstring and code comments of python code function and classes. "
    system_message += "Include all existing and necessary imports in your response."
    system_message += "If not find any python python code, display error message 'No Python Code Found'"
    system_message += "Generate docstring and comment in the below format as an example:"
    system_message += """
    def string_reverse(str1:str) -> str :
        \"""Returns the reversed String.
        Parameters:
            str1 (str):The string which is to be reversed.
        Returns:
            reverse(str1):The string which gets reversed.   
        \"""        

        reverse_str1 = ''
        i = len(str1)
        # loop to iterate the string in reverse
        while i > 0:
            reverse_str1 += str1[i - 1]
            i = i- 1
        return reverse_str1
    """

    return system_message

def user_prompt(python_code):
    user_message = "Create docstring of the python code functions and classes."
    user_message += "Respond only with python code; do not explain your work other than a few comments. "
    user_message += "Be sure to write a description of the function purpose with typing for each argument and return\n\n"
    user_message += python_code

    return user_message

def get_docstring_code(llm_model, model, python_code, file_path):
    if(llm_model == "openai"):
        llm_model = OpenAI()

    ai_model = get_model(llm_model)
    
    if(not (file_path == "" or file_path == None)):
        python_code = get_code_text(file_path)
        print(python_code)

    # Prepare prompts to read the links on a webpage, and respond in structured JSON.
    prompts = Prompts(system_prompt(), user_prompt(python_code))
    print("LLM Thinking...")
    print("Getting docstring code")

    result = get_llm_stream_yield_call(ai_model, model, prompts.get_messages())
    yield from result

def main():
    print("********** Application Started **********")

    gr.Interface(fn=get_docstring_code,
                 inputs=[gr.Dropdown(["openai", "ollama"], label="Select LLM"),
                         gr.Dropdown(["gpt-4o-mini", "llama3.2"], label="Select model"),
                         gr.Textbox(label="Enter Python code...", lines=10),
                         gr.Textbox(label="Enter Python file path..."),
                         ],
                 outputs=[gr.Textbox(label="Generated Python code with docstring:", lines=10)],
                 flagging_mode='never').launch()

if __name__ == '__main__':
    main()
