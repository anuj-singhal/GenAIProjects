# imports
import gradio as gr
from openai import OpenAI
from utils.modelutils import get_model,  get_llm_stream_yield_call
from utils.prompts import Prompts

def system_prompt():
    system_message = "You are an assistant that helps to solve assessment of staff data engineer role for emnify company"
    #system_message = "You are an assistant that helps to get the assessment questions of staff data engineer role for emnify company"


    return system_message

def user_prompt(question):
    user_prompt = "The assessment questions cover a range of topics including data architecture, " \
    "big data technologies, data pipelines, cloud services, and problem-solving capabilities to ensure " \
    "they're well-aligned with the responsibilities and challenges that might be faced in such a role"
    user_prompt += "In question it might ask for examples, provide example as you are working in retail bank wealth management project"
    user_prompt += "Respond in human form as much you can"
    user_prompt = "Pay attention to the below questions, and answer deeply as per the rols and provide your output\n\n"
    user_prompt += "Question: " + question

    return user_prompt

def get_cpp_code(llm_model, model, question):
    if(llm_model == "openai"):
        llm_model = OpenAI()

    ai_model = get_model(llm_model)
    
    # Prepare prompts to read the links on a webpage, and respond in structured JSON.
    prompts = Prompts(system_prompt(), user_prompt(question))
    print("LLM Thinking...")
    print("Getting Answer")

    result = get_llm_stream_yield_call(ai_model, model, prompts.get_messages())
    yield from result

def main():
    print("********** Application Started **********")

    gr.Interface(fn=get_cpp_code,
                 inputs=[gr.Dropdown(["openai", "ollama"], label="Select LLM"),
                         gr.Dropdown(["gpt-4o-mini", "llama3.2"], label="Select model"),
                         gr.Textbox(label="Enter Python code...", lines=10),
                         ],
                 outputs=[gr.Textbox(label="Answer by openai :", lines=10)],
                 flagging_mode='never').launch()

if __name__ == '__main__':
    main()
