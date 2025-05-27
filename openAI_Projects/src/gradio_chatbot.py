from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

def main():
    openai = OpenAI()
    model = "gpt-4o-mini"
    system_message = "You are an helpful assistant that can help to solve mathmatics problems"
    def chat(message, history):
        messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

        steam = openai.chat.completions.create(model=model, 
                                               messages=messages, 
                                               stream=True)
        
        response = ""

        for chunk in steam:
            response += chunk.choices[0].delta.content or ''
            yield response

    gr.ChatInterface(fn=chat, type="messages").launch()

if __name__ == "__main__":
    main()