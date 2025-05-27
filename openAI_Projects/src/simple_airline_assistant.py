from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

def main():
    openai = OpenAI()
    model = "gpt-4o-mini"
    system_message = "You are a helpful assistant for an Airline called FlightAI. "
    system_message += "Give short, courteous answers, no more than 1 sentence. "
    system_message += "Always be accurate. If you don't know the answer, say so."
    
    def chat(message, history):
        messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

        response = openai.chat.completions.create(model=model, 
                                               messages=messages, 
                                               )
        return response.choices[0].message.content

    gr.ChatInterface(fn=chat, type="messages").launch()

if __name__ == "__main__":
    main()