# imports
import gradio as gr
from openai import OpenAI
from utils.modelutils import get_model,  get_llm_stream_yield_call
from utils.prompts import Prompts

def default_python_code():
    python_code = """
        import time

        def calculate(iterations, param1, param2):
            result = 1.0
            for i in range(1, iterations+1):
                j = i * param1 - param2
                result -= (1/j)
                j = i * param1 + param2
                result += (1/j)
            return result

        start_time = time.time()
        result = calculate(100_000_000, 4, 1) * 4
        end_time = time.time()

        print(f"Result: {result:.12f}")
        print(f"Execution Time: {(end_time - start_time):.6f} seconds")
        """
    return python_code

def system_prompt():
    system_message = "You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. "
    system_message += "Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. "
    system_message += "The C++ response needs to produce an identical output in the fastest possible time."

    return system_message

def user_prompt(python_code):
    user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
    user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
    user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
    user_prompt += python_code

    return user_prompt

def get_cpp_code(llm_model, model, python_code):
    if(llm_model == "openai"):
        llm_model = OpenAI()

    ai_model = get_model(llm_model)
    
    # Prepare prompts to read the links on a webpage, and respond in structured JSON.
    prompts = Prompts(system_prompt(), user_prompt(python_code))
    print("LLM Thinking...")
    print("Getting converted cpp code")

    result = get_llm_stream_yield_call(ai_model, model, prompts.get_messages())
    yield from result

def main():
    print("********** Application Started **********")

    gr.Interface(fn=get_cpp_code,
                 inputs=[gr.Dropdown(["openai", "ollama"], label="Select LLM"),
                         gr.Dropdown(["gpt-4o-mini", "llama3.2"], label="Select model"),
                         gr.Textbox(label="Enter Python code...", value=default_python_code(), lines=10),
                         ],
                 outputs=[gr.Textbox(label="Converted C++ code:", lines=10)],
                 flagging_mode='never').launch()

if __name__ == '__main__':
    main()
