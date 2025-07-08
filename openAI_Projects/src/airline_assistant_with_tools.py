from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
import json

load_dotenv(override=True)

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city: str) -> str:
    """Returns the price of a ticket to the specified destination city.
    
    Parameters:
        destination_city (str): The city for which the ticket price is requested.
    
    Returns:
        str: The price of the ticket if the city is known; otherwise, "Unknown".
    """
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

def handle_tool_call(message) -> tuple:
    """Handles the tool call and extracts necessary information.
    
    Parameters:
        message: The message object containing tool call data.
        
    Returns:
        tuple: A tuple containing the response dictionary and the destination city.
    """
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city, "price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

def main() -> None:
    """Main function to initialize the OpenAI chat interface and run the assistant."""
    openai = OpenAI()
    model = "gpt-4o-mini"
    system_message = "You are a helpful assistant for an Airline called FlightAI. "
    system_message += "Give short, courteous answers, no more than 1 sentence. "
    system_message += "Always be accurate. If you don't know the answer, say so."
    
    tools = [{"type": "function", "function": price_function}]

    def chat(message: str, history: list) -> str:
        """Processes user messages and interacts with the OpenAI API to get responses.
        
        Parameters:
            message (str): The user's message.
            history (list): The history of messages in the conversation.
        
        Returns:
            str: The content of the response message from the AI.
        """
        messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
        response = openai.chat.completions.create(model=model, messages=messages, tools=tools)
        print(response)
        if response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            response, city = handle_tool_call(message)
            messages.append(message)
            messages.append(response)
            response = openai.chat.completions.create(model=model, messages=messages)
        
        return response.choices[0].message.content

    gr.ChatInterface(fn=chat, type="messages").launch()

if __name__ == "__main__":
    main()