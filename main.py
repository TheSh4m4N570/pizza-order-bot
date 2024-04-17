import os
import openai
from prompts import system_bot
from openai import OpenAI
from dotenv import load_dotenv

# Load env and OPEN AI KEY
load_dotenv()
API_KEY = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=API_KEY)


# define a simple function to test OPEN AI API

def get_completion(model='gpt-3.5-turbo-0125'):
    """
    :param model: The model to use for chat completion. Default is 'gpt-3.5-turbo-0125'.
    :return: None

    This method conducts a chat conversation with the OpenAI GPT-3.5 Turbo model. It prompts the user for input, sends
    it to the model, and prints the model's response. The conversation continues until the user inputs "quit".

    Example usage:

    ```
    get_completion()
    ```
    """
    messages = [
        {'role': 'system', 'content': system_bot},
    ]
    while True:
        # prompt user for input:
        message = input("User: ")

        # Exit the program if the user inputs "quits"
        if message.lower() == 'quit':
            break

        # Add each new message to the list of messages
        messages.append({'role': 'user', 'content': message})

        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        chat_message = response.choices[0].message.content
        print(f"Bot: {chat_message}")
        messages.append({'role': 'assistant', 'content': chat_message})


if __name__ == '__main__':
    get_completion()
