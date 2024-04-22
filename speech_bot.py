import os

import speech_recognition as sr
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from prompts import system_bot
from groq import Groq
import elevenlabs
from elevenlabs.client import ElevenLabs

env_path = find_dotenv()
load_dotenv(env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
client_open_ai = OpenAI(api_key=OPENAI_API_KEY)


def speech_recognize():
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone(device_index=0, sample_rate=44100) as source:
        print("Say something!")
        audio = r.listen(source)

    # recognize speech using Whisper API
    try:
        text_recognized = r.recognize_whisper_api(audio, api_key=OPENAI_API_KEY)
        print(text_recognized)
        return text_recognized
    except sr.RequestError as e:
        print(f"Could not request results from Whisper API; {e}")
        return None


def tts_output(text_recognized):
    eleven_client = ElevenLabs(
        api_key= ELEVEN_API_KEY,
    )
    audio = eleven_client.generate(
        text=text_recognized,
        voice="Lucien",
        model="eleven_multilingual_v2",
        optimize_streaming_latency=4
    )
    elevenlabs.stream(audio)

def get_completion(model='gpt-3.5-turbo-0125'):
    """
    :param model: The model to use for chat completion. Default is 'llama3-8b-8192'.
    :return: None

    This method conducts a chat conversation with the GROQ AI LLaMA3 8B META model. It prompts the user for input, sends
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
        message = speech_recognize()

        # Exit the program if the user inputs "quits"
        if message.lower() == 'Bye':
            break

        # Add each new message to the list of messages
        messages.append({'role': 'user', 'content': message})

        response = client_open_ai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        chat_message = response.choices[0].message.content
        print(f"Bot: {chat_message}")
        tts_output(chat_message)
        messages.append({'role': 'assistant', 'content': chat_message})


