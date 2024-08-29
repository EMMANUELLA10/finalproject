import pyttsx3
import datetime
import requests
#import google.generativeai as genai
#import spacy
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import concurrent.futures
import logging
from pydub import AudioSegment
import pyrebase
import os
import openai
import logging
import concurrent.futures
import re

#os.environ['GROQ_API_KEY'] = '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'

config = {
  "apiKey": "AIzaSyAovENLKzkuLVFvZatgCmWp8vTPRwfu9wQ",
  "authDomain": "twigpt-b33cf.firebaseapp.com",
  "databaseURL": "https://twigpt-b33cf-default-rtdb.firebaseio.com",
  "projectId": "twigpt-b33cf",
  "storageBucket": "twigpt-b33cf.appspot.com",
  "messagingSenderId": "1036309995365",
  "appId": "1:1036309995365:web:0fb00a0bc30df2b96c0b8c"
}

firebase = pyrebase.initialize_app(config)


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Spacy
#nlp = spacy.load("en_core_web_sm")

# Initialize pyttsx3 engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Female voice
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 50)

#Initialize the generative model for conversational AI
# model = genai.GenerativeModel('gemini-pro')
# talk = []
#from groq import Groq


# List to store old queries
old_queries = []

# Function to speak out text
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Function to translate text using Google Translate API
def translate_text(input_text, source_lang, target_lang):
    url = "https://translate.googleapis.com/translate_a/single"
    placeholder = "<COLON>"
    input_text = input_text.replace(':', placeholder)
    params = {
        "client": "gtx",
        "sl": source_lang,
        "tl": target_lang,
        "dt": "t",
        "q": input_text
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        translation = response.json()[0][0][0]
        translation = translation.replace(placeholder, ':')
        return translation
    except requests.RequestException as e:
        logging.error(f"Translation error: {e}")
        return None

# Function to handle conversational AI commands
from openai import OpenAI
# Initialize the OpenAI client with your API key
#openai.api_key = "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"
client = OpenAI(api_key="")

# Initialize context to keep track of the conversation
talk = []

def handle_conversational_ai_command(query):
    global talk
    
    # Add the user's query to the conversation history
    # talk.append({'role': 'user', 'content': query})
    
    # # Prepare the messages for the API request
    # messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    # messages.extend(talk)
    
    try:
        # Call the OpenAI API to get a response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages= [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                   "role": "user",
                    "content": query
                }

            ]
        )
        
        # Extract the response content
        answer = response.choices[0].message.content.strip()
        
        if answer:
            #print(answer)
            # Replace speak(answer) with actual function to handle speech output
            # speak(answer) 
            talk.append({'role': 'assistant', 'content': answer})
            return answer
        
        else:
            print("Sorry, I couldn't find an answer to your question.")
            # Replace speak("Sorry, I couldn't find an answer to your question.") with actual function
            # speak("Sorry, I couldn't find an answer to your question.")
            talk.append({'role': 'assistant', 'content': "Sorry, I couldn't find an answer to your question."})
            return "Sorry, I couldn't find an answer to your question."
    
    except Exception as e:
        # Handle exceptions
        print(f"An error occurred: {str(e)}")
        talk.append({'role': 'assistant', 'content': "An error occurred while processing your request."})
        return "An error occurred while processing your request."

# Example usage
# response = handle_conversational_ai_command("What is the capital of France?")
# print(response)



    # Perform the chat completion with the provided query
    # for message in response:
    #     # Print the entire message to debug the structure
    #     print(f"Debug message: {message}")
        
    #     # Check if message is a dict and contains 'choices'
    #     if isinstance(message, dict):
    #         if 'choices' in message:
    #             # Output the content if 'choices' is found
    #             print(message['choices'][0]['delta']['content'], end="")
    #         else:
    #             # Output the unexpected keys
    #             print(f"Unexpected response key: {list(message.keys())}")
    # for message in response:
    #     # Check if message is a dict and contains 'choices'
    #     if isinstance(message, dict) and 'choices' in message:
    #         print(message['choices'][0]['delta']['content'], end="")
    #     else:
    #         print("Unexpected response format:", message)


# Example usage
query = "What is the capital of France?"
handle_conversational_ai_command(query)


# from huggingface_hub import InferenceClient

# def handle_conversational_ai_command(query, model_name="mistralai/Mistral-7B-Instruct-v0.3", token="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"):
#     """
#     Get a chat completion from the specified Hugging Face model.

#     Args:
#         query (str): The user query to send to the model.
#         model_name (str): The name of the model to use (default is "mistralai/Mistral-7B-Instruct-v0.3").
#         token (str): The Hugging Face API token.

#     Returns:
#         str: The response from the model.
#     """
#     client = InferenceClient(model_name, token=token)

#     response_parts = []
#     try:
#         for message in client.chat_completion(
#             messages=[{"role": "user", "content": query}],
#             max_tokens=500,
#             stream=False,
#         ):
#             if message.choices[0].delta.content:
#                 response_parts.append(message.choices[0].delta.content)
        
#         # Combine the parts to form the full response
#         response = "".join(response_parts).replace('\n', ' ')
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         response = "Sorry, I couldn't process your request."

#    return response

# Example usage
# if __name__ == "__main__":
#     query = "What is the capital of France?"
#     result = get_chat_completion(query)
#     print(result)





# client = Groq()
# def handle_conversational_ai_command(query):
#     #global talk
#     #talk.append({'role': 'user', 'parts': [query]})
#     completion = client.chat.completions.create(
#         model="llama3-70b-8192",
#         messages=[
#             {
#                 "role": "user",
#                 "content": query #"who is nana akuffo addo"
#             }
#         ],
#         temperature=1,
#         max_tokens=1024,
#         top_p=1,
#         stream=True,
#         stop=None,
#     )

#     response_ = []
#     for chunk in completion:
#         content = chunk.choices[0].delta.content
#         if content:  # Ensure the content is not None
#             response_.append(content)

#     answer = "".join(response_).replace('\n', ' ')
#     # response = model.generate_content(talk, stream=True)
#     # answer_found = False
#     # num_sentences = 0
#     # answer = ""

#     # for chunk in response:
#     #     if hasattr(chunk, 'text'):
#     #         sentence = chunk.text.replace("*", "").strip()
#     #         if sentence:
#     #             answer += sentence + " "
#     #             num_sentences += 1
#     #             if num_sentences >= 4:
#     #                 break

#     # if answer:
#     #     answer_found = True
#     # if not answer_found:
#     #     answer = "Sorry, I couldn't find an answer to your question."
    
#     # talk.append({'role': 'model', 'parts': [answer if answer_found else "No answer found"]})
#     return answer
    

# # Example usage
# #talk = []
# user_query = "Who is Akuffo Addo?"
# response = handle_conversational_ai_command(user_query)
# print(response)


# Function to handle actions based on the input text
def perform_action(input_text):
        return handle_conversational_ai_command(input_text)
        return [text[i:i+50] for i in range(0, len(text), 50)]

# Function to handle the translation and action processing
def process_twi_text(input_text):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_english_translation = executor.submit(translate_text, input_text, "tw", "en")
        english_translation = future_english_translation.result()
        
        if english_translation:
            response_chunks = []
            response = perform_action(english_translation)

            for chunk in response:
                response_chunks.append(chunk)
            full_response = ''.join(response_chunks)

            future_twi_translation = executor.submit(translate_text, response, "en", "tw")
            twi_translation = future_twi_translation.result()
            if twi_translation:
                return english_translation, response, twi_translation
            else:
                return english_translation, response, "Twi translation failed."
        else:
            return None, None, "Translation failed."


#3ec290030be741dca1f7ea9e38d653f0
KHAYA_TOKEN = ''
# Text to speech
def convert_text_to_speech(text):
    try:
        url = "https://translation-api.ghananlp.org/tts/v1/tts"
        hdr = {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
            'Ocp-Apim-Subscription-Key': KHAYA_TOKEN,
        }
        data = {
            "text": text,
            "language": "tw"
        }
        req = requests.post(url, headers=hdr, json=data)
        if req.status_code == 200:
            return req.content
        else:
            print("Failed to convert text to speech:", req.status_code)
    except Exception as e:
        print("Error during text-to-speech conversion:", e)


def generateAudio(text):
    audio_data = convert_text_to_speech(text)

    # Save audio data to a file
    with open("tts_app/static/audio/tts_written_to_file.wav", 'bw') as file:
        file.write(audio_data)

    # Read the audio file
    audio = AudioSegment.from_file("tts_app/static/audio/tts_written_to_file.wav", format="wav")

    # Play the audio using pydub
    # play(audio)

    # Re-export the audio file
    audio.export('tts_app/static/audio/tts_written_to_file_re_exported.wav', format='wav')

    # Play the re-exported audio using pydub
    re_exported_audio = AudioSegment.from_file("tts_app/static/audio/tts_written_to_file_re_exported.wav", format="wav")
# Django view to render the home page
def index(request):
    return render(request, 'index.html')

# Django view to handle the translation and processing request
@csrf_exempt
def translate(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        english_translation, response, twi_translation = process_twi_text(input_text)
        
        # Convvert to audio
        generateAudio(str(twi_translation))
        localpath = 'tts_app/static/audio/tts_written_to_file.wav'
        cloudPath = 'images/images1.wav'
        firebase.storage().child(cloudPath).put(localpath)
        # play(audio.export)
        #6 Send to firebase and get audio link


        if english_translation and response:
            # Store the query and response
            old_queries.append({
                "query": input_text,
                "response": twi_translation
            })

        return JsonResponse({'response': twi_translation})
    return JsonResponse({'response': 'Invalid request'}, status=400)

# Django view to fetch old queries
@csrf_exempt
def get_old_queries(request):
    if request.method == 'GET':
        return JsonResponse(old_queries, safe=False)
    return JsonResponse({'response': 'Invalid request'}, status=400)


