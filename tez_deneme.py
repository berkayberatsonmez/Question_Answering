import responses
from neuralintents import GenericAssistant
import speech_recognition
import pyttsx3 as tts
import sys

recognizer = speech_recognition.Recognizer()

speaker = tts.init('sapi5')
voices = speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)
speaker.setProperty('rate', 150)

def speak(audio):
    speaker.say(audio)
    speaker.runAndWait()

def hello():
    speaker.say("Hello. What can I do for you?")
    speaker.runAndWait()

def quit():
    speaker.say("Bye")
    speaker.runAndWait()
    sys.exit(0)

def thanks():
    speak("You're welcome sir!")
    speaker.runAndWait()

def toothbrush():
    speak("The toothbrush is on shelf 1 in the closet in the bathroom.")
    speaker.runAndWait()

def Living_to_Bedroom():
    speak("After exiting the living room, if we go straight to the left, the first room on our left is the bedroom.")
    speaker.runAndWait()
def Living_to_Kitchen():
    speak("If we go straight after leaving the living room, we will go to the kitchen.")
    speaker.runAndWait()
def Living_to_Bathroom():
    speak("After exiting the hall, if we go straight to the left, the first room on our right is the bathroom.")
    speaker.runAndWait()

def Bedroom_to_Living():
    speak("After exiting the bedroom, if we go straight to the right, the first room to our right is the living room.")
    speaker.runAndWait()
def Bedroom_to_Bathroom():
    speak("If we go straight after leaving the bedroom, we'll go to the bathroom.")
    speaker.runAndWait()
def Bedroom_to_Kitchen():
    speak("After exiting the bedroom, if we go straight to the right, the first room on our left is the kitchen.")
    speaker.runAndWait()

mappings = {
    "greeting": hello,
    "exit": quit,
    "thanks": thanks,
    "Living_to_Bedroom": Living_to_Bedroom,
    "Living_to_Kitchen": Living_to_Kitchen,
    "Living_to_Bathroom": Living_to_Bathroom,
    "Bedroom_to_Living": Bedroom_to_Living,
    "Bedroom_to_Kitchen": Bedroom_to_Kitchen,
    "Bedroom_to_Bathroom": Bedroom_to_Bathroom,
    "toothbrush": toothbrush
}

assistant = GenericAssistant('deneme.json', intent_methods=mappings)
assistant.train_model()

while True:

    print("Listening...")
    try:
        with speech_recognition.Microphone() as mic:

            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)
            print("recognizing...")
            message = recognizer.recognize_google(audio, language='en-en')
            message = message.lower()
            print(message)

        assistant.request(message)
    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()