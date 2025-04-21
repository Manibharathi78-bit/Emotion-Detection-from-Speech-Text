# Emotion-Detection-from-Speech-Text
Developed a GUI-based emotion detection tool using Python, HuggingFace Transformers, and Tkinter.  Integrated BERT-based emotion classification to analyze both spoken and typed input.  Implemented real-time speech-to-text conversion using the SpeechRecognition library and Google API. 

import speech_recognition as sr
from transformers import pipeline
import tkinter as tk  #graphical user interfaces  

# Initialize model and tools
emotion_model = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
recognizer = sr.Recognizer()  #PROCESS THE SPEECH TO TEXT


# Emotion color mapping
emotion_colors = {
    "joy": "#90ee90",
    "anger": "#ff6961",
    "sadness": "#87cefa",
    "fear": "#ffcc00",
    "surprise": "#ffcc00",
    "neutral": "#d3d3d3"
}

def detect_emotion(text):
    result = emotion_model(text)[0]
    return result['label'].lower()


def start_listening():
    with sr.Microphone() as source:
        output_label.config(text="Listening... ")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio).capitalize()
            emotion = detect_emotion(text)
            output_label.config(text=text)
            emotion_display.config(
                text=f"Emotion: {emotion.capitalize()}",
                bg=emotion_colors.get(emotion, "white")
            )
        except sr.UnknownValueError:
            output_label.config(text="⚠️ Could not understand. Try again!")
        except sr.RequestError:
            output_label.config(text="⚠️ Network issue. Please check your connection.")


def detect_from_text():
    user_text = text_entry.get()
    if user_text.strip():
        emotion = detect_emotion(user_text)
        output_label.config(text=user_text)  
        emotion_display.config(
            text=f"Emotion: {emotion.capitalize()}",
            bg=emotion_colors.get(emotion, "white")
        )
    else:
        output_label.config(text="⚠️ Please enter some text.")


# GUI Setup
root = tk.Tk()
root.title("Emotion Voice & Text Detector")
root.geometry("520x420")

tk.Label(root, text="You said / typed:", font=("Arial", 14)).pack(pady=10)
output_label = tk.Label(root, text="", font=("Arial", 12), wraplength=450)
output_label.pack()

emotion_display = tk.Label(root, text="", font=("Arial", 18, "bold"), width=30)
emotion_display.pack(pady=20)

tk.Button(root, text="Start Listening", command=start_listening, font=("Arial", 12)).pack(pady=5)

tk.Label(root, text="Or type something below:", font=("Arial", 12)).pack(pady=5)
text_entry = tk.Entry(root, font=("Arial", 12), width=40)
text_entry.pack(pady=5)

tk.Button(root, text="Detect from Text", command=detect_from_text, font=("Arial", 12)).pack(pady=5)
tk.Button(root, text="❌ Exit", command=root.destroy, font=("Arial", 12)).pack(pady=10)

root.mainloop()
