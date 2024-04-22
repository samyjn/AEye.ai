import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
from PIL import Image

import os
import pickle
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

import urllib.request

# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # Open default camera (index 0)
    ret, frame = cap.read()  # Read a frame from the camera
    cap.release()  # Release the camera
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
    return None

def capture_mobile_image(URL):

    # URL = "http://10.12.37.153:8080/shot.jpg"


    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)

    frame = cv2.imdecode(img_arr, -1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return img


# Function to recognize speech
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        audio = r.listen(source)

    # Using Google to recognize audio
    try:
        MyText = r.recognize_google(audio)
        MyText = MyText.lower()
        return MyText
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None


genai.configure(api_key="AIzaSyAtq9sqiAdKiuPs9gKKSPkaPMbSdXW15rs")

def get_gemini_response(input_text, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input_text:
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text


vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# tokenizer = Tokenizer()
with open('pages/tokenizer1.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 35

with open('pages/features1.pkl', 'rb') as f:
    features = pickle.load(f)

savedmodel = tf.keras.models.load_model('pages/best_model_latest.h5')

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

def response_processing(text):
    words = text.split()
    
    words_to_remove = ['startseq', 'endseq']
    modified_words = [word for word in words 
    if word.lower() not in 
    [w.lower() for w in words_to_remove]]
    return ' '.join(modified_words)

def get_custom_model_response(image_path):
    # load image
    # image = load_img(image_path, target_size=(224, 224))
    image = image_path.resize((224,224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = vgg_model.predict(image, verbose=0)
    # predict from the trained model
    res = predict_caption(savedmodel, feature, tokenizer, max_length)
    resp = response_processing(res)
    return resp


def start_button():
    start = True

def stop_button():
    start = False

st.sidebar.header("Navigation")

# Sidebar navigation links with bullets
st.sidebar.markdown("- [GitHub Repo](https://github.com/samyjn/AEye.ai.git)")
st.sidebar.write("---")

# Connect with me section
st.sidebar.markdown("Connect with us:")
github_link = "[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat-square&logo=github)](https://github.com/samyjn)"
linkedin_link = "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com)"
email_link = "[![Email](https://img.shields.io/badge/Google-Mail-blue?style=flat-square&logo=gmail)](mailto:samyjn211@gmail.com)"

st.sidebar.markdown(github_link + " " + linkedin_link + " " + email_link)
st.sidebar.markdown("Created by Team AEye Bennett University")


def main():
    st.title("AEye.ai")
    st.write("AI Eyes in your service! Just Speak your question.")


    cam = st.checkbox('Select only if external camera source')
    if cam:
        URL = st.text_input('Give the Link to external camera source below')

    col1, col2 = st.columns([.5,1])
    with col1:
        start = st.button('Start', on_click=start_button)

    with col2:
        st.button('Stop', on_click=stop_button)


    if start:
        while True:

            input_text = recognize_speech()
            if input_text:
                st.write("You asked:", input_text)
                st.write("Capturing image...")

                if not cam:
                    image = capture_image()
                else:
                    image = capture_mobile_image(URL)

                if image is not None:
                    pil_image = Image.fromarray(image)
                    st.image(pil_image, caption='Captured Image', use_column_width=True)
                    
                    if ('describe' in input_text):

                        response_custom = get_custom_model_response(pil_image)
                        st.write("Response:", response_custom)
                        SpeakText(response_custom)

                    elif('guide' or 'directions' in input_text):

                        blind_string = "Consider I am Visually impaired "
                        input_text_new = blind_string+input_text
                        response = get_gemini_response(input_text_new, pil_image)
                        st.write("Response:", response)
                        SpeakText(response)

                    else:

                        response = get_gemini_response(input_text, pil_image)
                    
                        st.write("Response:", response)
                        SpeakText(response)
            #     else:
            #         st.warning("Failed to capture image. Please try again.")
            else:
                st.warning("Could not understand the speech. Please try again.")
        


    

if __name__ == "__main__":
    main()
