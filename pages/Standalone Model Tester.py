import streamlit as st
from PIL import Image
import numpy as np
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


st.info("NOTE: In order to use this, you need to give webcam access.")

vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# vgg_model = tf.keras.models.load_model('vgg16_model.keras')

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

def get_response(image_path):
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

    
    st.title("Personal Model Testing Facility")
    st.write("Capture an Image from Webcam or Upload an Image")


    capture_mode = st.radio("Choose Capture Mode", ("Upload", "Webcam"))


    if capture_mode == "Upload":

        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)


    else:
        # Webcam capture mode
        img_file_buffer = st.camera_input(
            label="",
            key="webcam",
            help="Make sure you have given webcam permission to the site"
        )

        if img_file_buffer is not None:

            image = Image.open(img_file_buffer)
            
            st.image(image, caption='Captured Image', use_column_width=True)
    
    # input=st.text_input("Input Prompt: ",key="input")
    submit = st.button("Get Response")

    if submit:
        response=get_response(image)
        st.subheader("The Response is")
        st.write(response)

    
if __name__ == "__main__":
    main()
