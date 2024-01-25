import tensorflow as tf
from tensorflow import keras
import numpy as np
import os, pickle, re
import streamlit as st

def load_pickle_file(filepath):
    with open(filepath, "rb") as f:
        pickle_object = pickle.load(f)
    return pickle_object

@st.cache_resource
def load_model(filepath):
    model_loaded = keras.models.load_model(filepath)
    return model_loaded

PATH = os.getcwd()
model_folder = "latest"
tokenizer_filepath = os.path.join(PATH, "saved_models", model_folder, "tokenizer.json")
label_encoder_path = os.path.join(PATH, "saved_models", model_folder, "label_encoder.json")
model_filepath = os.path.join(PATH, "saved_models", model_folder, "saved_model.h5")

tokenizer = load_pickle_file(tokenizer_filepath)
label_encoder = load_pickle_file(label_encoder_path)
model = load_model(model_filepath)

st.title("Text Classification for Ecommerce item")

with st.form("input_form"):
    text_input = st.text_area("Input item name/description here")
    submitted = st.form_submit_button("Submit")

text_inputs = [text_input]

def remove_unwanted_string(text_inputs):
    for index, data in enumerate(text_inputs):
        text_inputs[index] = re.sub("<.*?>", " ", data)
        text_inputs[index] = re.sub("[^a-zA-Z]", " ", data).lower()

    return text_inputs

text_removed = remove_unwanted_string(text_inputs)
text_token = tokenizer.texts_to_sequences(text_removed)
text_padded = keras.preprocessing.sequence.pad_sequences(text_token, maxlen=(200), padding="post", truncating="post")

y_score = model.predict(text_padded)
y_pred = np.argmax(y_score, axis=1)
label_map = {i:classes for i, classes in enumerate(label_encoder.classes_)}
result = label_map[y_pred[0]]

st.header('Label list')
st.write(label_encoder.classes_)
st.header('Prediction score')
st.write(y_score)
st.header('Final prediction')
st.write(f"The category of the item is: {result}")