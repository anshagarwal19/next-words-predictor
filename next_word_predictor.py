import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

def generate_text(model,tokenizer,seed_text,max_len,n_words=10):
        for _ in range(n_words):
            next_word = predict_next_word(seed_text)
            if next_word == "":
                 break
            seed_text +=" "+ next_word
        return seed_text
# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the *next 10 words*.")

user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = generate_text(model,tokenizer,user_input,max_len,10)
        st.success(f"*Predicted Next 10 Words:* {next_word}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")