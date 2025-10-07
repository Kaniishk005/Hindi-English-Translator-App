# app.py

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(page_title="Translator App", layout="centered")
st.title("Hindi-English Translator")

@st.cache_resource
def load_model(model_name):
    """
    Loads a translation model and tokenizer from Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    return translator

target_language = st.selectbox(
    "Select the language to translate to:",
    ("English", "Hindi")
)

if target_language == "English":
    model_name = "Helsinki-NLP/opus-mt-hi-en"
    source_text_label = "Enter Hindi Text Here:"
    button_label = "Translate to English"
    result_subheader = "Translated English Text:"
else:
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    source_text_label = "Enter English Text Here:"
    button_label = "Translate to Hindi"
    result_subheader = "Translated Hindi Text:"

try:
    translator = load_model(model_name)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

source_text = st.text_area(source_text_label, height=150)

if st.button(button_label):
    if source_text:
        with st.spinner("Translating..."):
            try:
                translation_result = translator(source_text)
                translated_text = translation_result[0]['translation_text']
                
                st.subheader(result_subheader)
                st.success(translated_text)
            except Exception as e:
                st.error(f"An error occurred during translation: {e}")
    else:
        st.warning("Please enter some text to translate.")