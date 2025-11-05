import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Translator App", layout="centered")
st.title("Hindi-English Translator")

# --- MODIFIED FUNCTION ---
# We load the base model and tokenizer once, and cache it.
# We will create the specific translator pipelines later.
@st.cache_resource
def load_nllb_model():
    """
    Loads the NLLB model and tokenizer from Hugging Face.
    """
    # <--- Use the NLLB model (600M parameters)
    model_name = "facebook/nllb-200-distilled-600M" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# --- LOAD MODEL OUTSIDE ---
# <--- Load the model and tokenizer only once at the start.
try:
    model, tokenizer = load_nllb_model()
except Exception as e:
    st.error(f"Error loading the NLLB model: {e}")
    st.stop()


target_language = st.selectbox(
    "Select the language to translate to:",
    ("English", "Hindi")
)

# --- MODIFIED LOGIC ---
if target_language == "English":
    # <--- Set language codes for NLLB (Hindi to English)
    src_lang = "hin_Deva"  # Hindi (Devanagari)
    tgt_lang = "eng_Latn"  # English (Latin)
    
    source_text_label = "Enter Hindi Text Here:"
    button_label = "Translate to English"
    result_subheader = "Translated English Text:"
else:
    # <--- Set language codes for NLLB (English to Hindi)
    src_lang = "eng_Latn"  # English (Latin)
    tgt_lang = "hin_Deva"  # Hindi (Devanagari)

    source_text_label = "Enter English Text Here:"
    button_label = "Translate to Hindi"
    result_subheader = "Translated Hindi Text:"

# --- CREATE PIPELINE ---
# <--- Create the specific translator pipeline *after*
#      we know the source and target languages.
#      We use device=-1 for CPU (safe default). Use 0 for GPU if available.
try:
    translator = pipeline(
        "translation", 
        model=model, 
        tokenizer=tokenizer, 
        src_lang=src_lang, 
        tgt_lang=tgt_lang,
        device=-1 # <--- Use -1 for CPU, 0 for GPU
    )
except Exception as e:
    st.error(f"Error creating the translation pipeline: {e}")
    st.stop()


source_text = st.text_area(source_text_label, height=150)

if st.button(button_label):
    if source_text:
        with st.spinner("Translating..."):
            try:
                # --- MODIFIED CALL ---
                # <--- Add num_beams=5 for much higher accuracy
                translation_result = translator(source_text, num_beams=5) 
                translated_text = translation_result[0]['translation_text']
                
                st.subheader(result_subheader)
                st.success(translated_text)
            except Exception as e:
                st.error(f"An error occurred during translation: {e}")
    else:
        st.warning("Please enter some text to translate.")