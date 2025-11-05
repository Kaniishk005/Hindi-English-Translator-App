import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --- 1. CONFIG & STATE ---
# Use the full page width
st.set_page_config(page_title="Translator App", layout="wide")

# Initialize session state to hold the translated text
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
# We also use session state for the source text to enable the "Clear" button
if "source_text" not in st.session_state:
    st.session_state.source_text = ""


# --- 2. MODEL LOADING ---
@st.cache_resource
def load_nllb_model():
    """
    Loads the NLLB model and tokenizer from Hugging Face.
    """
    model_name = "facebook/nllb-200-distilled-600M" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

try:
    model, tokenizer = load_nllb_model()
except Exception as e:
    st.error(f"Error loading the NLLB model: {e}")
    st.stop()


# --- 3. SIDEBAR (for controls) ---
st.sidebar.title("Language preference")
target_language = st.sidebar.selectbox(
    "Select the language to translate to:",
    ("English", "Hindi")
)


# (Optional) You could add your GitHub link here
# st.sidebar.markdown("Find the code on [GitHub](your-link-here)")


# --- 4. MAIN PAGE LAYOUT ---
# --- 4. MAIN PAGE LAYOUT (Corrected for Alignment) ---
st.title("Hindi-English Translator")

# Define language codes and labels (This part is unchanged)
if target_language == "English":
    src_lang = "hin_Deva"
    tgt_lang = "eng_Latn"
    source_text_label = "Enter Hindi Text (हिन्दी)"
    result_subheader = "Translated English Text (English)"
else:
    src_lang = "eng_Latn"
    tgt_lang = "hin_Deva"
    source_text_label = "Enter English Text (English)"
    result_subheader = "Translated Hindi Text (हिन्दी)"

# ... (translator pipeline code) ...
# ... (make sure pipeline code is *before* the columns) ...

# Use columns for a side-by-side layout
col1, col2 = st.columns(2)

with col1:
    # --- THIS IS THE FIX ---
    # 1. Put the label *outside* as a subheader
    st.subheader(source_text_label)
    
    # 2. Add label_visibility="collapsed" to the text_area
    source_text = st.text_area(
        "Source Text", # Label is now just for internal use
        height=300, 
        key="source_text",
        label_visibility="collapsed" # Hides the "Source Text" label
    )

with col2:
    # --- THIS WAS ALREADY CORRECT ---
    st.subheader(result_subheader)
    
    st.text_area(
        "Result",
        value=st.session_state.translated_text,
        height=300,
        disabled=True,
        label_visibility="collapsed" # Hides the "Result" label
    )

# ... (rest of the button code is the same) ...

# --- 5. BUTTONS ---
# Place buttons in their own columns for alignment
btn_col1, btn_col2, _ = st.columns([1, 1, 3])

with btn_col1:
    if st.button("Translate", type="primary", use_container_width=True):
        if source_text:
            with st.spinner("Translating..."):
                try:
                    translation_result = translator(source_text, num_beams=5) 
                    # Save the result to session state
                    st.session_state.translated_text = translation_result[0]['translation_text']
                except Exception as e:
                    st.error(f"An error occurred during translation: {e}")
        else:
            st.warning("Please enter some text to translate.")

with btn_col2:
    if st.button("Clear Text", use_container_width=True):
        # Clear both the source and translated text
        st.session_state.source_text = ""
        st.session_state.translated_text = ""