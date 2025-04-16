# (1) imports
import streamlit as st
import string
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import nltk

@st.cache_resource
def download_nltk_resources():
    nltk.download('popular')

download_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import PerceptronTagger

# (2) Loading the pre-trained DistilBERT model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    model.eval()
    return tokenizer, model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_model_and_tokenizer()

# (3) Loading the trained logistic regression model
@st.cache_resource
def load_lr_model(model_path="fake_news_lr_model.joblib"):
    try:
        lr_model = joblib.load(model_path)
        print("Logistic Regression Model successfully loaded (text input).")
        return lr_model
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please make sure that the file is in the same directory.")
        return None

fake_news_lr_model = load_lr_model()

# (4) Text processing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words])

def lemmatize_text(text):
    words = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

def get_sus_score(embeddings):
    if fake_news_lr_model is not None:
        prediction_proba = fake_news_lr_model.predict_proba(embeddings.reshape(1, -1))[0][1]
        return prediction_proba
    else:
        return 0.5

# (5) Page Title
st.title("📰🔍Fake News Detector")
st.subheader("Get a Sus Score via Text Input")
st.write('<p style="font-size: 1.2em;;">Article Title & Content:</p>', unsafe_allow_html=True)

st.caption("ℹ️ For paragraph-level analysis, please ensure paragraphs are separated by a blank line (press Enter twice).")

text = st.text_area("", key="article_text_input", height=250)

# (6) Get Sus Score Button
col_button = st.columns(3)[1]
with col_button:
    button_style = """
        <style>
        .stButton>button {
            width: 100%;
            padding: 10px;
            font-size: 1.2em;
            font-weight: bold;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    get_score_button = st.button("Get Sus Score", key="get_sus_score_button")

# (7) Settings for the Sus Score
if get_score_button:
    # Check if the text input exists and is not empty
    if 'article_text_input' in st.session_state and st.session_state.article_text_input:
        full_text = st.session_state.article_text_input

        # --- 1. Calculate Overall Score ---
        processed_full_text = preprocess_text(full_text)
        full_embeddings = get_embeddings(processed_full_text)
        overall_sus_score = get_sus_score(full_embeddings)

        # Display the overall score first
        st.markdown(f"### **Sus Score: {overall_sus_score:.2f}**")

        # Initialize color and label with defaults
        color = 'grey'
        label = 'assessment unavailable'

        # Determine color and label based on the overall score
        if overall_sus_score >= 0.8:
            color = 'darkred'
            label = "super sus - proceed with caution"
        elif overall_sus_score >= 0.6:
            color = 'orangered'
            label = "very sus - double-check before trusting"
        elif overall_sus_score >= 0.4: # Threshold for "kinda sus" and above
            color = 'goldenrod'
            label = "kinda sus - dig deeper for sure"
        elif overall_sus_score >= 0.2:
            color = 'forestgreen'
            label = "not too sus - you can probably trust it"
        else: # Covers scores < 0.2
            color = 'seagreen'
            label = "pretty legit – you’re probably safe"

        # Display the overall assessment using the determined color and label
        st.markdown(f"<div style='color:{color}; font-size:24px; font-weight: bold;'>Overall assessment: This article is {label}!</div>", unsafe_allow_html=True)

        # Add a visual separator
        st.markdown("---")

        # --- Paragraph Analysis Block ---
        # Condition: Execute analysis per paragraph only if score is >= 0.4
        if overall_sus_score >= 0.4:

            # --- 2. Paragraph-Level Analysis ---
            st.write("Analyzing individual paragraphs (available for Sus Scores >= 0.4)...")

            # Split the text into paragraphs based on double newlines
            paragraphs = full_text.split('\n\n')
            paragraph_scores = [] # To store results for each paragraph

            # Initialize progress bar
            progress_bar = st.progress(0)
            total_paragraphs = len(paragraphs)

            if total_paragraphs > 1:
                # Iterate through each paragraph
                for i, paragraph in enumerate(paragraphs):
                    # Process only non-empty paragraphs
                    if paragraph.strip():
                        processed_paragraph = preprocess_text(paragraph)
                        paragraph_embedding = get_embeddings(processed_paragraph)
                        paragraph_score = get_sus_score(paragraph_embedding)
                        paragraph_scores.append({"paragraph": paragraph, "score": paragraph_score})

                    # Update progress bar
                    progress_bar.progress((i + 1) / total_paragraphs)
                # Remove progress bar after completion
                progress_bar.empty()

                # --- 3. Identify and Display Suspicious Paragraphs ---
                suspicious_threshold = 0.7 # Planned adjustment based on testing
                highly_suspicious_paragraphs = [p for p in paragraph_scores if p["score"] >= suspicious_threshold]
                highly_suspicious_paragraphs.sort(key=lambda x: x["score"], reverse=True)

                if highly_suspicious_paragraphs:
                    st.warning("🚨 **Warning:** The following paragraphs were identified as particularly suspicious:")
                    for item in highly_suspicious_paragraphs[:3]: # Show top 3
                        st.markdown(f"> **Paragraph (Sus Score: {item['score']:.2f}):**\n> _{item['paragraph'][:300]}_...")
                    if len(highly_suspicious_paragraphs) > 3:
                        st.markdown(f"... and {len(highly_suspicious_paragraphs) - 3} more paragraphs above the threshold.")
                else:
                    # If score >= 0.4 but no paragraph scores >= 0.7
                    st.info("ℹ️ Although the overall score suggests some suspicion, no single paragraph exceeded the high suspicion threshold (0.7). The suspicion might stem from the combination of paragraphs. Ensure paragraphs were separated by blank lines for accurate analysis.")

            else:
                # Only one paragraph recognized or no paragraphs
                st.info("Detailed paragraph analysis skipped: The text was treated as a single block or no paragraphs were found after splitting.")
                if total_paragraphs == 1 and len(paragraphs[0]) > 100 : # If it was just one long paragraph
                    st.caption("(Tip: Ensure paragraphs in the source document are clearly separated, e.g., by blank lines, for better paragraph analysis.)")
                progress_bar.empty() # Ensure that the bar is gone

        else: # overall_sus_score < 0.4
            st.success("✅ Overall Sus Score is low. Detailed paragraph analysis was skipped.")

    else:
        # Warn the user if the text area was empty when the button was clicked
        st.warning("Please enter some text to analyze.")

# Other Sidebar-Content (About the Sus Score)
st.sidebar.header("About the Sus Score")
st.sidebar.info("""
    The **Sus Score** indicates how likely an article is to be fake, based on machine learning analysis.

    **How it works:**

    - An **Overall Sus Score** (0=likely real, 1=likely fake) evaluates the entire article.
    - **Suspicious Paragraphs:** If paragraphs are separated by blank lines in the input, the tool analyzes each paragraph. Paragraphs with high individual scores are highlighted, showing which parts contribute most to suspicion.
    - A higher score means a higher level of suspicion.

    This helps to flag suspicious articles and specific sections for verification. Paragraph analysis depends on the correct formatting of the input.
""")