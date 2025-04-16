# (1) imports
import streamlit as st
import string
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import nltk
import fitz  # PyMuPDF
import docx # python-docx
import io    # Bytes to python-docx

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
        print("Logistic Regression Model successfully loaded (Upload).")
        return lr_model
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please make sure that the file is in the same directory.")
        return None

fake_news_lr_model = load_lr_model()

# (4) Text processing functions
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

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
st.subheader("Get a Sus Score via Upload")

# (6) Get Sus Score Upload
uploaded_file = st.file_uploader("Upload txt, pdf, or docx file:", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    try:
        file_extension = uploaded_file.name.lower().split('.')[-1]
        text = "" # Initialize empty string for the extracted text
        extraction_successful = False # Flag for successful text extraction

        # --- Text extraction based on file type ---
        st.write(f"Attempting to read '{uploaded_file.name}'...") # Info for User
        if file_extension == "txt":
            text = uploaded_file.read().decode("utf-8")
            st.info("Successfully read TXT file.")
            extraction_successful = True

        elif file_extension == "pdf":
            pdf_bytes = uploaded_file.getvalue() # Read file as bytes
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                # Check whether the PDF is password-protected
                if doc.is_encrypted:
                    st.error("The uploaded PDF file is password-protected and cannot be processed.")
                    text = None # Signals an error
                else:
                    for page in doc:
                        text += page.get_text()
                    if text.strip():
                        st.info(f"Successfully read PDF file ({len(doc)} pages).")
                        extraction_successful = True
                    else:
                         st.warning("PDF read, but no text could be extracted (is it an image-only PDF?).")
                         text = "" # Set to empty string instead of None

        elif file_extension == "docx":
            # python-docx requires a file path or a file-like object
            # use io.BytesIO to treat the bytes from the upload as a file-like object
            try:
                docx_bytes = io.BytesIO(uploaded_file.getvalue())
                doc = docx.Document(docx_bytes)
                paragraphs_text = [para.text for para in doc.paragraphs]
                # Filter empty paragraphs and join with double breaks
                text = "\n\n".join(filter(None, paragraphs_text))
                if text.strip():
                    st.info("Successfully read DOCX file.")
                    extraction_successful = True
                else:
                    st.warning("DOCX read, but no text could be extracted.")
                    text = ""
            except Exception as docx_error:
                # Catch specific docx errors if necessary
                st.error(f"Could not read the DOCX file. It might be corrupted or in an unsupported format. Error: {docx_error}")
                text = None # signals an error

        else:
            # Fallback for unexpected but permitted file types
             st.error(f"Unsupported file type '{file_extension}' encountered.")
             text = None # Signals an error


        # --- Further processing only if text has been successfully extracted ---
        if extraction_successful and text is not None and text.strip(): # Check whether text has been extracted and does not only consist of spaces
            st.subheader("Extracted text (first 500 characters):")
            st.write(text[:500] + "...") # Display part of the text

            # --- Logic for score calculation and display ---
            processed_full_text = preprocess_text(text)
            full_embeddings = get_embeddings(processed_full_text)
            overall_sus_score = get_sus_score(full_embeddings)

            st.markdown(f"### **Sus Score: {overall_sus_score:.2f}**")

            # Initialize color and label with default values
            color = 'grey'
            label = 'assessment unavailable'

            # determine color and label based on the score
            if overall_sus_score >= 0.8:
                color = 'darkred'
                label = "super sus - proceed with caution"
            elif overall_sus_score >= 0.6:
                color = 'orangered'
                label = "very sus - double-check before trusting"
            elif overall_sus_score >= 0.4:
                color = 'goldenrod'
                label = "kinda sus - dig deeper for sure"
            elif overall_sus_score >= 0.2:
                color = 'forestgreen'
                label = "not too sus - you can probably trust it"
            else: # Covers scores < 0.2
                color = 'seagreen'
                label = "pretty legit – you’re probably safe"

            # show score
            st.markdown(f"<div style='color:{color}; font-size:24px; font-weight: bold;'>Overall assessment: This article is {label}!</div>", unsafe_allow_html=True)
            st.markdown("---")

            # --- Paragraph Analysis ---
            if overall_sus_score >= 0.4:
                st.write("Analyzing individual paragraphs (available for Sus Scores >= 0.4)...")
                st.write("Please note: The quality of the paragraph analysis depends on the success of the text extraction from your file.")

                # Split the text into paragraphs based on double newlines
                paragraphs = text.split('\n\n')
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
                            st.markdown(f"> **Paragraph (Sus Score: {item['score']:.2f}):**\n> _{item['paragraph'][:300]}_...") # Show only the beginning of the paragraph
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

        elif text is not None and not text.strip() and extraction_successful:
             # Case: File read, but contained no (meaningful) text
             st.warning("The uploaded file was read successfully, but appears to contain no text or only whitespace.")
        elif not extraction_successful and text is None:
            # Error was already reported during extraction (e.g. PDF encrypted, DOCX corrupt)
            pass # st.error has already been displayed
        else:
             # AGeneral case, if something unexpected happens
             st.error("Could not extract text from the uploaded file for processing.")

    except Exception as e:
        # intercepts all other unexpected errors during processing
        st.error(f"An unexpected error occurred while processing the file '{uploaded_file.name}': {e}")
        st.exception(e) # Displays the traceback for debugging

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