# (1) imports
import streamlit as st
import string
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import nltk
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json
import time
import random

# (II) Function for recognizing the news source
def identify_news_source(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    if "bbc.com" in domain:
        return "bbc"
    elif "foxnews.com" in domain:
        return "fox"
    elif "aljazeera.com" in domain:
        return "aljazeera"
    return None

# (III) Functions for scraping the article text for each news source
def scrape_bbc_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        title_element = soup.find('h1')
        if not title_element:
            title_element = soup.find('meta', property="og:title")
            title = title_element['content'] if title_element and 'content' in title_element.attrs else ""
        else:
            title = title_element.text.strip()

        # Extract main text directly from the <article> tag
        main_text = []
        article_element = soup.find('article')
        if article_element:
            p_elements = article_element.find_all('p')
            for p in p_elements:
                main_text.append(p.text.strip())

        return title, "", " ".join(main_text)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return "", "", ""
    except Exception as e:
        st.error(f"Error parsing BBC article: {e}")
        return "", "", ""

def scrape_fox_news_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find JSON-LD script tag with type “NewsArticle”
        news_article_json = soup.find('script', type='application/ld+json')
        if news_article_json:
            try:
                article_data = json.loads(news_article_json.string)
                if article_data.get('@type') == 'NewsArticle':
                    title = article_data.get('headline', '')
                    text = article_data.get('articleBody', '')
                    return title, "", text
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON from Fox News article: {e}")
                return "", "", ""

        # Fallback strategy (meta tags for titles)
        title_element = soup.find('title')
        title = title_element.text.strip() if title_element else ""
        og_title_element = soup.find('meta', property='og:title')
        if og_title_element and not title:
            title = og_title_element['content']

        return title, "", "" # No fallback strategy implemented for the text yet

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return "", "", ""
    except Exception as e:
        st.error(f"Error parsing Fox News article: {e}")
        return "", "", ""

def scrape_aljazeera_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        title = ""
        description = ""
        main_text = []

        # Extraction of title and description from structured data
        script_tags = soup.find_all('script', {'type': 'application/ld+json'})
        for script_tag in script_tags:
            try:
                json_data = json.loads(script_tag.string)
                if isinstance(json_data, dict) and json_data.get('@type') == 'NewsArticle':
                    title = json_data.get('headline', "")
                    description = json_data.get('description', "")
                    break
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON: {e}")
                continue

        # Fallback for Title
        if not title:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.text.split('|')[0].strip()

        # Extraction of the main text
        body_elements = soup.find_all('div', class_='wysiwyg')
        for body in body_elements:
            paragraphs = body.find_all('p')
            for p in paragraphs:
                main_text.append(p.text.strip())

        return title, "", " ".join(main_text)

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {url} - {e}")
        return "", "", ""
    except Exception as e:
        st.error(f"Error parsing Al Jazeera article: {url} - {e}")
        return "", "", ""

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
        print("Logistic Regression Modell successfully loaded (URL).")
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
st.subheader("Get a Sus Score via URL")

url = st.text_input("Enter the URL of the news page:")
if url:
    news_source = identify_news_source(url)
    title = ""
    subtitle = ""
    main_text = ""

    with st.spinner(f"Scraping article from {url}..."):
        try:
            if news_source == "bbc":
                title, subtitle, main_text = scrape_bbc_article(url)
            elif news_source == "fox":
                title, subtitle, main_text = scrape_fox_news_article(url)
            elif news_source == "aljazeera":
                title, subtitle, main_text = scrape_aljazeera_article(url)
            elif news_source is None:
                st.warning("News source not recognized. Proceeding with generic extraction (may be less accurate).")

                # --- Generic Scraping ---
                headers = {'User-Agent': 'Mozilla/5.0'} # Einfacher User-Agent
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status() # Fehler bei 4xx oder 5xx Statuscodes
                soup = BeautifulSoup(response.content, 'html.parser')
                    
                body_elements = soup.find_all('p') # Simple extraction of <p> tags
                title_element = soup.find('h1')
                title = title_element.text.strip() if title_element else "Title not found"

                # Filters empty paragraphs and joins with double line breaks
                paragraphs_text = [p.text.strip() for p in body_elements if p.text.strip()]
                main_text = " ".join(paragraphs_text)
                subtitle = "" # Generic scraper does not extract subtitle
        
            # --- Check if scraping was successful ---
            if title and main_text:
                scraping_successful = True
                st.success("Article content extracted successfully.")
            elif url: # If URL is there, but no text was extracted
                st.error("Could not extract sufficient text content (title and main text) from the URL.")

        except requests.exceptions.RequestException as req_err:
            st.error(f"Network error during scraping: {req_err}")
        except Exception as e:
            st.error(f"An error occurred during scraping: {e}")
            st.exception(e) # Traceback

    # --- 2. processing if scraping was successful ---
    if scraping_successful:
        st.subheader("Extracted Content:")
        st.write(f"**Title:** {title}")
        if subtitle: # Show subtitle only if available
             st.write(f"**Subtitle:** {subtitle}")
        st.write(f"**Text (excerpt):** {main_text[:500]}...") # preview

        # --- 3. calculate total score ---
        # Combine relevant parts for the overall score
        full_text_for_scoring = title + " " + subtitle + " " + main_text
        processed_text = preprocess_text(full_text_for_scoring)
        embeddings = get_embeddings(processed_text)
        overall_sus_score = get_sus_score(embeddings)
        
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
        elif overall_sus_score >= 0.4:
            color = 'goldenrod'
            label = "kinda sus - dig deeper for sure"
        elif overall_sus_score >= 0.2:
            color = 'forestgreen'
            label = "not too sus - you can probably trust it"
        else:
            color = 'seagreen'
            label = "pretty legit – you’re probably safe"

        # Display the overall assessment using the determined color and label
        st.markdown(f"<div style='color:{color}; font-size:24px; font-weight: bold;'>This article is {label}!</div>", unsafe_allow_html=True)

        # Add a visual separator
        st.markdown("---")

        # --- Paragraph Analysis Block ---
        # Condition: Execute analysis per paragraph only if score is >= 0.4
        if overall_sus_score >= 0.4:
            
            # --- 2. Paragraph-Level Analysis ---
            st.write("Analyzing individual paragraphs (available for Sus Scores >= 0.4)...")

            # Use 'main_text' for paragraph analysis (without title/subtitle)
            # Success depends heavily on whether the scraper delivers '\n\n' between paragraphs!
            paragraphs = main_text.split('\n\n')
            paragraph_scores = []

            progress_bar = st.progress(0)
            # Filter empty strings that could be created by split
            valid_paragraphs = [p for p in paragraphs if p.strip()]
            total_valid_paragraphs = len(valid_paragraphs)

            if total_valid_paragraphs > 1:
                for i, paragraph in enumerate(valid_paragraphs):
                    if paragraph.strip():
                        processed_paragraph = preprocess_text(paragraph)
                        paragraph_embedding = get_embeddings(processed_paragraph)
                        paragraph_score = get_sus_score(paragraph_embedding)
                        paragraph_scores.append({"paragraph": paragraph, "score": paragraph_score})
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_valid_paragraphs)
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
                st.caption("(Paragraph analysis works best if the source text has clearly separated paragraphs and the scraper preserves them.)")
                progress_bar.empty() # Ensure that the bar is gone

        else: # overall_sus_score < 0.4
            st.success("✅ Overall Sus Score is low. Detailed paragraph analysis was skipped.")

        if overall_sus_score > 0.7:
            st.warning("Warning: This article has a high suspicion of being fake news.")
        elif overall_sus_score > 0.4:
            st.info("This article has moderate suspicion. Please verify further.")
        else:
            st.success("This article seems safe.")
    elif url:
        st.warning("No relevant text content could be extracted from the specified URL. Please try another URL or use the other analysis mode.")

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