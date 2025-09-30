# app.py
import os
import streamlit as st
from rake_nltk import Rake
import nltk
import requests
import pandas as pd
 
# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
 
# -----------------------------
# Hugging Face API Setup
# -----------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
USE_HF_API = True if HF_API_TOKEN else False
 
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if USE_HF_API else None
 
def analyze_sentiment_hf(text):
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": text})
        response.raise_for_status()
        data = response.json()
        # Multi-class: positive, neutral, negative
        sentiments = {item['label'].lower(): item['score'] for item in data[0]}
        return sentiments
    except Exception as e:
        st.warning(f"Hugging Face API failed: {e}")
        return None
 
def analyze_sentiment_local(text):
    # Simple local rule-based sentiment as fallback
    positive_words = ["good", "great", "love", "excellent", "happy"]
    negative_words = ["bad", "terrible", "hate", "poor", "sad"]
    text_lower = text.lower()
    score = sum(word in text_lower for word in positive_words) - sum(word in text_lower for word in negative_words)
    if score > 0:
        return {"positive": 1.0, "neutral": 0.0, "negative": 0.0}
    elif score < 0:
        return {"positive": 0.0, "neutral": 0.0, "negative": 1.0}
    else:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
 
# -----------------------------
# Keyword Extraction
# -----------------------------
rake_extractor = Rake()
 
def extract_keywords(text, max_keywords=6):
    rake_extractor.extract_keywords_from_text(text)
    keywords = rake_extractor.get_ranked_phrases()[:max_keywords]
    return keywords
 
# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Sentiment Analysis App")
st.write("Analyze text sentiment and extract keywords!")
 
# Input: Text or File
text_input = st.text_area("Enter text here:")
uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
 
texts = []
if uploaded_file:
    texts = uploaded_file.read().decode("utf-8").splitlines()
elif text_input:
    texts = [text_input]
 
if texts:
    results = []
    for t in texts:
        # Sentiment analysis
        sentiments = analyze_sentiment_hf(t) if USE_HF_API else analyze_sentiment_local(t)
        # Keyword extraction
        keywords = extract_keywords(t)
        results.append({"Text": t, "Positive": sentiments["positive"], "Neutral": sentiments["neutral"], "Negative": sentiments["negative"], "Keywords": ", ".join(keywords)})
 
    df = pd.DataFrame(results)
    st.dataframe(df)
 
    # Option to download results
    st.download_button("Download CSV", df.to_csv(index=False), file_name="sentiment_results.csv")
