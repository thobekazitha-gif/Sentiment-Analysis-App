# app.py
import os
import streamlit as st
import requests
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
 
# -----------------------------
# App description
# -----------------------------
st.title("Sentiment Analysis Dashboard")
st.markdown("""
This app allows you to analyze the sentiment of text data and extract keywords.  
**Features include:**
- Sentiment classification: Positive, Neutral, Negative
- Keyword extraction (without NLTK)
- Batch processing for multiple texts
- Download results as CSV
- Interactive sentiment distribution chart
- Works with Hugging Face API or local fallback if API fails
""")
 
# -----------------------------
# Hugging Face API Setup
# -----------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
USE_HF_API = True if HF_API_TOKEN else False
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if USE_HF_API else None
 
# -----------------------------
# Sentiment analysis functions
# -----------------------------
def analyze_sentiment_hf(text):
    if not HF_API_TOKEN:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": text})
        response.raise_for_status()
        data = response.json()
        # Ensure proper structure
        if isinstance(data, list) and isinstance(data[0], list):
            sentiments = {item['label'].lower(): item['score'] for item in data[0]}
            return {
                "positive": sentiments.get("positive", 0.0),
                "neutral": sentiments.get("neutral", 0.0),
                "negative": sentiments.get("negative", 0.0)
            }
        else:
            st.warning("Unexpected HF API response. Using default neutral sentiment.")
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    except Exception as e:
        st.warning(f"Hugging Face API failed: {e}. Using default neutral sentiment.")
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
 
def analyze_sentiment_local(text):
    positive_words = ["good", "great", "love", "excellent", "happy", "awesome"]
    negative_words = ["bad", "terrible", "hate", "poor", "sad", "awful"]
    text_lower = text.lower()
    score = sum(word in text_lower for word in positive_words) - sum(word in text_lower for word in negative_words)
    if score > 0:
        return {"positive": 1.0, "neutral": 0.0, "negative": 0.0}
    elif score < 0:
        return {"positive": 0.0, "neutral": 0.0, "negative": 1.0}
    else:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
 
# -----------------------------
# Keyword extraction
# -----------------------------
def extract_keywords(text, max_keywords=6):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set([
        "the", "and", "is", "in", "it", "of", "to", "a", "for", "on", "with", "that", "this",
        "i", "was", "but", "are", "my", "so", "at", "as", "be", "an", "they", "or", "its"
    ])
    filtered_words = [w for w in words if w not in stopwords]
    word_counts = Counter(filtered_words)
    keywords = [w for w, _ in word_counts.most_common(max_keywords)]
    return keywords
 
# -----------------------------
# Input: Text or File
# -----------------------------
text_input = st.text_area("Enter text here:")
uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
 
texts = []
if uploaded_file:
    texts = uploaded_file.read().decode("utf-8").splitlines()
elif text_input:
    texts = [text_input]
 
# -----------------------------
# Processing
# -----------------------------
if texts:
    results = []
    for t in texts:
        sentiments = analyze_sentiment_hf(t) if USE_HF_API else analyze_sentiment_local(t)
        keywords = extract_keywords(t)
        results.append({
            "Text": t,
            "Positive": sentiments["positive"],
            "Neutral": sentiments["neutral"],
            "Negative": sentiments["negative"],
            "Keywords": ", ".join(keywords)
        })
 
    df = pd.DataFrame(results)
    st.dataframe(df)
 
    # Download CSV
    st.download_button("Download CSV", df.to_csv(index=False), file_name="sentiment_results.csv")
 
    # -----------------------------
    # Sentiment distribution chart
    # -----------------------------
    st.subheader("Sentiment Distribution")
    sentiment_totals = df[["Positive", "Neutral", "Negative"]].sum()
    fig, ax = plt.subplots()
    ax.bar(sentiment_totals.index, sentiment_totals.values, color=["green", "gray", "red"])
    ax.set_ylabel("Sum of Scores")
    st.pyplot(fig)
