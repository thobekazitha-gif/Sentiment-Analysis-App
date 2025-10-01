# app.py - Full Sentiment Analysis Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from rake_nltk import Rake
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
import io

# Ensure better visuals
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
sns.set_style("whitegrid")

# -----------------------
# Utility functions
# -----------------------
@st.cache_data
def get_sentiment_label(text: str):
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral"
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

@st.cache_data
def extract_keywords(text: str, max_keywords=20):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    kws = rake.get_ranked_phrases()
    return kws[:max_keywords]

def plot_pie_counts(series, title="Sentiment Distribution"):
    fig, ax = plt.subplots()
    counts = series.value_counts()
    counts.plot.pie(autopct='%1.1f%%', colors=['#8BC34A','#F0625F','#90CAF9'], ax=ax)
    ax.set_ylabel("")
    ax.set_title(title)
    return fig

def plot_bar_counts(series, title="Sentiment Counts"):
    fig, ax = plt.subplots()
    sns.countplot(x=series, order=["Positive","Neutral","Negative"], palette=['#8BC34A','#90CAF9','#F0625F'], ax=ax)
    ax.set_title(title)
    return fig

def make_wordcloud_from_list(words_list, width=800, height=400):
    joined = " ".join(words_list) if words_list else ""
    if joined.strip() == "":
        return None
    wc = WordCloud(width=width, height=height, background_color='white').generate(joined)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def compute_confusion_and_report(true_labels, pred_labels, labels_order=["Positive","Negative","Neutral"]):
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)
    report = classification_report(true_labels, pred_labels, labels=labels_order, zero_division=0)
    return cm, report

def to_csv_bytes(df: pd.DataFrame):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode('utf-8')

# -----------------------
# App layout
# -----------------------
st.title("📊 Sentiment Analysis Dashboard")
st.write("Enter text to analyze, or upload a CSV with a `text` column. Optional `label` column will enable evaluation metrics.")

tab1, tab2 = st.tabs(["Single Text Analysis", "Dataset Analysis"])

# -----------------------
# Single Text Analysis
# -----------------------
with tab1:
    st.header("Analyze a Single Text")
    user_text = st.text_area("Paste or type text here:", height=160)

    col1, col2 = st.columns([2,1])

    with col1:
        if st.button("Analyze Text", key="analyze_single"):
            if user_text.strip() == "":
                st.warning("Please enter text to analyze.")
            else:
                label = get_sentiment_label(user_text)
                st.markdown(f"**Sentiment:** :blue[{label}]")
                polarity = TextBlob(user_text).sentiment.polarity
                st.markdown(f"**Polarity score:** {polarity:.3f}")

                # Extract keywords and show top 10
                keywords = extract_keywords(user_text, max_keywords=30)
                if keywords:
                    st.subheader("Top keywords")
                    st.write(keywords[:10])
                else:
                    st.write("No keywords found.")

                # Wordcloud
                wc_fig = make_wordcloud_from_list(keywords)
                if wc_fig:
                    st.pyplot(wc_fig)
    with col2:
        st.info("Quick tips:\n- Short sentences can be ambiguous.\n- TextBlob is rule-based; for advanced models consider transformers (needs more storage).")

# -----------------------
# Dataset Analysis
# -----------------------
with tab2:
    st.header("Batch Dataset Analysis")
    uploaded = st.file_uploader("Upload CSV file (must contain 'text' column). Optional: 'label' column for evaluation.", type=["csv"])

    if uploaded is None:
        st.info("No dataset uploaded yet. You can upload a CSV to analyze multiple texts at once.")
    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            df = None

        if df is not None:
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                st.subheader("Preview data")
                st.dataframe(df.head())

                # Run sentiment labeling (cached)
                with st.spinner("Computing sentiment for each row..."):
                    df['sentiment'] = df['text'].astype(str).apply(get_sentiment_label)

                # show counts & charts
                st.subheader("Sentiment Summary")
                counts = df['sentiment'].value_counts()
                st.write(counts)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.pyplot(plot_pie_counts(df['sentiment']))
                with col_b:
                    st.pyplot(plot_bar_counts(df['sentiment']))

                # Optional filter by sentiment in sidebar
                st.sidebar.header("Filters & Export")
                chosen = st.sidebar.multiselect("Filter by sentiment", options=["Positive","Neutral","Negative"], default=["Positive","Neutral","Negative"])
                filtered_df = df[df['sentiment'].isin(chosen)]
                st.sidebar.markdown(f"Showing {len(filtered_df)} rows after filter")

                # Keywords across dataset
                st.subheader("Keywords (from all texts)")
                all_keywords = []
                for txt in df['text'].astype(str).tolist():
                    kws = extract_keywords(txt, max_keywords=7)
                    all_keywords.extend(kws)
                if all_keywords:
                    wc_fig2 = make_wordcloud_from_list(all_keywords)
                    if wc_fig2:
                        st.pyplot(wc_fig2)
                else:
                    st.write("No keywords extracted.")

                # If label exists, show confusion matrix & classification report
                if 'label' in df.columns:
                    st.subheader("Evaluation (using 'label' column)")
                    # attempt to normalize labels to Positive/Negative/Neutral
                    # if labels are not in that format, attempt to map using TextBlob heuristics
                    true_labels = df['label'].astype(str).tolist()
                    pred_labels = df['sentiment'].astype(str).tolist()

                    # Only include rows where true label is in the expected set or mapable
                    allowed = ["Positive","Negative","Neutral"]
                    # If user labels are free text, try mapping them:
                    def map_label(x):
                        x = str(x).strip()
                        if x.lower() in ["pos","positive","p","1","+","plus"]:
                            return "Positive"
                        if x.lower() in ["neg","negative","n","-","minus","0"]:
                            return "Negative"
                        return "Neutral"
                    mapped_true = [map_label(x) for x in true_labels]

                    cm, report = compute_confusion_and_report(mapped_true, pred_labels, labels_order=["Positive","Negative","Neutral"])
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Positive","Negative","Neutral"], yticklabels=["Positive","Negative","Neutral"], ax=ax_cm)
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    st.pyplot(fig_cm)

                    st.subheader("Classification Report")
                    st.text(report)

                # Export analyzed CSV
                st.subheader("Export")
                csv_bytes = to_csv_bytes(filtered_df)
                st.download_button("Download analyzed CSV", data=csv_bytes, file_name="analyzed_sentiment.csv", mime="text/csv")
