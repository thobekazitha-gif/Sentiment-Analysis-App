%%writefile Sentiment-Analysis-App/app.py
import streamlit as st
import pandas as pd
from transformers import pipeline
from rake_nltk import Rake
from fpdf import FPDF
import json
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load sentiment analysis model
@st.cache_resource
def load_model():
    # Using a local transformer pipeline for multi-class classification
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_model()

# Function to perform sentiment analysis
def analyze_sentiment(text):
    if not text:
        return None
    result = sentiment_analyzer(text)[0]
    # Map labels to positive/negative/neutral if necessary (this model provides positive/negative)
    # For a neutral class, a different model or custom logic would be needed
    return {"label": result['label'], "score": result['score']}

# Function to extract keywords
def extract_keywords(text):
    if not text:
        return []
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

# Function for leave-one-out token importance (simplified)
def get_token_importance(text):
    words = text.split()
    original_result = analyze_sentiment(text)
    if not original_result:
        return []

    importance = []
    for i in range(len(words)):
        modified_text = " ".join(words[:i] + words[i+1:])
        if modified_text:
            modified_result = analyze_sentiment(modified_text)
            if modified_result:
                # Simple measure: change in confidence for the original label
                confidence_change = original_result['score'] - modified_result['score']
                importance.append((words[i], confidence_change))
        else:
             # If removing a word makes the text empty, the importance is high
             importance.append((words[i], original_result['score']))

    # Sort by importance (descending)
    importance.sort(key=lambda item: item[1], reverse=True)
    return importance

# Function to create PDF
def create_pdf(data, filename="sentiment_analysis_results.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 12)
    for item in data:
        pdf.cell(200, 10, txt = f"Text: {item['text']}", ln = 1, align = 'L')
        pdf.cell(200, 10, txt = f"Sentiment: {item['sentiment']}", ln = 1, align = 'L')
        pdf.cell(200, 10, txt = f"Confidence: {item['score']:.2f}", ln = 1, align = 'L')
        pdf.cell(200, 10, txt = f"Keywords: {', '.join(item.get('keywords', []))}", ln = 1, align = 'L')
        pdf.cell(200, 10, txt = f"Token Importance (Top 5): {', '.join([f'{word} ({imp:.2f})' for word, imp in item.get('token_importance', [])[:5]])}", ln = 1, align = 'L')
        pdf.cell(200, 10, txt = "-"*20, ln = 1, align = 'L')
    pdf.output(filename)
    return filename

st.title("Sentiment Analysis Dashboard")

st.sidebar.header("Input Options")
input_option = st.sidebar.radio("Select input method:", ("Direct Text Input", "File Upload", "Evaluation Mode"))

data = []

if input_option == "Direct Text Input":
    st.header("Direct Text Input")
    text_input = st.text_area("Enter text here:")
    if st.button("Analyze Text"):
        if text_input:
            result = analyze_sentiment(text_input)
            keywords = extract_keywords(text_input)
            token_importance = get_token_importance(text_input)
            sentiment_label = result['label']
            sentiment_score = result['score']
            st.write(f"Sentiment: {sentiment_label}")
            st.write(f"Confidence: {sentiment_score:.2f}")
            st.write(f"Keywords: {', '.join(keywords)}")
            st.write(f"Token Importance (Top 5): {', '.join([f'{word} ({imp:.2f})' for word, imp in token_importance[:5]])}")
            data.append({"text": text_input, "sentiment": sentiment_label, "score": sentiment_score, "keywords": keywords, "token_importance": token_importance})
        else:
            st.warning("Please enter some text to analyze.")

elif input_option == "File Upload":
    st.header("File Upload")
    uploaded_file = st.file_uploader("Upload a text file (.txt, .csv, .json)", type=["txt", "csv", "json"])
    if uploaded_file is not None:
        file_type = uploaded_file.type
        texts = []
        if file_type == "text/plain":
            text_data = uploaded_file.getvalue().decode("utf-8")
            texts = text_data.splitlines()
        elif file_type == "text/csv":
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts = df['text'].tolist()
                elif len(df.columns) == 1:
                     texts = df.iloc[:, 0].tolist()
                else:
                    st.error("CSV file must have a 'text' column or be a single column of texts.")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
        elif file_type == "application/json":
            try:
                json_data = json.load(uploaded_file)
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                        elif isinstance(item, str):
                             texts.append(item)
                        else:
                            st.warning("JSON list should contain objects with a 'text' key or be a list of strings.")
                else:
                    st.error("JSON file must be a list of texts or objects with a 'text' key.")
            except Exception as e:
                st.error(f"Error reading JSON file: {e}")

        if texts:
            st.write(f"Analyzing {len(texts)} lines of text...")
            batch_results = []
            # Implement batch processing for efficiency
            batch_size = st.sidebar.slider("Batch Size", 1, 100, 32)
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                results = sentiment_analyzer(batch_texts)
                for j, text in enumerate(batch_texts):
                    result = results[j]
                    keywords = extract_keywords(text)
                    token_importance = get_token_importance(text)
                    batch_results.append({"text": text, "sentiment": result['label'], "score": result['score'], "keywords": keywords, "token_importance": token_importance})
                st.write(f"Processed {min(i + batch_size, len(texts))} of {len(texts)} texts.")
            data = batch_results
            st.write("Analysis Complete.")

elif input_option == "Evaluation Mode":
    st.header("Evaluation Mode")
    st.write("Upload a CSV file with 'text' and 'label' columns for evaluation.")
    uploaded_file = st.file_uploader("Upload CSV file for evaluation", type="csv")
    if uploaded_file is not None:
        try:
            eval_df = pd.read_csv(uploaded_file)
            if 'text' in eval_df.columns and 'label' in eval_df.columns:
                st.write(f"Analyzing {len(eval_df)} texts for evaluation...")
                predictions = []
                # Implement batch processing for efficiency
                batch_size = st.sidebar.slider("Evaluation Batch Size", 1, 100, 32)
                for i in range(0, len(eval_df), batch_size):
                    batch_texts = eval_df['text'].tolist()[i:i+batch_size]
                    results = sentiment_analyzer(batch_texts)
                    predictions.extend([r['label'] for r in results])
                    st.write(f"Processed {min(i + batch_size, len(eval_df))} of {len(eval_df)} texts.")

                eval_df['prediction'] = predictions

                st.header("Evaluation Results")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(eval_df['label'], eval_df['prediction'])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                # Performance Metrics
                st.subheader("Performance Metrics")
                accuracy = accuracy_score(eval_df['label'], eval_df['prediction'])
                st.write(f"Accuracy: {accuracy:.4f}")
                report = classification_report(eval_df['label'], eval_df['prediction'], output_dict=True)
                st.json(report)

                st.header("Generate Evaluation Report")
                st.write("Use the metrics above to write your evaluation report, discussing model limitations and confidence.")
                st.text_area("Evaluation Report Template", """
## Sentiment Analysis Model Evaluation Report

**Model:** [Specify Model Used, e.g., distilbert-base-uncased-finetuned-sst-2-english]
**Evaluation Data:** [Number of samples]

**Metrics:**
- Accuracy: [Value]
- Precision (per class): [Values]
- Recall (per class): [Values]
- F1-Score (per class): [Values]

**Confusion Matrix:**
[Insert Confusion Matrix Interpretation]

**Discussion of Model Limitations:**
[Discuss limitations based on the confusion matrix and metrics, e.g., issues with specific classes, handling of sarcasm, context, etc.]

**Domain-Specific Issues:**
[Discuss any challenges or observations related to the specific domain of the evaluation data.]

**Confidence Analysis:**
[Analyze the confidence scores. Do high confidence scores correlate with correct predictions? Are there patterns in confidence for misclassified examples?]

**Conclusion:**
[Summarize the evaluation findings and potential next steps.]
                """)


        except Exception as e:
            st.error(f"Error processing evaluation file: {e}")


if data and input_option != "Evaluation Mode":
    st.header("Analysis Results")
    results_df = pd.DataFrame(data)
    st.dataframe(results_df)

    st.header("Sentiment Distribution")
    sentiment_counts = results_df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.header("Confidence Histogram")
    fig, ax = plt.subplots()
    ax.hist(results_df['score'], bins=20)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


    st.header("Export Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Export as CSV"):
            csv_file = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_file,
                file_name='sentiment_analysis_results.csv',
                mime='text/csv',
            )
    with col2:
        if st.button("Export as JSON"):
            # Exclude token importance from JSON export for simplicity, or format as needed
            json_data_export = [{"text": item["text"], "sentiment": item["sentiment"], "score": item["score"], "keywords": item.get("keywords", [])} for item in data]
            json_file = json.dumps(json_data_export, indent=4).encode('utf-8')
            st.download_button(
                label="Download JSON",
                data=json_file,
                file_name='sentiment_analysis_results.json',
                mime='application/json',
            )
    with col3:
        if st.button("Export as PDF"):
            pdf_filename = create_pdf(data)
            with open(pdf_filename, "rb") as f:
                pdf_file = f.read()
            st.download_button(
                label="Download PDF",
                data=pdf_file,
                file_name=pdf_filename,
                mime='application/pdf',
            )

st.sidebar.header("Tips for Explainability")
st.sidebar.info("Analyze individual texts to see the top keywords and token importance scores.")
st.sidebar.header("Settings")
# Add any future settings here, e.g., model selection, confidence threshold
