# app.py
import os
import streamlit as st
import pandas as pd
import json
import time
from typing import List, Dict, Tuple
import requests
from rake_nltk import Rake
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
from fpdf import FPDF
from io import BytesIO
from tqdm import tqdm
 
# CONFIG
USE_HF_API = True  # set False to use local transformers
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"  # 3-class model: negative/neutral/positive
BATCH_SIZE = 16
 
if USE_HF_API and not HF_API_TOKEN:
    st.warning("HF_API_TOKEN not found. Please set env var 'HF_API_TOKEN' or set USE_HF_API = False for local mode.")
 
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
 
# --- Utility functions ------------------------------------------------------
def call_hf_inference(texts: List[str], model: str = HF_MODEL, token: str = HF_API_TOKEN) -> List[Dict]:
    """Call Hugging Face inference API in batches. Returns raw JSON predictions per text."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://api-inference.huggingface.co/models/{model}"
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        payload = {"inputs": batch, "options": {"wait_for_model": True}}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            # HF returns one entry per input
            if isinstance(out, dict) and out.get("error"):
                raise RuntimeError(out)
            results.extend(out)
        except Exception as e:
            st.error(f"Hugging Face API error: {e}")
            # populate with placeholder neutral low-confidence
            results.extend([[
                {"label":"NEUTRAL","score":0.0}
            ] for _ in batch])
    return results
 
# Local model option
def local_classify(texts: List[str]):
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=HF_MODEL, device= -1)  # CPU by default
    out = [clf(t)[0] for t in texts]
    return out
 
def parse_hf_output(hf_item) -> Tuple[str, float]:
    """
    Parse HF output for single input (list of label/score dicts) into (label, score).
    Returns label normalized to POSITIVE/NEGATIVE/NEUTRAL and top score.
    """
    try:
        # hf_item expected like [{'label':'LABEL_0','score':0.8}, ...] or list of dicts
        if isinstance(hf_item, list):
            best = max(hf_item, key=lambda x: x.get("score", 0))
            lab = best.get("label", "")
            scr = float(best.get("score", 0.0))
        elif isinstance(hf_item, dict):
            # sometimes returns dict for single text
            lab = hf_item.get("label","")
            scr = float(hf_item.get("score",0.0))
        else:
            return "NEUTRAL", 0.0
    except Exception:
        return "NEUTRAL", 0.0
 
    # normalize label mapping (model-specific)
    lab_clean = lab.upper()
    if "NEG" in lab_clean:
        return "negative", scr
    if "POS" in lab_clean:
        return "positive", scr
    if "NEU" in lab_clean or "NEUTRAL" in lab_clean:
        return "neutral", scr
    # fallback: inspect label words
    if "NEGATIVE" in lab_clean:
        return "negative", scr
    if "POSITIVE" in lab_clean:
        return "positive", scr
    return lab.lower(), scr
 
# Keyword extraction
rake_extractor = Rake()
def extract_keywords(text: str, max_keywords=10) -> List[Tuple[str,float]]:
    rake_extractor.extract_keywords_from_text(text)
    ranked = rake_extractor.get_ranked_phrases_with_scores()
    return ranked[:max_keywords]
 
# Explainability: leave-one-out token importance
def token_importance_leave_one_out(text: str, classify_fn, base_prob: float, base_label: str) -> List[Tuple[str, float]]:
    # naive tokenization by whitespace — fast and sufficient for explainability
    tokens = text.split()
    importance = []
    for i, _ in enumerate(tokens):
        masked = " ".join(tokens[:i] + tokens[i+1:])
        try:
            pred = classify_fn([masked])[0]
            label, score = parse_hf_output(pred)
            # we use probability of the base_label to compute drop
            # if parse_hf_output returns only top label, we will approximate by comparing top label match:
            if label == base_label:
                importance.append((tokens[i], base_prob - score))
            else:
                # if base_label disappears, give a stronger positive importance
                importance.append((tokens[i], base_prob))
        except Exception:
            importance.append((tokens[i], 0.0))
    # sort by importance descending
    importance_sorted = sorted(importance, key=lambda x: x[1], reverse=True)
    return importance_sorted
 
# Wrapper classify function that returns raw HF output for a list of texts
def classify_texts(texts: List[str]) -> List[Tuple[str,float,Dict]]:
    if USE_HF_API:
        raw = call_hf_inference(texts)
    else:
        raw = local_classify(texts)
    parsed = []
    for item in raw:
        lab, score = parse_hf_output(item)
        parsed.append((lab, score, item))
    return parsed
 
# CSV/JSON export helpers
def df_to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')
 
def df_to_json_bytes(df: pd.DataFrame):
    return df.to_json(orient="records", force_ascii=False).encode('utf-8')
 
def create_pdf_report(df: pd.DataFrame):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Sentiment Analysis Report", ln=True)
    pdf.ln(4)
    for i, row in df.iterrows():
        text = str(row.get("text", ""))[:400]
        pdf.multi_cell(0, 6, f"{i+1}. [{row.get('predicted_label')}] ({row.get('confidence'):.2f}) {text}")
        pdf.ln(1)
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer
 
# --- Streamlit UI -----------------------------------------------------------
st.title("Sentiment Analysis Dashboard — CAPACITI Tech Career Accelerator")
st.markdown("Analyze sentiment in single texts or batches. Multi-class (positive/neutral/negative), confidence scores, keyword extraction, token-level explainability, visualizations, and exports.")
 
# Sidebar config
st.sidebar.header("Settings")
batch_size = st.sidebar.number_input("Batch size (API requests)", min_value=1, max_value=128, value=8)
st.sidebar.write("Model:")
st.sidebar.write(HF_MODEL if USE_HF_API else "Local transformers")
 
# Input area
st.header("Input")
col1, col2 = st.columns([2,1])
 
with col1:
    text_input = st.text_area("Enter text to analyze (or leave empty to upload file)", height=150)
    uploaded_file = st.file_uploader("Upload CSV or JSON file (single column 'text' or list of objects)", type=['csv','json','txt'])
    if uploaded_file:
        st.write("File uploaded:", uploaded_file.name)
with col2:
    st.info("Tip: For explainability, use shorter texts (< 80 words) for faster leave-one-out analysis.")
    st.button("Run analysis", key="run_button")
 
# Prepare texts list
texts = []
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_in = pd.read_csv(uploaded_file)
            if 'text' not in df_in.columns:
                # try first column
                df_in['text'] = df_in.iloc[:,0].astype(str)
            texts = df_in['text'].astype(str).tolist()
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            if isinstance(data, list):
                texts = [str(item.get('text', item)) for item in data]
            else:
                texts = [str(data.get('text', ""))]
        else:
            content = uploaded_file.read().decode('utf-8')
            texts = content.splitlines()
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
elif text_input.strip():
    texts = [text_input.strip()]
 
if not texts:
    st.warning("No input provided. Enter text or upload a file to continue.")
    st.stop()
 
# Limit for UI demo to avoid huge bills — but allow user to proceed
if len(texts) > 500:
    st.warning("Large upload detected. Processing many texts may take time and cost API credits. Proceeding anyway.")
 
# Classify
with st.spinner("Classifying..."):
    results = classify_texts(texts)
 
# Build results dataframe
rows = []
for t, (lab, score, raw) in zip(texts, results):
    kws = extract_keywords(t, max_keywords=6)
    rows.append({
        "text": t,
        "predicted_label": lab,
        "confidence": score,
        "keywords": "; ".join([k for _,k in kws]) if isinstance(kws, list) else ""
    })
df_results = pd.DataFrame(rows)
 
# Show table and downloads
st.header("Results")
st.dataframe(df_results[['predicted_label','confidence','keywords','text']].rename(columns={"predicted_label":"label","confidence":"score"}), use_container_width=True)
 
csv_bytes = df_to_csv_bytes(df_results)
json_bytes = df_to_json_bytes(df_results)
pdf_buf = create_pdf_report(df_results)
 
st.download_button("Download CSV", csv_bytes, file_name="sentiment_results.csv", mime="text/csv")
st.download_button("Download JSON", json_bytes, file_name="sentiment_results.json", mime="application/json")
st.download_button("Download PDF report", pdf_buf, file_name="sentiment_report.pdf", mime="application/pdf")
 
# Visualizations
st.header("Visualizations")
colA, colB = st.columns(2)
 
with colA:
    st.subheader("Sentiment distribution")
    chart_df = df_results['predicted_label'].value_counts().rename_axis('label').reset_index(name='count')
    st.bar_chart(chart_df.set_index('label'))
 
with colB:
    st.subheader("Confidence histogram")
    st.histogram = None
    st.write(df_results['confidence'].describe())
    st.pyplot()  # placeholder in case you add matplotlib hist
 
# Comparative analysis
st.header("Compare texts")
if len(texts) >= 2:
    sel = st.multiselect("Select indices to compare (0-based)", options=list(range(len(texts))), default=[0, min(1, len(texts)-1)])
    if len(sel) >= 2:
        comp = df_results.loc[sel]
        st.write(comp[['predicted_label','confidence','keywords','text']])
else:
    st.info("Upload or enter at least 2 texts to use comparative analysis.")
 
# Explainability for a selected sample
st.header("Explainability (token importance)")
sample_idx = st.number_input("Sample index for explainability", min_value=0, max_value=len(texts)-1, value=0, step=1)
sample_text = texts[sample_idx]
sample_label, sample_conf, _ = results[sample_idx]
st.markdown(f"**Predicted:** {sample_label} — confidence {sample_conf:.3f}")
st.write(sample_text[:1000])
 
if st.button("Compute token importance (leave-one-out)"):
    with st.spinner("Computing... (this may be slow for long texts)"):
        def classify_fn_local(batch):
            # wrapper that calls HF API or local and returns raw outputs
            if USE_HF_API:
                return call_hf_inference(batch)
            else:
                return local_classify(batch)
        importance = token_importance_leave_one_out(sample_text, classify_fn_local, sample_conf, sample_label)
        st.table(pd.DataFrame(importance, columns=["token","importance"]).head(20))
 
# Evaluation mode (for 50-sample accuracy test)
st.header("Evaluation (accuracy report)")
with st.expander("Run evaluation on a labeled CSV (columns: text, label)"):
    eval_file = st.file_uploader("Upload labeled CSV for evaluation", type=['csv'], key="eval_csv")
    if eval_file:
        eval_df = pd.read_csv(eval_file)
        if 'text' not in eval_df.columns or 'label' not in eval_df.columns:
            st.error("CSV must have 'text' and 'label' columns.")
        else:
            if st.button("Run evaluation"):
                texts_eval = eval_df['text'].astype(str).tolist()
                true_labels = eval_df['label'].astype(str).tolist()
                with st.spinner("Classifying evaluation texts..."):
                    preds = classify_texts(texts_eval)
                    pred_labels = [p[0] for p in preds]
                acc = accuracy_score(true_labels, pred_labels)
                st.write("Accuracy:", acc)
                st.write("Classification report:")
                st.text(classification_report(true_labels, pred_labels, digits=3))
                cm = confusion_matrix(true_labels, pred_labels, labels=["positive","neutral","negative"])
                st.write("Confusion matrix (rows=true, cols=pred):")
                st.write(pd.DataFrame(cm, index=["pos","neu","neg"], columns=["pos","neu","neg"]))
 
st.markdown("---")
st.caption("Notes: This app uses leave-one-out token importance for local explainability. For production, consider caching classifier results and using faster tokenizers. Keep an eye on API usage and model limits.")
