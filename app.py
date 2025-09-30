# app.py
"""
Sentiment Analysis Dashboard (single-file)
- Multi-page: Landing, Input, Dashboard
- Hugging Face Inference API if HF_API_TOKEN present (robust handling)
- Local fallback sentiment analyzer
- NLTK-free keyword extraction and explanation
- CSV / JSON / PDF export
- Simple aesthetic CSS (beige/white/sky-blue)
"""
 
import os
import io
import json
import tempfile
import streamlit as st
import pandas as pd
import requests
import re
from collections import Counter
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
 
# ---------------------------
# CONFIG
# ---------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Set in environment or Streamlit secrets
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else None
BATCH_SIZE = 16
 
# Light lexicons for local fallback and explanations
POSITIVE_LEXICON = set(["good","great","love","excellent","happy","awesome","delicious","satisfied","recommend","best","nice","pleased","amazing"])
NEGATIVE_LEXICON = set(["bad","terrible","hate","poor","sad","awful","disappoint","disappointed","worst","angry","broken","slow","rude"])
 
# ---------------------------
# Styling
# ---------------------------
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7f6f3 0%, #eaf6ff 100%); }
    .hero { padding: 2rem; border-radius: 12px; background: linear-gradient(90deg, #ffffffcc, #f0fbffcc); }
    .card { background: rgba(255,255,255,0.9); border-radius: 10px; padding: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
    .keyword { display:inline-block; padding:4px 8px; margin:4px; border-radius:12px; background:#e6f7ff; color:#036; font-weight:600; }
    .neg { background:#ffe6e6; color:#880000; }
    .pos { background:#e6ffec; color:#006600; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ---------------------------
# Utility functions
# ---------------------------
def call_hf_api(texts):
    """Call HF Inference API in batches. Returns list of parsed dicts with pos/neu/neg (floats)."""
    results = []
    if not HF_API_TOKEN:
        return None
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        payload = {"inputs": batch, "options": {"wait_for_model": True}}
        try:
            resp = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            # Expect out to be list per input; some models return single list per each
            for entry in out:
                # entry is list of label/score dicts OR an error dict
                if isinstance(entry, list):
                    # Normalize labels -> lowercase keys: positive/neutral/negative
                    parsed = {}
                    for item in entry:
                        lab = str(item.get("label","")).lower()
                        score = float(item.get("score", 0.0))
                        if "pos" in lab: parsed["positive"] = score
                        elif "neu" in lab: parsed["neutral"] = score
                        elif "neg" in lab: parsed["negative"] = score
                    # guarantee keys
                    results.append({
                        "positive": parsed.get("positive", 0.0),
                        "neutral": parsed.get("neutral", 0.0),
                        "negative": parsed.get("negative", 0.0),
                    })
                else:
                    # Unexpected response, fallback neutral
                    results.append({"positive":0.0,"neutral":1.0,"negative":0.0})
        except Exception as e:
            # On any API failure, return None to let caller use fallback
            st.warning(f"Hugging Face API error: {e}")
            return None
    return results
 
def analyze_local(texts):
    """Simple rule-based sentiment per text; returns list of dicts same format as HF API parsing."""
    out = []
    for t in texts:
        low = t.lower()
        pos_score = sum(low.count(w) for w in POSITIVE_LEXICON)
        neg_score = sum(low.count(w) for w in NEGATIVE_LEXICON)
        if pos_score + neg_score == 0:
            out.append({"positive":0.0,"neutral":1.0,"negative":0.0})
        else:
            # normalize to probabilities
            total = pos_score + neg_score
            out.append({
                "positive": pos_score/total if pos_score>0 else 0.0,
                "neutral": 0.0,
                "negative": neg_score/total if neg_score>0 else 0.0
            })
    return out
 
def extract_keywords_simple(text, max_k=6):
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    stop = set(["the","and","is","in","it","of","to","a","for","on","with","that","this","i","was","but","are","my","so","at","as","be","an","they","or","its","very"])
    filtered = [w for w in words if w not in stop and len(w) > 2]
    counts = Counter(filtered)
    return [w for w,_ in counts.most_common(max_k)]
 
def explain_text(text, sentiments, keywords):
    """
    Produce a short explanation string and a list of "drivers" (keywords or lexicon matches).
    """
    drivers = []
    low = text.lower()
    for k in keywords:
        if k in low:
            drivers.append(k)
    # lexicon matches
    pos_hits = [w for w in POSITIVE_LEXICON if w in low]
    neg_hits = [w for w in NEGATIVE_LEXICON if w in low]
    drivers.extend(pos_hits + neg_hits)
    # Build explanation
    top_label = max(("positive", sentiments["positive"]), ("neutral", sentiments["neutral"]), ("negative", sentiments["negative"]), key=lambda x: x[1])
    label = top_label[0]
    score = top_label[1]
    expl = f"Predicted **{label.capitalize()}** (confidence {score:.0%})."
    if drivers:
        expl += " Key drivers: " + ", ".join(drivers[:6]) + "."
    else:
        expl += " No clear keywords found; fallback to general tone."
    return expl, drivers
 
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')
 
def df_to_json_bytes(df):
    return df.to_json(orient="records", force_ascii=False).encode('utf-8')
 
def create_pdf(df, business_name):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Sentiment Analysis Report - {business_name}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    for i, row in df.iterrows():
        pdf.multi_cell(0, 6, f"{i+1}. [{row['Top_Label'].upper()} {row['Top_Score']:.0%}] Keywords: {row['Keywords']}")
        pdf.multi_cell(0, 6, f"    \"{row['Text'][:300]}\"")
        pdf.ln(1)
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf
 
# ---------------------------
# App UI: multipage via sidebar
# ---------------------------
PAGES = ["Landing", "Input", "Dashboard"]
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", PAGES)
 
# Persistent storage (simple) in session_state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Text","Positive","Neutral","Negative","Keywords","Top_Label","Top_Score","Drivers"])
 
# ---------------------------
# Landing page
# ---------------------------
if page == "Landing":
    st.markdown('<div class="hero card">', unsafe_allow_html=True)
    st.markdown("<h1 style='color:#03396c'>Sentiment Insights</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#024b7a'>Understand reviews for any business — restaurants, retail, dealerships and more.</h3>", unsafe_allow_html=True)
    st.markdown("<p>Enter a business name, upload reviews, or paste a review to begin. The Dashboard tab holds detailed results.</p>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        business = st.text_input("Business / Location name (e.g., Casa d' Tia, Grid & Grill, Uptown Auto)", value=st.session_state.get("business", "Your Business Name"))
        st.session_state.business = business
        st.text_area("Show similar reviews (example):", value="The food was disappointing\nService was excellent and staff were friendly\nPrice is reasonable but wait time was long", height=120)
        st.markdown("**Aesthetic visuals:**")
        st.image("https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=1200&q=80", caption="Aesthetic placeholder", use_column_width=True)
    with col2:
        st.markdown("**Quick actions**")
        if st.button("Go to Input"):
            st.experimental_set_query_params(page="Input")
            st.experimental_rerun()
        st.markdown("**Color palette suggestion:** Beige background, white content cards, sky-blue accents.")
    st.markdown('</div>', unsafe_allow_html=True)
 
# ---------------------------
# Input page
# ---------------------------
elif page == "Input":
    st.header("Upload or Type Reviews")
    st.markdown("You can paste one review or upload a `.txt` (one review per line) or `.csv` with a `text` column.")
    with st.form("input_form"):
        text_input = st.text_area("Paste a review (single) or leave empty for upload", height=120)
        uploaded_file = st.file_uploader("Upload .txt or .csv (text column)", type=["txt","csv"])
        use_hf = st.checkbox("Use Hugging Face API (if token is configured)", value=True)
        submit = st.form_submit_button("Analyze")
    if submit:
        texts = []
        if uploaded_file:
            name = uploaded_file.name.lower()
            try:
                if name.endswith(".txt"):
                    content = uploaded_file.read().decode("utf-8").splitlines()
                    texts = [line for line in content if line.strip()]
                else:
                    df_upload = pd.read_csv(uploaded_file)
                    if "text" in df_upload.columns:
                        texts = df_upload["text"].astype(str).tolist()
                    else:
                        # try first column
                        texts = df_upload.iloc[:,0].astype(str).tolist()
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                texts = []
        if text_input and not uploaded_file:
            texts = [text_input.strip()]
        if not texts:
            st.warning("No text provided.")
        else:
            # Attempt HF if requested and token exists
            hf_results = None
            if use_hf and HF_API_TOKEN:
                hf_results = call_hf_api(texts)
            if hf_results is None:
                st.info("Using local fallback analyzer (no HF or API failed).")
                hf_results = analyze_local(texts)
            # Build DataFrame and explanations
            rows = []
            for t, sent in zip(texts, hf_results):
                keywords = extract_keywords_simple(t, max_k=6)
                expl, drivers = explain_text(t, sent, keywords)
                # Top label and score
                top_label = max(("positive", sent["positive"]), ("neutral", sent["neutral"]), ("negative", sent["negative"]), key=lambda x: x[1])
                rows.append({
                    "Text": t,
                    "Positive": sent["positive"],
                    "Neutral": sent["neutral"],
                    "Negative": sent["negative"],
                    "Keywords": ", ".join(keywords),
                    "Top_Label": top_label[0],
                    "Top_Score": top_label[1],
                    "Drivers": ", ".join(drivers),
                    "Explanation": expl
                })
            df_new = pd.DataFrame(rows)
            # Append to session_state
            st.session_state.data = pd.concat([st.session_state.data, df_new], ignore_index=True)
            st.success(f"Analyzed {len(df_new)} review(s) and added to Dashboard.")
            st.experimental_set_query_params(page="Dashboard")
            st.experimental_rerun()
 
# ---------------------------
# Dashboard page
# ---------------------------
elif page == "Dashboard":
    st.header("Analysis Results")
    st.markdown(f"**Business:** {st.session_state.get('business','(not set)')}")
    df = st.session_state.data.copy()
    if df.empty:
        st.info("No analysis results yet — go to Input and analyze some reviews.")
    else:
        # Top KPI: counts by Top_Label
        counts = df["Top_Label"].value_counts().reindex(["positive","neutral","negative"]).fillna(0).astype(int)
        col1, col2, col3, col4 = st.columns([2,3,3,3])
        with col1:
            st.metric("Total reviews", len(df))
        with col2:
            st.metric("Positive", counts.get("positive",0))
        with col3:
            st.metric("Neutral", counts.get("neutral",0))
        with col4:
            st.metric("Negative", counts.get("negative",0))
        st.markdown("---")
        # Charts
        st.subheader("Sentiment Breakdown")
        breakdown = pd.DataFrame({
            "label": ["Positive","Neutral","Negative"],
            "count": [counts.get("positive",0), counts.get("neutral",0), counts.get("negative",0)]
        })
        fig = px.bar(breakdown, x="label", y="count", color="label", color_discrete_map={"Positive":"#2ca02c","Neutral":"#7f7f7f","Negative":"#d62728"})
        st.plotly_chart(fig, use_container_width=True)
        # Intensity distribution (use Top_Score numeric)
        st.subheader("Intensity Distribution (Top score per review)")
        fig2 = px.histogram(df, x="Top_Score", nbins=10, title="Confidence/Intensity distribution")
        st.plotly_chart(fig2, use_container_width=True)
        # Show individual result cards
        st.subheader("Individual Reviews")
        for idx, row in df.iterrows():
            label = row["Top_Label"]
            score = row["Top_Score"]
            k = row["Keywords"]
            drivers = row.get("Drivers","")
            colA, colB = st.columns([9,3])
            with colA:
                box_css = "pos" if label=="positive" else ("neg" if label=="negative" else "")
                st.markdown(f"<div class='card' style='border-left:6px solid {'#2ca02c' if label=='positive' else ('#d62728' if label=='negative' else '#9b9b9b')};'>", unsafe_allow_html=True)
                st.markdown(f"**{label.capitalize()}** — Confidence: {score:.0%}")
                st.markdown(f"> {row['Text']}")
                st.markdown(f"**Explanation:** {row.get('Explanation','')}")
                if drivers:
                    st.markdown("**Drivers:** " + ", ".join([d.strip() for d in drivers.split(",") if d.strip()]))
                st.markdown("</div>", unsafe_allow_html=True)
            with colB:
                if k:
                    st.markdown("<div>", unsafe_allow_html=True)
                    for kw in k.split(","):
                        kw = kw.strip()
                        if kw:
                            st.markdown(f"<span class='keyword'>{kw}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        # Exports
        st.subheader("Export results")
        csv_bytes = df_to_csv_bytes(df)
        json_bytes = df_to_json_bytes(df)
        pdf_buf = create_pdf(df, st.session_state.get("business","Business"))
        colx1, colx2, colx3 = st.columns(3)
        with colx1:
            st.download_button("Download CSV", csv_bytes, file_name="sentiment_results.csv", mime="text/csv")
        with colx2:
            st.download_button("Download JSON", json_bytes, file_name="sentiment_results.json", mime="application/json")
        with colx3:
            st.download_button("Download PDF", pdf_buf, file_name="sentiment_report.pdf", mime="application/pdf")
        # Clear / reset
        if st.button("Clear all results"):
            st.session_state.data = pd.DataFrame(columns=["Text","Positive","Neutral","Negative","Keywords","Top_Label","Top_Score","Drivers"])
            st.success("Dashboard cleared.")
 
