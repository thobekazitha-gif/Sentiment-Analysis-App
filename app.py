# app.py
import os
import json
from io import BytesIO
import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF

from nlp_utils import (
    predict_batch_hf,
    predict_batch_vader,
    extract_keywords_rake,
    explain_with_lime
)

st.set_page_config(page_title="Sentiment Insights", layout="wide")
st.title("Sentiment Insights — Streamlit Dashboard")

# Sidebar
st.sidebar.header("Settings")
api_choice = st.sidebar.selectbox("Primary NLP", ["Hugging Face Inference API", "VADER (local fallback)"])
hf_api_key = os.environ.get("HF_API_KEY")
if api_choice.startswith("Hugging Face") and not hf_api_key:
    st.sidebar.warning("Set HF_API_KEY env var to enable Hugging Face. Using VADER fallback will still work.")
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=128, value=16)
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.5, 0.99, 0.7, 0.01)

# Tabs
tab1, tab2 = st.tabs(["Single Text Analysis", "Dataset Analysis"])

with tab1:
    st.subheader("Single Text Analysis")
    txt = st.text_area("Enter text to analyze", height=180)
    if st.button("Analyze text"):
        if not txt or not txt.strip():
            st.error("Enter text first.")
        else:
            try:
                if api_choice.startswith("Hugging Face") and hf_api_key:
                    preds = predict_batch_hf([txt], api_key=hf_api_key)
                else:
                    preds = predict_batch_vader([txt])
            except Exception as e:
                st.error(f"API error: {e}")
                st.info("Falling back to VADER.")
                preds = predict_batch_vader([txt])
            p = preds[0]
            st.markdown(f"**Label:** {p['label']}  \n**Confidence score:** {p['score']:.3f}")
            st.markdown("**Keywords (RAKE):**")
            st.write(", ".join(extract_keywords_rake(txt, n=8)))
            with st.expander("Show raw probabilities"):
                st.json(p.get("probs", {}))
            # explanation
            with st.expander("Explain prediction (LIME) — may take a few seconds"):
                try:
                    expl = explain_with_lime(txt, api_choice="hf" if api_choice.startswith("Hugging Face") else "vader", hf_api_key=hf_api_key)
                    st.table(pd.DataFrame(expl, columns=["token", "weight"]))
                except Exception as e:
                    st.error(f"Explanation error: {e}")

with tab2:
    st.subheader("Dataset Analysis")
    uploaded = st.file_uploader("Upload CSV (col named 'text' & optional 'label')", type=["csv"])
    sample_btn = st.button("Load sample 50-row dataset")
    if sample_btn:
        # create a small synthetic sample if no file
        sample_texts = [
            "The food was delicious and the staff were friendly.",
            "Waited 45 minutes and they got my order wrong — awful service.",
            "Decent value but the portion was small.",
        ]
        df_sample = pd.DataFrame({
            "text": sample_texts * 17  # ~51 rows
        })
        uploaded_df = df_sample
        st.session_state["uploaded_df"] = uploaded_df
        st.success("Sample loaded into session")
    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            st.session_state["uploaded_df"] = df
    if "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"]
        st.write(f"Dataset rows: {len(df)}")
        if st.button("Run batch analysis"):
            texts = df["text"].astype(str).tolist()
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                try:
                    if api_choice.startswith("Hugging Face") and hf_api_key:
                        preds = predict_batch_hf(batch, api_key=hf_api_key)
                    else:
                        preds = predict_batch_vader(batch)
                except Exception as e:
                    st.warning(f"Batch API error: {e} — falling back to VADER for this batch.")
                    preds = predict_batch_vader(batch)
                for t, p in zip(batch, preds):
                    kws = extract_keywords_rake(t, n=6)
                    results.append({"text": t, "label": p["label"], "score": p["score"], "probs": json.dumps(p.get("probs", {})), "keywords": ", ".join(kws)})
            df_results = pd.DataFrame(results)
            st.session_state["df_results"] = df_results
            st.success("Batch analysis completed")
    # show results and charts
    if "df_results" in st.session_state:
        res = st.session_state["df_results"]
        st.dataframe(res[["text", "label", "score", "keywords"]])
        # distribution
        dist = res['label'].value_counts().reset_index()
        dist.columns = ["label", "count"]
        fig = px.pie(dist, names='label', values='count', title='Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)
        # downloads
        st.download_button("Download CSV", data=res.to_csv(index=False).encode(), file_name="sentiment_results.csv")
        st.download_button("Download JSON", data=res.to_json(orient="records").encode(), file_name="sentiment_results.json")
        # PDF summary
        if st.button("Generate PDF summary (first 50 rows)"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Sentiment Analysis Summary", ln=True)
            pdf.cell(0, 8, f"Total texts: {len(res)}", ln=True)
            for i, r in res.head(50).iterrows():
                pdf.multi_cell(0, 6, f"{i}. [{r['label']}:{r['score']:.2f}] {r['text']}")
            out = BytesIO()
            pdf.output(out)
            out.seek(0)
            st.download_button("Download PDF", data=out, file_name="sentiment_summary.pdf", mime="application/pdf")

        st.subheader("Explain a row (LIME)")
        idx = st.number_input("Row index", min_value=0, max_value=len(res)-1, value=0)
        if st.button("Explain row"):
            txt = res.loc[idx, "text"]
            st.write(txt)
            with st.expander("Keywords"):
                st.write(res.loc[idx, "keywords"])
            with st.expander("LIME explanation"):
                try:
                    explanation = explain_with_lime(txt, api_choice="hf" if api_choice.startswith("Hugging Face") else "vader", hf_api_key=hf_api_key)
                    st.table(pd.DataFrame(explanation, columns=["token", "weight"]))
                except Exception as e:
                    st.error(f"Explanation error: {e}")
