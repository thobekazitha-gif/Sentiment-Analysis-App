# Sentiment Insights — Interactive Dashboard

## What this is
A Streamlit app that performs multi-class sentiment analysis (Positive / Negative / Neutral) using Hugging Face Inference API (default) with a VADER fallback. It includes confidence scoring, keyword extraction (RAKE), LIME explanations, visualizations, batch processing, and exports (CSV/JSON/PDF).

## Quick start (local)
1. Clone repo.
2. Create virtual env and install:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
3. Set env var for Hugging Face:
   export HF_API_KEY="your_hf_token"
4. Run app:
   streamlit run app.py

## Deployment
- Streamlit Cloud: push to GitHub, create a new app in Streamlit Cloud pointing at `app.py`. In Streamlit Cloud, define `HF_API_KEY` in secrets.
- Docker: (optional) Dockerfile that installs dependencies + runs `streamlit run app.py`.

## API integration
- Default uses HF model `cardiffnlp/twitter-roberta-base-sentiment-latest` (changeable in `nlp_utils.py`).
- If HF API not available, the app falls back to VADER.

## Evaluation (accuracy report)
See `evaluation/evaluate_accuracy.py` to compute confusion matrix and metrics on a labeled dataset (minimum 50 samples). The evaluation output includes a confusion matrix and CSV of predictions vs manual labels.

## Known limitations (short)
- HF inference costs and rate limits; batched requests are throttled.
- Some models were trained on tweets and may misinterpret domain-specific phrases.
- LIME is approximate and can be slow for large inputs.
