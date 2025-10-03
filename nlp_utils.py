# nlp_utils.py
import os
import time
import requests
from typing import List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
from lime.lime_text import LimeTextExplainer

# ------------------ Sentiment Tools ------------------
analyzer = SentimentIntensityAnalyzer()

# Simple sentence tokenizer for RAKE to avoid punkt/punkt_tab issues
def simple_split_sentences(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]

rake = Rake(language='english', sentence_tokenizer=simple_split_sentences)

HF_API_URL = "https://api-inference.huggingface.co/models"
DEFAULT_HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# ---------- Hugging Face Inference ----------
def predict_batch_hf(texts: List[str], model: str = DEFAULT_HF_MODEL, api_key: str = None, sleep: float = 0.12) -> List[Dict]:
    if api_key is None:
        raise RuntimeError("Hugging Face API key not provided.")
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{HF_API_URL}/{model}"
    results = []
    for t in texts:
        payload = {"inputs": t}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            out = resp.json()
            if isinstance(out, dict) and out.get("error"):
                raise RuntimeError(out["error"])
            if isinstance(out, list):
                probs = {o["label"]: o["score"] for o in out}
                best = max(out, key=lambda x: x.get("score", 0))
                results.append({"label": best["label"], "score": float(best["score"]), "probs": probs})
            else:
                results.append({"label": "NEUTRAL", "score": 0.0, "probs": {}})
        else:
            raise RuntimeError(f"HF API error {resp.status_code}: {resp.text}")
        time.sleep(sleep)
    return results

# ---------- VADER Fallback ----------
def predict_batch_vader(texts: List[str]) -> List[Dict]:
    results = []
    for t in texts:
        vs = analyzer.polarity_scores(t)
        compound = vs["compound"]
        if compound >= 0.05:
            label = "POSITIVE"
            score = compound
        elif compound <= -0.05:
            label = "NEGATIVE"
            score = -compound
        else:
            label = "NEUTRAL"
            score = 1 - abs(compound)
        probs = {"neg": vs["neg"], "neu": vs["neu"], "pos": vs["pos"], "compound": vs["compound"]}
        results.append({"label": label, "score": float(abs(score)), "probs": probs})
    return results

# ---------- RAKE Keyword Extraction ----------
def extract_keywords_rake(text: str, n: int = 6) -> List[str]:
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:n]

# ---------- LIME Explainer ----------
class _PredictWrapper:
    def __init__(self, api_choice="hf", hf_model=None, hf_api_key=None):
        self.api_choice = api_choice
        self.hf_model = hf_model
        self.hf_api_key = hf_api_key

    def predict_proba(self, texts):
        """
        Return probabilities as a 2D list of floats: [[POS, NEG, NEU], ...]
        """
        res = []
        if self.api_choice == "hf":
            try:
                hf_preds = predict_batch_hf(texts, model=self.hf_model or DEFAULT_HF_MODEL, api_key=self.hf_api_key)
            except Exception as e:
                hf_preds = predict_batch_vader(texts)
            for r in hf_preds:
                probs = r.get("probs", {})
                # Ensure order POS, NEG, NEU and default 0.0
                res.append([
                    float(probs.get("POSITIVE", probs.get("pos", 0.0))),
                    float(probs.get("NEGATIVE", probs.get("neg", 0.0))),
                    float(probs.get("NEUTRAL", probs.get("neu", 0.0)))
                ])
        else:
            vader_preds = predict_batch_vader(texts)
            for r in vader_preds:
                probs = r.get("probs", {})
                res.append([
                    float(probs.get("pos", 0.0)),
                    float(probs.get("neg", 0.0)),
                    float(probs.get("neu", 0.0))
                ])
        return res

def explain_with_lime(text, api_choice="hf", hf_api_key=None, hf_model=None, num_features=10):
    class_names = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    explainer = LimeTextExplainer(class_names=class_names)
    wrapper = _PredictWrapper(api_choice=api_choice, hf_model=hf_model, hf_api_key=hf_api_key)
    # labels=[0,1,2] to avoid tuple indexing errors
    exp = explainer.explain_instance(text, wrapper.predict_proba, num_features=num_features, labels=[0,1,2])
    # return explanation for POSITIVE only
    return exp.as_list(label=0)
