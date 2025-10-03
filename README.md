# Sentiment Insights — Interactive Dashboard

## Overview
**Sentiment Insights** is a Python-based sentiment analysis project that uses the **Hugging Face Inference API** to classify text into **Positive**, **Negative**, or **Neutral**.  

Features include:

- Multi-class sentiment predictions with **confidence scores**.
- **Keyword extraction** using RAKE.
- **Explainability** via LIME (local token-level explanations).
- Batch processing for CSV datasets.
- Export options: CSV, JSON, PDF.
- Local fallback using **VADER** if Hugging Face API is unavailable.
- Evaluation metrics including **confusion matrix** and **classification report**.

This setup does **not require Colab** and is fully GitHub- and Hugging Face–ready.

---

## Quick Start (Local)

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd sentiment-insights
