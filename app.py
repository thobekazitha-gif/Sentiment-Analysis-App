import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
 
# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
 
# ---------------------- Styling ----------------------
st.set_page_config(page_title="Sentiment Insights", layout="wide")
 
page_style = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0e0d0d;
    color: #ffffff;
}
h1, h2, h3, h4, h5, h6, p {
    color: #f2f2f2;
}
.block-container {
    background: rgba(20, 20, 20, 0.85);
    padding: 2rem;
    border-radius: 12px;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)
 
# ---------------------- App Title ----------------------
st.title("Sentiment Insights")
st.subheader("Understand reviews for any business — restaurants, retail, dealerships, and more.")
st.markdown("🚀 Powered by **Logic League** | Gain instant insights from customer reviews using AI.")
 
# ---------------------- Inputs ----------------------
with st.container():
    st.markdown("### Enter your business details")
    business_name = st.text_input("Business / Location Name", placeholder="e.g., Casa d'Tia, Uptown Auto, Fashion Square Mall")
    reviews_input = st.text_area(
        "Paste customer reviews (one per line)",
        placeholder="The food was disappointing\nService was excellent and staff were friendly\nPrice is reasonable but wait time was long",
        height=150
    )
 
# ---------------------- Processing ----------------------
if st.button("Analyze Reviews"):
    if business_name.strip() == "" or reviews_input.strip() == "":
        st.error("⚠️ Please enter a business name and at least one review.")
    else:
        reviews = [r.strip() for r in reviews_input.split("\n") if r.strip()]
        results = sentiment_pipeline(reviews)
 
        # Sentiment counters
        summary_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        detailed_results = []
 
        for r, res in zip(reviews, results):
            label = res['label'].upper()
            score = round(res['score'], 2)
            if label not in summary_counts:
                summary_counts[label] = 0
            summary_counts[label] += 1
            detailed_results.append((r, label, score))
 
        # ---------------------- Dashboard ----------------------
        st.header("📊 Analysis Results")
        st.markdown(f"**Business:** {business_name}")
 
        # Summary description
        st.markdown("### 📝 Summary")
        total = len(reviews)
        pos = summary_counts["POSITIVE"]
        neg = summary_counts["NEGATIVE"]
        neu = summary_counts.get("NEUTRAL", 0)
 
        st.markdown(
            f"- Total Reviews: **{total}**\n"
            f"- Positive: **{pos}**\n"
            f"- Negative: **{neg}**\n"
            f"- Neutral: **{neu}**\n\n"
            f"➡️ Overall, this business has **{round((pos/total)*100,1)}% positive sentiment**, "
            f"with key strengths and weaknesses visible in customer feedback."
        )
 
        # Sentiment distribution chart
        fig, ax = plt.subplots()
        labels = list(summary_counts.keys())
        values = list(summary_counts.values())
        ax.bar(labels, values, color=["#4CAF50", "#F44336", "#FFC107"])
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)
 
        # ---------------------- Detailed Table ----------------------
        st.markdown("### Detailed Review Insights")
        for rev, lab, sc in detailed_results:
            color = "🟢" if lab == "POSITIVE" else "🔴" if lab == "NEGATIVE" else "🟡"
            st.markdown(f"- {color} **{lab} ({sc})** → {rev}")
