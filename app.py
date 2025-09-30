import streamlit as st
 
# ----------------- Page Config -----------------
st.set_page_config(page_title="Logic League Sentiment Analysis", layout="wide")
 
# ----------------- Custom CSS -----------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f5f5f5;
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #8e2de2, #4a00e0, #00c6ff, #ff758c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        font-weight: 400;
        color: #dcdcdc;
        text-align: center;
        margin-bottom: 3rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 2rem;
        margin: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0px 8px 30px rgba(0,0,0,0.8);
    }
    .card h3 {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
    }
    .card p {
        color: #e0e0e0;
        font-size: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        font-size: 1.1rem;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff758c, #ff7eb3);
        color: #fff;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
# ----------------- Landing Page -----------------
st.markdown("<h1 class='main-title'>Logic League Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>Harness AI to unlock insights from customer reviews across restaurants, retail, dealerships, and beyond.</p>",
    unsafe_allow_html=True
)
 
# Feature Cards
col1, col2, col3 = st.columns(3)
 
with col1:
    st.markdown(
        """
        <div class='card'>
            <h3>⚡ Instant Results</h3>
            <p>Upload reviews or type them in. Get fast, accurate sentiment detection: Positive, Negative, or Neutral.</p>
        </div>
        """, unsafe_allow_html=True)
 
with col2:
    st.markdown(
        """
        <div class='card'>
            <h3>🌍 Multi-Industry</h3>
            <p>Analyze feedback for restaurants, retail, car dealerships, or any business with customer interactions.</p>
        </div>
        """, unsafe_allow_html=True)
 
with col3:
    st.markdown(
        """
        <div class='card'>
            <h3>📊 Interactive Dashboard</h3>
            <p>Track sentiment trends, compare multiple locations, and gain insights into customer experiences.</p>
        </div>
        """, unsafe_allow_html=True)
 
# Call to Action
st.markdown("### Ready to explore your reviews?")
if st.button("🚀 Go to Dashboard"):
    st.switch_page("pages/dashboard.py")  # requires you to create a `pages/dashboard.py`
