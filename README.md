Sentiment Analysis Dashboard
 
An interactive web app that analyzes sentiment in text data — customer reviews, social media posts, or any text source.  
Built with Streamlit and integrates with the Hugging Face Inference API for state-of-the-art sentiment classification.  
 

 
 Features
- Text input: Analyze a single text or upload a file (CSV/JSON/TXT).
- Multi-class classification: Positive, Neutral, Negative.
- Confidence scoring: See how certain the model is.
- Keyword extraction: Highlights key phrases driving sentiment.
- Batch processing: Handle multiple texts at once.
- Visualizations: Sentiment distribution charts & confidence stats.
- Comparative analysis: Compare multiple texts side by side.
- Explainability: Token-level importance (why a text got its sentiment).
- Export results: CSV, JSON, PDF reports.
- Evaluation mode: Upload labeled data, generate confusion matrix & metrics.
 
 
 Tech Stack
- [Streamlit](https://streamlit.io/) — interactive UI
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/) — sentiment model
- [RAKE](https://pypi.org/project/rake-nltk/) — keyword extraction
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics
- [FPDF](https://pyfpdf.readthedocs.io/) — PDF export
 

 
    Getting Started 
    1. Clone the repository
bash
•	git clone https://github.com/<your-username>/sentiment-dashboard.git
cd sentiment-dashboard
 
    2. Install dependencies 
bash
•	pip install -r requirements.txt
 
    3. Set your Hugging Face API Token 
You need a free Hugging Face account and token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
 
•	macOS/Linux   
  bash
  export HF_API_TOKEN="hf_yourtokenhere"

•	Windows (PowerShell)   
  bash
  setx HF_API_TOKEN "hf_yourtokenhere"  
 
    4. Run the application 
bash
streamlit run app.py 
  
 
 Deployment (Streamlit Cloud) 
1. Push your repo to GitHub.
2. Go to Streamlit Cloud (https://share.streamlit.io) and create a new app.
3. Select your repo and set the main file to `app.py`.
4. Add your Hugging Face token in   Secrets   as: 
   ini
   HF_API_TOKEN = "hf_yourtokenhere"
  
5. Deploy 
 
  
 
  Accuracy & Evaluation 
  At least   50 sample texts   were manually labeled and compared with API outputs.
  Evaluation includes: 
•	Confusion matrix
•	Precision, Recall, F1-score
•	Discussion of limitations and misclassifications
  The full   accuracy report   is included in /docs/accuracy_report.pdf. 
  
 
  Project Structure
sentiment-dashboard/
│── app.py                  Main Streamlit app
│── requirements.txt        Dependencies
│── utils/                  Helper functions
│── data/                   Sample data
│── docs/                   Reports & documentation
│── README.md               This file

 
   Limitations 
•	Struggles with sarcasm, slang, and mixed-language text
•	Performance may vary for very long documents
•	API-dependent: requires an internet connection
 
  
   Deliverables 
•	GitHub repo with full source code
•	 Deployed app (or deployment guide)
•	 Accuracy report (50+ texts with confusion matrix & discussion)
•	 3-minute demo video showcasing features
•	 Documentation: 
o	API selection justification
o	Implementation challenges
o	User guide with examples
 
 
  Author 
Developed by   Logic League   for the   Tech Career Accelerator CAPACITI   program.
Braamfontein, South Africa
  [capaciti.org.za](https://capaciti.org.za)
 

