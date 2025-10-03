Streamlit Cloud (recommended for simple deployment)

Push repo to GitHub.

Go to Streamlit Cloud, create new app linking your repo and app.py.

Add secret HF_API_KEY in Streamlit secrets (do not commit key).

Set branch & launch.

Hugging Face Spaces

Create a Space with streamlit runtime.

Upload repo or link GitHub.

Add HF token if you need privileged keys (Spaces have restrictions — you may use free hosted models without token sometimes).

Docker (self-hosting)

Write Dockerfile:

FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]


Build and run: docker build -t sentiment-app . then docker run -p 8501:8501 -e HF_API_KEY="hf_xxx" sentiment-app

Production considerations

Use a hosted inference model (self-hosted transformer) for privacy/cost control for large volumes.

Add rate-limiting, retries with exponential backoff, and monitoring (Prometheus/CloudWatch).
