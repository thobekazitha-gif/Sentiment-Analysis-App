# evaluation/evaluate_accuracy.py
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from nlp_utils import predict_batch_hf, predict_batch_vader

DATA_PATH = "evaluation/sample_manual_labels.csv"
hf_api_key = os.environ.get("HF_API_KEY")

def load_or_generate_sample(path=DATA_PATH):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "text" not in df.columns or "label" not in df.columns:
            raise RuntimeError("CSV must contain 'text' and 'label' columns.")
        return df
    # generate synthetic 50+ sample (balanced-ish)
    texts = []
    labels = []
    pos = [
        "The service was excellent and I loved the food.",
        "Absolutely fantastic experience, will come back!",
        "High quality and great value."
    ]
    neg = [
        "Terrible service, never coming back.",
        "Food was cold and staff were rude.",
        "Very disappointed with the experience."
    ]
    neu = [
        "The place was okay, nothing special.",
        "It was fine overall; neither good nor bad.",
        "Average experience, nothing remarkable."
    ]
    for i in range(18):
        texts.append(pos[i % len(pos)])
        labels.append("POSITIVE")
    for i in range(17):
        texts.append(neg[i % len(neg)])
        labels.append("NEGATIVE")
    for i in range(16):
        texts.append(neu[i % len(neu)])
        labels.append("NEUTRAL")
    df = pd.DataFrame({"text": texts, "label": labels})
    os.makedirs("evaluation", exist_ok=True)
    df.to_csv(path, index=False)
    return df

def run_evaluation():
    df = load_or_generate_sample()
    texts = df['text'].tolist()
    try:
        preds = predict_batch_hf(texts, api_key=hf_api_key)
        print("Used Hugging Face for predictions.")
    except Exception as e:
        print(f"HF error: {e}. Falling back to VADER.")
        preds = predict_batch_vader(texts)
    pred_labels = [p['label'] for p in preds]
    df['predicted'] = pred_labels
    os.makedirs("evaluation", exist_ok=True)
    df.to_csv("evaluation/predictions_vs_manual.csv", index=False)
    print("Saved predictions to evaluation/predictions_vs_manual.csv")
    print("\nClassification report:")
    print(classification_report(df['label'], df['predicted'], digits=3))
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(df['label'], df['predicted'], labels=["POSITIVE","NEGATIVE","NEUTRAL"])
    print(cm)

if __name__ == "__main__":
    run_evaluation()
