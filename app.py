
import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import re

# Load model
with open("model_lr.pkl", "rb") as f:
    model = pickle.load(f)

# Load BERT
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("🧠 Fake News Detector")
tweet = st.text_area("Enter Tweet")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter something.")
    else:
        clean = preprocess(tweet)
        vector = bert_model.encode([clean])
        pred = model.predict(vector)[0]
        label = "✅ Real (1)" if pred == 1 else "🚫 Fake (0)"
        st.success(f"Prediction: {label}")
