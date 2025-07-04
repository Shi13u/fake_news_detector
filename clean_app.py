#!/usr/bin/env python
# coding: utf-8

# ================================================
# PART 1: Load Data and Initial Preprocessing
# ================================================

from google.colab import drive
import pandas as pd
import numpy as np
import re
import nltk
import pickle

# Mount Google Drive
drive.mount('/content/drive')

# Load dataset
df = pd.read_csv("/content/drive/MyDrive/fake_news_detection/train (1).csv")

# Drop rows with missing 'text'
df = df.dropna(subset=['text'])

# Fill missing keyword/location with placeholders
df['keyword'] = df['keyword'].fillna('none')
df['location'] = df['location'].fillna('unknown')

# Display basic info
print("Data shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# ================================================
# PART 2: Text Cleaning and Normalization
# ================================================

# Download resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Combined text cleaning function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

# Apply cleaning
df['clean_text'] = df['text'].apply(preprocess_text)
df['normalized_text'] = df['clean_text'].apply(lemmatize_text)

# ================================================
# PART 3: Exploratory Data Analysis (EDA)
# ================================================

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Plot class distribution
sns.countplot(x='target', data=df)
plt.title('Distribution of Real (1) and Fake (0) Tweets')
plt.show()

# WordClouds
for label, title in zip([0, 1], ["Fake", "Real"]):
    words = ' '.join(df[df['target'] == label]['normalized_text'])
    wordcloud = WordCloud(width=1200, height=400, background_color='white').generate(words)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud - {title} Tweets')
    plt.show()

# ================================================
# PART 4: Feature Extraction - TF-IDF & BERT
# ================================================

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['normalized_text'])

# Save for inference
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# Optional: BERT embeddings via sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
X_bert = model.encode(df['normalized_text'].tolist(), batch_size=32, show_progress_bar=True)

# ================================================
# PART 5: Train-Test Split
# ================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_bert, df['target'], test_size=0.2, random_state=42, stratify=df['target'])
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, df['target'], test_size=0.2, random_state=42, stratify=df['target'])

# ================================================
# PART 6: Model Training
# ================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

lr_best = LogisticRegression(penalty='l2', solver='liblinear', C=1, max_iter=1000)
lr_best.fit(X_train, y_train)

svm_best = LinearSVC(C=1.0, loss='squared_hinge')
svm_best.fit(X_train, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train_tfidf)

# ================================================
# PART 7: Evaluation & Confusion Matrix
# ================================================

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 (macro):", f1_score(y_true, y_pred, average='macro'))
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, f"Confusion Matrix - {name}")

# Predictions
lr_preds = lr_best.predict(X_test)
svm_preds = svm_best.predict(X_test)
nb_preds = nb_model.predict(X_test_tfidf)

evaluate_model("Logistic Regression", y_test, lr_preds)
evaluate_model("SVM", y_test, svm_preds)
evaluate_model("Naive Bayes", y_test_tfidf, nb_preds)

# ================================================
# PART 8: ROC & PR Curves
# ================================================

from sklearn.metrics import roc_curve, auc, precision_recall_curve

lr_probs = lr_best.predict_proba(X_test)[:, 1]
svm_scores = svm_best.decision_function(X_test)
nb_probs = nb_model.predict_proba(X_test_tfidf)[:, 1]

# ROC curves
plt.figure(figsize=(10, 6))
for fpr, tpr, name, score in [
    *[(roc_curve(y_test, lr_probs)[0], roc_curve(y_test, lr_probs)[1], 'Logistic Regression', auc(*roc_curve(y_test, lr_probs)[:2]))],
    *[(roc_curve(y_test, svm_scores)[0], roc_curve(y_test, svm_scores)[1], 'SVM', auc(*roc_curve(y_test, svm_scores)[:2]))],
    *[(roc_curve(y_test_tfidf, nb_probs)[0], roc_curve(y_test_tfidf, nb_probs)[1], 'Naive Bayes', auc(*roc_curve(y_test_tfidf, nb_probs)[:2]))]
]:
    plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.grid(True)
plt.show()

# ================================================
# PART 9: Final Submission Prep
# ================================================

test_df = pd.read_csv("/content/drive/MyDrive/fake_news_detection/test (1).csv")
sample_sub = pd.read_csv("/content/drive/MyDrive/fake_news_detection/sample_submission.csv")

# Ensure order match
test_df = test_df[test_df['id'].isin(sample_sub['id'])]

# Clean and normalize
test_df['clean_text'] = test_df['text'].fillna('').apply(preprocess_text)
test_df['normalized_text'] = test_df['clean_text'].apply(lemmatize_text)

# Load saved vectorizer
from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

X_test_final = bert_model.encode(test_df['normalized_text'].tolist(), batch_size=32, show_progress_bar=True)
test_df['target'] = lr_best.predict(X_test_final)

# Submission
submission = sample_sub[['id']].merge(test_df[['id', 'target']], on='id')
submission.to_csv("submission_safe.csv", index=False)
print("Submission file saved as submission_safe.csv")

# Replace with your actual token

app_code = '''
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
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-z\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
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
'''
with open("app.py", "w") as f:
    f.write(app_code)

import pickle
with open("model_lr.pkl", "wb") as f:
    pickle.dump(lr_best, f)

from pyngrok import ngrok
import time

# Kill any existing tunnels or Streamlit apps

# Connect tunnel properly (note: addr=8501, proto='http')
public_url = ngrok.connect(addr=8501, proto="http")
print("Your Streamlit app is live at:", public_url)

# Launch Streamlit

# Let Streamlit start up
time.sleep(3)

