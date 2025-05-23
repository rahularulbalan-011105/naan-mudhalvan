import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("💬 Sentiment Analysis App")
st.write("Enter a sentence to predict sentiment (Happy or Sad).")

# User input
user_input = st.text_area("Enter your text here", height=300)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X_vec = vectorizer.transform([user_input])
        prediction = model.predict(X_vec)[0]
        probability = model.predict_proba(X_vec)[0][1]

        sentiment = "HAPPY 😊" if prediction == 1 else "SAD 😞"
        st.subheader(f"Prediction: {sentiment}")
