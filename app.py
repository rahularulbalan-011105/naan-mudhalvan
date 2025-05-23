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

# Optional: Display Evaluation Metrics from Test Set
if st.checkbox("Show model evaluation on test set"):
    # Load test data
    df = pd.read_csv("your_dataset.csv")
    df['label'] = df['label'].astype(int)
    X_test_vec = vectorizer.transform(df['text'])
    y_true = df['label']
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)

    st.markdown("### 📊 Model Metrics")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**ROC AUC:** {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

