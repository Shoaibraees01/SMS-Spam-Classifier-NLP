import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# ---------------------------
# Load saved models & vectorizers
# ---------------------------
@st.cache_resource
def load_models():
    tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
    lr_model = joblib.load("artifacts/logreg_tfidf.pkl")
    lstm_model = tf.keras.models.load_model("artifacts/lstm_model.h5", compile=False)

    # ✅ Fix: load tokenizer correctly as string
    with open("artifacts/tokenizer.json", "r") as f:
        tokenizer_json = f.read()  # read as string
    tokenizer = tokenizer_from_json(tokenizer_json)

    return tfidf, lr_model, lstm_model, tokenizer


tfidf, lr_model, lstm_model, tokenizer = load_models()

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="📩 SMS Spam Classifier", page_icon="📱", layout="centered")

# Header
st.markdown(
    """
    <div style="text-align:center">
        <h1>📱 SMS Spam Classifier</h1>
        <p style="font-size:18px;">Classify SMS messages as <b>Spam</b> or <b>Ham (Not Spam)</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# User Input Section
# ---------------------------
st.write("### ✍️ Enter your SMS message:")
user_input = st.text_area("", height=120, placeholder="Type your SMS message here...")

# Model Selection
st.write("### ⚙️ Choose a Model:")
model_choice = st.radio(
    "Select the model you want to use:",
    ("Logistic Regression (TF-IDF)", "LSTM Neural Network"),
    horizontal=True
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Classify Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        if model_choice == "Logistic Regression (TF-IDF)":
            # Logistic Regression Prediction
            x_tfidf = tfidf.transform([user_input])
            prob = lr_model.predict_proba(x_tfidf)[0][1]  # probability of spam
            label = "Spam" if prob > 0.5 else "Ham"

        else:
            # LSTM Prediction
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=100)  # must match training maxlen
            prob = lstm_model.predict(padded)[0][0]
            label = "Spam" if prob > 0.5 else "Ham"

        # ---------------------------
        # Show Results
        # ---------------------------
        if label == "Spam":
            st.error(f"🚨 Prediction: **Spam**")
        else:
            st.success(f"✅ Prediction: **Ham (Not Spam)**")

        st.progress(float(prob))
        st.info(f"📊 Spam Probability: **{prob:.2f}**")

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:14px; color:gray;">
        Built with ❤️ using Streamlit | Choose between ML & DL models
    </div>
    """,
    unsafe_allow_html=True 
)
