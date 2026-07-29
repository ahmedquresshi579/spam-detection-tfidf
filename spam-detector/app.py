"""
Spam Detector — Streamlit App
==============================
Run:
    streamlit run streamlit_app.py

Loads the model + vectorizer saved by train.py and lets you test messages live.
"""

import os
import re
import joblib
import streamlit as st

# Resolve paths relative to THIS file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.joblib")
BEST_MODEL_NAME_PATH = os.path.join(BASE_DIR, "best_model_name.txt")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    try:
        with open(BEST_MODEL_NAME_PATH) as f:
            model_name = f.read().strip()
    except FileNotFoundError:
        model_name = model.__class__.__name__
    return model, vectorizer, model_name


st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Spam Email/SMS Detector")

try:
    model, vectorizer, model_name = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train.py` first in this same folder "
        "to generate `spam_model.joblib` and `vectorizer.joblib`."
    )
    st.stop()

st.caption(f"Currently serving: **{model_name}** (TF-IDF + this model)")

message = st.text_area(
    "Paste a message to check:",
    height=150,
    placeholder="e.g. Congratulations! You've won a free prize, click here to claim...",
)

if st.button("Check", type="primary"):
    if not message.strip():
        st.warning("Type something first.")
    else:
        cleaned = clean_text(message)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        if pred == 1:
            st.error(f"🚨 SPAM  —  {prob:.1%} spam probability")
        else:
            st.success(f"✅ HAM (not spam)  —  {prob:.1%} spam probability")

        st.progress(float(prob))

with st.expander("What's happening under the hood"):
    st.markdown(
        """
        1. Your text gets cleaned (lowercase, punctuation stripped, urls removed)
        2. TF-IDF turns it into a vector of weighted word/phrase features
        3. The trained classifier scores that vector
        4. You get a spam/ham label + probability
        """
    )
