# 📧 Spam Email/SMS Detector

A binary classifier that flags messages as **spam** or **ham** (not spam), built with TF-IDF + a trained classifier, served through a Streamlit app.

## Pipeline

```
Raw text
  → Clean (lowercase, strip punctuation/urls, remove stopwords)
  → TF-IDF vectorization
  → Train/test split (80/20, stratified)
  → Train Decision Tree AND Neural Network (MLP)
  → Evaluate both (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
  → Save the better-performing model
  → Streamlit app for live predictions
```

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,574 real SMS messages labeled ham/spam. Included as `sms.tsv` (tab-separated: `label \t message`).

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.958 | 0.886 | 0.785 | 0.833 | 0.889 |
| **Neural Network (MLP)** | **0.981** | **0.978** | **0.879** | **0.926** | **0.987** |

Neural Network wins and is the one saved/served by the app.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Train the model** (already done — `spam_model.joblib` and `vectorizer.joblib` are included, so you can skip this unless you want to retrain):

```bash
python train.py
```

**2. Run the app:**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste in a message, hit Check, get a spam/ham prediction with probability.

## Files

| File | What it is |
|---|---|
| `sms.tsv` | Dataset |
| `train.py` | Full training pipeline — cleans data, vectorizes, trains both models, evaluates, saves the best one |
| `app.py` | Streamlit app — loads the saved model and serves live predictions |
| `spam_model.joblib` | Trained classifier (saved) |
| `vectorizer.joblib` | Fitted TF-IDF vectorizer (saved) |
| `requirements.txt` | Python dependencies |

## Notes

- Model is trained on SMS text, not email — structurally similar (short text, spam vs. not) but may not generalize perfectly to longer email-style text with headers/HTML/footers.
- `.joblib` files are binary — not meant to be opened in a text editor. They store the trained model/vectorizer objects so they don't need to be retrained every time the app runs.
