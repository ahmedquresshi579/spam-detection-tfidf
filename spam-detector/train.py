"""
Spam Email/SMS Detection — Training Pipeline
=============================================
Pipeline: raw text -> TF-IDF vectorization -> classifier (Decision Tree or Neural Net)

Run:
    python train.py

This trains BOTH a Decision Tree and a Neural Network (MLPClassifier), prints
a comparison, and saves the BETTER one (+ the TF-IDF vectorizer) to disk so
the Streamlit app can load them.
"""

import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

DATA_PATH = "sms.tsv"          # label \t message
MODEL_PATH = "spam_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"


def clean_text(text: str) -> str:
    """Basic text cleaning before TF-IDF does its thing."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # strip urls
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
    return text


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].apply(clean_text)
    return df


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]   # prob of "spam" class, needed for ROC-AUC

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(cm)
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    return {"name": name, "model": model, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} messages | spam={df['label'].sum()} ham={(df['label']==0).sum()}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # --- TF-IDF vectorization (stop_words='english' handles stopword removal for us) ---
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # --- Model 1: Decision Tree ---
    dt = DecisionTreeClassifier(max_depth=25, min_samples_leaf=2, random_state=42)
    dt.fit(X_train, y_train)
    dt_result = evaluate("Decision Tree", dt, X_test, y_test)

    # --- Model 2: Neural Network (MLP) ---
    nn = MLPClassifier(hidden_layer_sizes=(64,), activation="relu",
                        max_iter=300, random_state=42, early_stopping=True)
    nn.fit(X_train, y_train)
    nn_result = evaluate("Neural Network (MLP)", nn, X_test, y_test)

    # --- Pick the better model by F1 (matters more than accuracy on imbalanced spam data) ---
    best = max([dt_result, nn_result], key=lambda r: r["f1"])
    print(f"\n>>> Best model: {best['name']} (F1={best['f1']:.4f}) — saving this one.")

    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open("best_model_name.txt", "w") as f:
        f.write(best["name"])

    print(f"Saved model to {MODEL_PATH}, vectorizer to {VECTORIZER_PATH}")

    # --- quick demo: predict on a few custom messages ---
    demo_messages = [
        "Congratulations! You've WON a free iPhone, click here to claim now!!!",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "URGENT: your account will be suspended. verify your password immediately",
    ]
    predict_custom(demo_messages, best["model"], vectorizer)


def predict_custom(messages, model, vectorizer):
    """Predict spam/ham for a list of raw text messages."""
    cleaned = [clean_text(m) for m in messages]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    print("\n--- Custom predictions ---")
    for msg, pred, prob in zip(messages, preds, probs):
        label = "SPAM" if pred == 1 else "HAM"
        print(f"[{label} | spam_prob={prob:.2f}] {msg}")


if __name__ == "__main__":
    main()