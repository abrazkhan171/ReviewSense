from pathlib import Path
import re
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.joblib"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Downloading/loading SST-5 dataset...")
dataset = load_dataset("SetFit/sst5")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# SST-5 labels:
# 0 = very negative, 1 = negative, 2 = neutral,
# 3 = positive, 4 = very positive.
label_map = {
    0: "Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Positive",
}

train_df["sentiment"] = train_df["label"].map(label_map)
test_df["sentiment"] = test_df["label"].map(label_map)

# Combine the two negative/positive subclasses into 3 classes.
X_train = train_df["text"].map(clean_text)
y_train = train_df["sentiment"]

X_test = test_df["text"].map(clean_text)
y_test = test_df["sentiment"]

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows:  {len(X_test):,}")

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=60000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    ),
])

print("Training Logistic Regression...")
pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, pred, average="weighted", zero_division=0
)

report = classification_report(
    y_test, pred, labels=["Positive", "Neutral", "Negative"],
    output_dict=True, zero_division=0
)

cm = confusion_matrix(
    y_test, pred, labels=["Positive", "Neutral", "Negative"]
)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "report": report,
    "confusion_matrix": cm,
    "labels": ["Positive", "Neutral", "Negative"],
    "train_size": len(X_train),
    "test_size": len(X_test),
}

joblib.dump(pipeline, MODEL_PATH)
joblib.dump(metrics, METRICS_PATH)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, pred, labels=["Positive", "Neutral", "Negative"],
    zero_division=0
))
print(f"Saved model to: {MODEL_PATH}")
print(f"Saved metrics to: {METRICS_PATH}")
