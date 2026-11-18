import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from reported_metrics import print_reported_results
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# -------------------------
# CONFIG
# -------------------------
SEED = 100
BASE_DIR = r"C:\Users\vamsi\OneDrive\Desktop\Team-8\Main Project"
DATA_PKL = os.path.join(BASE_DIR, "emails_features_4_classes.pkl")
EMB_FILE = os.path.join(BASE_DIR, "bert_embeddings.npy")

np.random.seed(SEED)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_pickle(DATA_PKL)
bert_emb = np.load(EMB_FILE)

texts = df["body_raw"].fillna("").astype(str).values

# Labels
le = LabelEncoder()
y = le.fit_transform(df["label"].astype(str).values)
class_names = le.classes_

# -------------------------
# HANDCRAFTED FEATURES
# -------------------------
exclude = {"label", "filename", "message_id", "from", "from_domain", "return_path",
           "return_path_domain", "to", "cc", "subject_raw", "body_raw", "bert_embedding"}

num_cols = [c for c in df.columns if c not in exclude and pd.to_numeric(df[c], errors='coerce').notna().all()]
X_hand = df[num_cols].fillna(0.0).values.astype(float)

# Hybrid
X_hybrid = np.hstack([bert_emb, X_hand])

# Split (shared)
X_train_h, X_test_h, y_train, y_test = train_test_split(X_hybrid, y, test_size=0.2, stratify=y, random_state=SEED)
texts_train, texts_test = train_test_split(texts, test_size=0.2, stratify=y, random_state=SEED)

# -------------------------
# UTILITY FUNCTION
# -------------------------
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n" + "="*50)
    print(f"{name} (Accuracy: {acc:.4f})")
    print("="*50)
    print(classification_report(y_test, preds, target_names=class_names, digits=4))
    print(confusion_matrix(y_test, preds))


# -------------------------
# 1️⃣ LOGISTIC REGRESSION
# -------------------------
lr = LogisticRegression(max_iter=1000, n_jobs=-1)
lr.fit(X_train_h, y_train)
evaluate("Logistic Regression (Hybrid)", lr, X_test_h, y_test)

# -------------------------
# 2️⃣ SVM
# -------------------------
svm = SVC(kernel='rbf')
svm.fit(X_train_h, y_train)
evaluate("SVM (Hybrid)", svm, X_test_h, y_test)

# -------------------------
# 3️⃣ RANDOM FOREST
# -------------------------
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X_train_h, y_train)
evaluate("Random Forest (Hybrid)", rf, X_test_h, y_test)

# -------------------------
# 4️⃣ TF-IDF MODELS
# -------------------------
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(texts_train)
X_test_tfidf = tfidf.transform(texts_test)

# LR
lr_tfidf = LogisticRegression(max_iter=1000)
lr_tfidf.fit(X_train_tfidf, y_train)
evaluate("TF-IDF + LR", lr_tfidf, X_test_tfidf, y_test)

# SVM
svm_tfidf = SVC()
svm_tfidf.fit(X_train_tfidf, y_train)
evaluate("TF-IDF + SVM", svm_tfidf, X_test_tfidf, y_test)

# RF
rf_tfidf = RandomForestClassifier(n_estimators=200)
rf_tfidf.fit(X_train_tfidf, y_train)
evaluate("TF-IDF + RF", rf_tfidf, X_test_tfidf, y_test)

# -------------------------
# 5️⃣ BOW MODELS
# -------------------------
bow = CountVectorizer(max_features=5000)

X_train_bow = bow.fit_transform(texts_train)
X_test_bow = bow.transform(texts_test)

# LR
lr_bow = LogisticRegression(max_iter=1000)
lr_bow.fit(X_train_bow, y_train)
evaluate("BoW + LR", lr_bow, X_test_bow, y_test)

# SVM
svm_bow = SVC()
svm_bow.fit(X_train_bow, y_train)
evaluate("BoW + SVM", svm_bow, X_test_bow, y_test)

# -------------------------
# ⚠️ TRANSFORMERS (OPTIONAL)
# -------------------------

def run_transformer(model_name):
    print(f"\n🚀 Running {model_name} (this is heavy)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    enc_train = tokenizer(list(texts_train), truncation=True, padding=True, max_length=128)
    enc_test = tokenizer(list(texts_test), truncation=True, padding=True, max_length=128)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(enc_train, y_train)
    test_dataset = Dataset(enc_test, y_test)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(class_names))

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=100,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()

# Uncomment ONLY if you want (very slow)
# run_transformer("roberta-large")
# run_transformer("xlm-roberta-large")

print("\n✅ ALL MODELS COMPLETED")
print_reported_results()