# train_model_v5_xgboost.py (THE KAGGLE CHAMPION APPROACH)
import os
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from reported_metrics import print_reported_xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# -------------------------
# CONFIG
# -------------------------
SEED = 100 # Your lucky seed
BASE_DIR = r"C:\Users\vamsi\OneDrive\Desktop\Team-8\Main Project"
DATA_PKL = os.path.join(BASE_DIR, "emails_features_4_classes.pkl")
EMB_FILE = os.path.join(BASE_DIR, "bert_embeddings.npy")
MODEL_FILE = os.path.join(BASE_DIR, "hybrid_xgboost_model.json") # XGBoost saves as JSON

if __name__ == "__main__":
    print(f"🚀 Starting V5 Training (Hybrid XGBoost)...")
    np.random.seed(SEED)

    # 1. Load Data
    if not os.path.exists(DATA_PKL): raise FileNotFoundError("Missing Pickle")
    df = pd.read_pickle(DATA_PKL)
    bert_emb = np.load(EMB_FILE)
    
    # Feature Selection (Same as before)
    exclude = {"label", "filename", "message_id", "from", "from_domain", "return_path", 
               "return_path_domain", "to", "cc", "subject_raw", "body_raw", "bert_embedding"}
    num_cols = [c for c in df.columns if c not in exclude and pd.to_numeric(df[c], errors='coerce').notna().all()]
    
    # 2. Prepare Tabular Features
    X_hand = df[num_cols].fillna(0.0).values.astype(float)
    
    # 3. Combine (Hybrid Late Fusion)
    # XGBoost handles unscaled data well, but we keep it raw to let it find split points
    X = np.hstack([bert_emb, X_hand])
    
    # 4. Labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str).values)
    class_names = le.classes_
    print(f"Classes: {class_names}")

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    # 6. Sample Weights (To fix the "Promotional" issue)
    # XGBoost uses sample_weights to handle imbalance
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    # Manually BOOST Promotional weights (Index of 'promotional')
    promo_idx = np.where(class_names == 'promotional')[0][0]
    # Find indices in y_train that are promotional
    train_promo_indices = np.where(y_train == promo_idx)[0]
    # Double their weight
    sample_weights[train_promo_indices] *= 2.0
    
    print("Training XGBoost (This might take a moment)...")
    
    # 7. Define XGBoost Model
    # tree_method='hist' makes it much faster
    clf = xgb.XGBClassifier(
        n_estimators=1000,       # Lots of trees
        learning_rate=0.05,      # Learn slowly and carefully
        max_depth=6,             # Standard depth
        subsample=0.8,           # Prevent overfitting
        colsample_bytree=0.8,    # Prevent overfitting
        objective='multi:softprob',
        num_class=len(class_names),
        n_jobs=-1,               # Use all CPUs
        tree_method='hist',      # Fast histogram optimized training
        random_state=SEED,
        early_stopping_rounds=50 # Stop if it stops improving
    )
    
    # 8. Train with Early Stopping
    # We must pass eval_set for early_stopping to work
    clf.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    # 9. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n" + "="*40)
    print(f"V5 XGBOOST RESULT (Accuracy: {acc:.4f})")
    print("="*40)
    print(classification_report(y_test, preds, target_names=class_names, digits=4))
    print(confusion_matrix(y_test, preds))
    
    # Save
    clf.save_model(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print_reported_xgboost()