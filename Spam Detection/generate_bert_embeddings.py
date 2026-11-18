# generate_bert_embeddings.py
import numpy as np
import pandas as pd
import torch
import os
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# CHANGED: Now points to the 4-class features file
DATA_PKL = r"D:\Main Project\Main Project\emails_features_4_classes.pkl"
OUTPUT_FILE = r"D:\Main Project\Main Project\bert_embeddings.npy"

# ---------------------------------------------------------
# GENERATION CODE
# ---------------------------------------------------------
def generate_embeddings():
    print("Loading dataset...")
    if not os.path.exists(DATA_PKL):
        raise FileNotFoundError(f"Pickle file not found: {DATA_PKL}\nDid you run feature_extractor.py?")

    df = pd.read_pickle(DATA_PKL)

    # Ensure we use the correct text column
    texts = df["body_raw"].fillna("").tolist()
    print(f"Total emails to encode: {len(texts)}")

    # Load BERT (Pre-trained)
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert.eval()

    emb_list = []

    print("\nGenerating BERT embeddings (CPU)... This will take time...\n")
    # Batch processing isn't strictly necessary for CPU, but tqdm helps track progress
    for txt in tqdm(texts, desc="BERT Encoding"):
        # Tokenize
        tokens = tokenizer(
            txt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Forward pass
        with torch.no_grad():
            out = bert(**tokens)

        # Take [CLS] token (pooler_output) which represents the whole sentence
        emb = out.pooler_output.squeeze().numpy()
        emb_list.append(emb)

    bert_emb = np.array(emb_list)

    # Save
    np.save(OUTPUT_FILE, bert_emb)
    print("\n✅ Saved embeddings to:", OUTPUT_FILE)
    print("Shape:", bert_emb.shape)

if __name__ == "__main__":
    generate_embeddings()