import hashlib
import pandas as pd

def compute_review_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def add_review_hash_column(df, review_col="review_text"):
    df["review_hash"] = df[review_col].apply(compute_review_hash)
    return df

def filter_new_reviews(df, seen_hashes):
    return df[~df["review_hash"].isin(seen_hashes)]

def load_seen_hashes(filepath="seen_hashes.csv"):
    try:
        return pd.read_csv(filepath)["review_hash"].tolist()
    except FileNotFoundError:
        return []

def update_seen_hashes(new_hashes, filepath="seen_hashes.csv"):
    pd.DataFrame({"review_hash": new_hashes}).to_csv(filepath, index=False)
