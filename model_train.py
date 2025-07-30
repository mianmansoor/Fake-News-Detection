# main.py
import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm, trange
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Load cleaned dataset ===
try:
    df = pd.read_csv("cleaned_preprocessed_news.csv")
    print("✅ Loaded 'cleaned_preprocessed_news.csv'")
except FileNotFoundError:
    raise FileNotFoundError("❌ File 'cleaned_preprocessed_news.csv' not found.")

# Check required columns
if not {"text", "label"}.issubset(df.columns):
    raise ValueError("❌ Required columns 'text' and 'label' not found!")

X_text = df["text"].fillna("")
y = df["label"].values

# === Load vectorizers ===
def load_vectorizer(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Vectorizer not found: {filename}")
    return joblib.load(filename)

print("📦 Loading vectorizers...")
word_vectorizer = load_vectorizer("tfidf_words.pkl")
char_vectorizer = load_vectorizer("tfidf_chars.pkl")
count_vectorizer = load_vectorizer("count_vectorizer.pkl")

# === Transform text ===
print("⚙️ Vectorizing text...")
word_features = word_vectorizer.transform(X_text)
char_features = char_vectorizer.transform(X_text)
count_features = count_vectorizer.transform(X_text)
X_combined = hstack([word_features, char_features, count_features])
print(f"✅ Vectorization complete → Shape: {X_combined.shape}")

# === Model training ===
def train_and_evaluate_models(X, y, save_model_path="best_model.pkl"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB()
    }

    best_model = None
    best_f1 = 0
    best_model_name = ""

    for name, model in models.items():
        print(f"\n🚀 Training: {name}")
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"   Fold {fold}/5", end=" ")
            for i in trange(1, 101, desc=f"      ▶️ Progress (Fold {fold})", leave=False):
                pass  # fake progress (you can remove this if it's too much)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
            print(f"→ F1: {f1:.4f}")

        avg_f1 = np.mean(f1_scores)
        print(f"✅ {name} | Avg F1 Score: {avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model = model
            best_model_name = name

    print(f"\n🏆 Best Model: {best_model_name} | Avg F1: {best_f1:.4f}")

    # Final train-test split
    print("\n📦 Training best model on full training set...")
    X_train_final, X_test, y_train_final, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    best_model.fit(X_train_final, y_train_final)
    joblib.dump(best_model, save_model_path)
    print(f"💾 Best model saved as '{save_model_path}'")

    print("\n📊 Final Evaluation on Hold-out Test Set:")
    y_pred = best_model.predict(X_test)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

# === Run it ===
if __name__ == "__main__":
    train_and_evaluate_models(X_combined, y)
