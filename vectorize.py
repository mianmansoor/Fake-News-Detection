# vectorize.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# === Load cleaned dataset ===
try:
    df = pd.read_csv("data/cleaned_preprocessed_news.csv")
    print("✅ Loaded 'cleaned_preprocessed_news.csv'")
except FileNotFoundError:
    raise FileNotFoundError("❌ File 'cleaned_preprocessed_news.csv' not found.")

# === Fill NaNs in text ===
texts = df["text"].fillna("")

# === Create and fit Word-level TF-IDF Vectorizer ===
print("🔤 Fitting word-level TF-IDF...")
word_vectorizer = TfidfVectorizer(
    analyzer="word", max_features=10000, stop_words="english"
)
word_vectorizer.fit(texts)
joblib.dump(word_vectorizer, "model/tfidf_words.pkl")
print("✅ Saved: tfidf_words.pkl")

# === Create and fit Char-level TF-IDF Vectorizer ===
print("🔡 Fitting char-level TF-IDF...")
char_vectorizer = TfidfVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=5000
)
char_vectorizer.fit(texts)
joblib.dump(char_vectorizer, "model/tfidf_chars.pkl")
print("✅ Saved: tfidf_chars.pkl")

# === Create and fit CountVectorizer ===
print("🔢 Fitting CountVectorizer...")
count_vectorizer = CountVectorizer(max_features=5000, stop_words="english")
count_vectorizer.fit(texts)
joblib.dump(count_vectorizer, "model/count_vectorizer.pkl")
print("✅ Saved: count_vectorizer.pkl")

print("\n🎉 All vectorizers generated and saved successfully!")
