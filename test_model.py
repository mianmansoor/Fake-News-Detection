import pandas as pd
import joblib
import random
from scipy.sparse import hstack

# === Load model and vectorizers ===
try:
    print("📦 Loading model and vectorizers...")
    model = joblib.load("model/best_model.pkl")
    tfidf_words = joblib.load("model/tfidf_words.pkl")
    tfidf_chars = joblib.load("model/tfidf_chars.pkl")
    count_vect = joblib.load("model/count_vectorizer.pkl")
    print("✅ All components loaded successfully!")
except Exception as e:
    print(f"❌ Error loading components: {e}")
    exit(1)

# === Sample Testing Function ===
def test_sample_predictions():
    """Test 10 random samples from the dataset"""
    print("\n🧪 Testing 10 Random Samples from Dataset")
    print("========================================\n")

    try:
        df = pd.read_csv("data/cleaned_preprocessed_news.csv")
        df = df.dropna(subset=["text", "label"])
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    samples = df.sample(10, random_state=42)
    correct = 0

    for idx, row in samples.iterrows():
        text = row["text"]
        true_label = row["label"]

        word_features = tfidf_words.transform([text])
        char_features = tfidf_chars.transform([text])
        count_features = count_vect.transform([text])
        features = hstack([word_features, char_features, count_features])
        prediction = model.predict(features)[0]

        if prediction == true_label:
            correct += 1

        print(f"📰 Article: {text[:80]}...")
        print(f"🔍 Prediction: {'FAKE ❌' if prediction == 1 else 'REAL ✅'}")
        print(f"✅ Actual: {'FAKE ❌' if true_label == 1 else 'REAL ✅'}")
        print("--------------------------------------------------")

    print(f"\n🎯 Accuracy on 10 samples: {correct}/10 correct ✅")

# === Interactive Prediction ===
def interactive_testing():
    """Take user input and predict whether it's fake or real"""
    print("\n🔍 Fake News Detector")
    print("====================================")
    print("📄 Enter your news article (paste text and press Enter twice):")

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user.")
            return

    article = " ".join(lines).strip()

    if not article:
        print("❌ No input provided.")
        return

    word_features = tfidf_words.transform([article])
    char_features = tfidf_chars.transform([article])
    count_features = count_vect.transform([article])
    features = hstack([word_features, char_features, count_features])
    prediction = model.predict(features)[0]

    print("\n====================================")
    print("📄 Your Input:\n", article[:500], "..." if len(article) > 500 else "")
    print("🧠 Prediction:", "FAKE ❌" if prediction == 1 else "REAL ✅")
    print("====================================\n")

# === Menu Main Function ===
def main():
    """Main function with menu options"""
    print("=" * 40)
    print("🔍 FAKE NEWS DETECTION MODEL TESTER")
    print("=" * 40)
    
    while True:
        print("\n🎯 Choose testing mode:")
        print("1. Test sample articles")
        print("2. Interactive testing")
        print("3. Exit")
        
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                test_sample_predictions()
            elif choice == '2':
                interactive_testing()
            elif choice == '3':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# === Entry Point ===
if __name__ == "__main__":
    main()
