# 📰 Fake News Detection using NLP and Machine Learning

This project is a machine learning pipeline for detecting fake news articles using classical ML algorithms and natural language processing techniques.

---

## 🔍 Features

- Advanced text preprocessing using spaCy (NER, POS filtering)
- Vectorization using:
  - Word-level TF-IDF
  - Character-level TF-IDF
  - Count Vectorizer
- Model comparison using:
  - Logistic Regression
  - Naive Bayes
- Evaluation with F1 Score, Accuracy, Precision, and Recall
- Interactive testing with command-line interface

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
````

### 2. Create a virtual environment

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 🧪 Dataset

📄 Dataset
we used the WELFake_Dataset.csv availaible online [click here to get the Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

Place your dataset CSV file (with at least `text` and `label` columns) at:

```
data/WELFake_Dataset.csv
```

---

## 🚀 Running the Project

### Step 1: Preprocess the text

```bash
python preprocessing.py
```

### Step 2: Generate vectorizers

```bash
python vectorize.py
```

### Step 3: Train and evaluate models

```bash
python model_train.py
```

### Step 4: Test random samples and Predict manually entered news articles

```bash
python test_model.py
```

---

## 📊 Sample Results

```
Accuracy : 0.93
Precision: 0.92
Recall   : 0.94
F1 Score : 0.93
```

---

## 📁 Project Structure

```
├── data/
│   ├── WELFake_Dataset.csv
│   └── cleaned_preprocessed_news.csv
├── model/
│   ├── best_model.pkl
│   ├── count_vectorizer.pkl
│   ├── tfidf_chars.pkl
│   └── tfidf_words.pkl
├── vectorize.py
├── model_train.py
├── test_model.py
├── preprocessing.py
├── requirements.txt
└── README.md
```

---

## 🧠 Skills Demonstrated

* Natural Language Processing (NLP)
* Feature Engineering
* Machine Learning Model Evaluation
* Pipeline Structuring
* Model Persistence

---

## 📬 Contact

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/mansoorfareed) or drop suggestions via issues or pull requests!

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

```

---

Let me know if you want me to:
- Suggest a GitHub repository name
- Write a LinkedIn post caption
- Design a project thumbnail  
I'm happy to help!
```
