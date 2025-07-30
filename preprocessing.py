# preprocessing.py
import re
import pandas as pd
from tqdm import tqdm
import spacy
import time
import os

# Load spaCy model globally to avoid reloading
nlp = spacy.load("en_core_web_sm", disable=["parser"])

def clean_text_improved(text):
    """Improved text cleaning that preserves important context"""
    if pd.isna(text) or text == '':
        return ''

    text = str(text).lower()

    text = re.sub(r'http\S+|www\.\S+', ' url_link ', text)

    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
        "'m": " am", "let's": "let us", "that's": "that is"
    }

    for contraction, expansion in contractions.items():
        text = re.sub(contraction, expansion, text)

    text = re.sub(r'[^\w\s\.\!\?\,\-\$\%\#\@]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def preprocess_batch_improved(text_series, batch_size=100):
    """Improved preprocessing with spaCy entity and POS filtering"""
    cleaned_texts = text_series.fillna('').apply(clean_text_improved).tolist()
    preprocessed = []

    print(f"Processing {len(cleaned_texts)} texts...")
    start_time = time.time()

    for i in tqdm(range(0, len(cleaned_texts), batch_size)):
        batch = cleaned_texts[i:i+batch_size]
        for doc in nlp.pipe(batch, batch_size=min(64, len(batch))):
            tokens = []
            entities = []

            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP', 'FAC', 'EVENT']:
                    entities.append(ent.text.lower().replace(' ', '_'))

            for token in doc:
                if (not token.is_space and
                    len(token.text.strip()) > 1 and
                    not token.is_punct):

                    if (token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'NUM'] or
                        token.like_num or
                        token.ent_type_ in ['PERSON', 'ORG', 'GPE', 'NORP'] or
                        token.text.lower() in ['not', 'never', 'breaking', 'urgent', 'exclusive', 'confirmed']):

                        if token.pos_ == 'PROPN' or token.ent_type_:
                            tokens.append(token.text.lower())
                        elif token.like_num or token.pos_ == 'NUM':
                            tokens.append(token.text.lower())
                        else:
                            tokens.append(token.lemma_.lower())

            all_tokens = tokens + entities
            preprocessed.append(" ".join(all_tokens) if all_tokens else "")

    print(f"✅ Done in {round(time.time() - start_time, 2)} seconds.")
    return preprocessed

# === Run Preprocessing Locally ===

if __name__ == "__main__":
    # 👇 Change this to your CSV file path
    input_file = "data/WELFake_Dataset.csv"
    output_file = "data/cleaned_preprocessed_news.csv"
    text_column = "text"  # 👈 Change to your actual text column name (e.g., 'title', 'content')

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        exit(1)

    print(f"📂 Reading dataset from: {input_file}")
    df = pd.read_csv(input_file)

    if text_column not in df.columns:
        print(f"❌ Column '{text_column}' not found in dataset.")
        print(f"✅ Available columns: {list(df.columns)}")
        exit(1)

    df['clean_text'] = preprocess_batch_improved(df[text_column])
    df.to_csv(output_file, index=False)

    print(f"\n✅ Preprocessed data saved to: {output_file}")
