import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

print("=" * 50)
print("  STEP 2: DATA CLEANER")
print("=" * 50)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'â€œ|â€|[â€™â€˜]', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

df = pd.read_csv("data/raw_data.csv")
print(f"Raw data loaded: {len(df)} rows")

df['cleaned_text'] = df['text'].apply(clean_text)
df = df[df['cleaned_text'].str.len() > 10]
df = df.drop_duplicates(subset=['cleaned_text'])
df = df.dropna(subset=['cleaned_text'])

print(f"After cleaning: {len(df)} rows")

df['text_length'] = df['cleaned_text'].apply(len)
df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))

df.to_csv("data/cleaned_data.csv", index=False)
print(f"File saved: data/cleaned_data.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['word_count'], bins=20, color='steelblue', edgecolor='black')
axes[0].set_title('Word Count Distribution')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')

axes[1].hist(df['text_length'], bins=20, color='orange', edgecolor='black')
axes[1].set_title('Text Length Distribution')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("outputs/cleaning_stats.png", dpi=150)
plt.show()
print("Plot saved: outputs/cleaning_stats.png")

print("\nSample cleaned data:")
print(df[['text', 'cleaned_text']].head(3))