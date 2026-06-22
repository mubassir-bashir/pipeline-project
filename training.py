import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

print("=" * 50)
print("  STEP 4: MODEL TRAINING")
print("=" * 50)

df = pd.read_csv("data/labeled_data.csv").dropna(subset=['cleaned_text', 'sentiment'])
print(f"Data loaded: {len(df)} rows")
print(f"Sentiment counts:\n{df['sentiment'].value_counts()}")

X = df['cleaned_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    ))
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("Training complete!")

with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

test_data = pd.DataFrame({'X_test': X_test, 'y_test': y_test})
test_data.to_csv("data/test_data.csv", index=False)

print(f"\nModel saved: models/sentiment_model.pkl")
print(f"Test data saved: data/test_data.csv")

sample_texts = [
    "I love this so much it makes me happy",
    "This is terrible and awful",
    "The weather is okay today"
]
print("\nQuick test predictions:")
for text in sample_texts:
    pred = pipeline.predict([text])[0]
    print(f"  '{text[:40]}...' => {pred}")