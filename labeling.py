import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

print("=" * 50)
print("  STEP 3: SENTIMENT LABELING")
print("=" * 50)

nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = sia.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def get_score(text):
    return sia.polarity_scores(str(text))['compound']

df = pd.read_csv("data/cleaned_data.csv")
print(f"Data loaded: {len(df)} rows")

df['sentiment'] = df['cleaned_text'].apply(get_sentiment)
df['vader_score'] = df['cleaned_text'].apply(get_score)

df.to_csv("data/labeled_data.csv", index=False)
print(f"File saved: data/labeled_data.csv")

counts = df['sentiment'].value_counts()
print(f"\nSentiment Distribution:")
print(counts)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ['#2ecc71', '#e74c3c', '#95a5a6']
axes[0].bar(counts.index, counts.values, color=colors[:len(counts)])
axes[0].set_title('Sentiment Distribution (Bar)')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Count')
for i, (idx, val) in enumerate(counts.items()):
    axes[0].text(i, val + 0.5, str(val), ha='center', fontweight='bold')

axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=colors[:len(counts)], startangle=90)
axes[1].set_title('Sentiment Distribution (Pie)')

plt.tight_layout()
plt.savefig("outputs/sentiment_distribution.png", dpi=150)
plt.show()
print("Plot saved: outputs/sentiment_distribution.png")