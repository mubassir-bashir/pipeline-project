# 📰 Sentiment Analysis Pipeline Project

A beginner-friendly NLP pipeline that scrapes BBC News articles, cleans the data, runs sentiment analysis, and generates an evaluation report.

---

## 🔷 Pipeline Overview

```
BBC News Website
      ↓
scraper.py → raw_data.csv
      ↓
cleaner.py → cleaned_data.csv
      ↓
sentiment_model.py → output_data.csv
      ↓
report.py → report_chart.png
```

---

## 📁 Project Structure

```
pipeline-project/
├── scraper.py            # Scrapes BBC News articles
├── cleaner.py            # Cleans raw text data
├── sentiment_model.py    # Trains & tests sentiment model
├── report.py             # Generates evaluation report & chart
├── raw_data.csv          # Raw scraped data
├── cleaned_data.csv      # Cleaned dataset
├── output_data.csv       # Final output with sentiments
├── report_chart.png      # Sentiment bar chart
└── README.md             # Project documentation
```

---

## ⚙️ Installation

Make sure Python is installed, then run:

```bash
pip install requests beautifulsoup4 pandas scikit-learn textblob nltk matplotlib lxml
```

---

## 🚀 How to Run

Run each file one by one in order:

**Step 1 — Scrape Data:**
```bash
python scraper.py
```
Scrapes BBC News and saves articles to `raw_data.csv`

**Step 2 — Clean Data:**
```bash
python cleaner.py
```
Cleans text and saves to `cleaned_data.csv`

**Step 3 — Train & Test Model:**
```bash
python sentiment_model.py
```
Labels sentiments, trains model, and prints accuracy report

**Step 4 — Generate Report:**
```bash
python report.py
```
Saves bar chart to `report_chart.png`

---

## 📊 Results

| Sentiment | Count |
|-----------|-------|
| Neutral   | 22    |
| Positive  | 12    |
| Negative  | 8     |

![Sentiment Chart](report_chart.png)

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `requests` | Fetch website content |
| `BeautifulSoup` | Parse HTML |
| `pandas` | Data handling |
| `TextBlob` | Sentiment labeling |
| `scikit-learn` | Train/Test model |
| `matplotlib` | Chart generation |

---

## 👨‍💻 Author

**Mubussir Bahir**
Internship Project — Glaxit Software House
