import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_reviews(url, pages=2):
    all_reviews = []
    
    for page in range(1, pages + 1):
        response = requests.get(
            f"{url}?page={page}",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(response.text, "html.parser")
        reviews = soup.find_all("p")
        
        for r in reviews:
            text = r.get_text(strip=True)
            if len(text) > 20:
                all_reviews.append({"text": text})
    
    df = pd.DataFrame(all_reviews)
    df.to_csv("raw_data.csv", index=False)
    print(f"Done! Total rows: {len(df)}")
    return df

df = scrape_reviews("https://quotes.toscrape.com")
print(df.head())