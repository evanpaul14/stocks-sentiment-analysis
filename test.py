import requests
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline
HEADERS = {'User-Agent': 'Evan Paul evanbobanpaul@gmail.com'}
url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
resp = requests.get(url, headers=HEADERS)
html = resp.text

# 2. clean text
soup = BeautifulSoup(html, "html.parser")
for tag in soup(["script","style"]): tag.decompose()
text = soup.get_text(" ")
text = re.sub(r"\s+", " ", text)

# 3. split by items
def extract_item(item):
    pattern = rf"Item\s+{item}\.?[\s:–-]+(.*?)(?=Item\s+\d+[A]?\b|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""

risk_text = extract_item("1A")  # Risk Factors
mda_text = extract_item("7")    # MD&A

# 4. sentiment / key sentence scoring
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipeline("sentiment-analysis")

def analyze_section(name, sec_text):
    if not sec_text:
        print(f"No section {name} found")
        return
    sentences = sent_tokenize(sec_text)
    summary = summarizer(" ".join(sentences[:40]), max_length=250, min_length=80, do_sample=False)[0]['summary_text']
    senti = sentiment(summary)[0]
    print(f"\n📘 {name} summary:")
    print(summary)
    print(f"→ Sentiment: {senti['label']} ({senti['score']:.2f})")

analyze_section("Item 1A – Risk Factors", risk_text)
analyze_section("Item 7 – MD&A", mda_text)