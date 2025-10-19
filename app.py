from google import genai
from pygooglenews import GoogleNews
from yahooquery import search
from datetime import datetime, timedelta
import json


company_name = input("Enter the company name: ")
results = search(company_name)
stock_symbol = results['quotes'][0]['symbol']
print(f"Stock symbol: {stock_symbol}")

num_articles = 10


client = genai.Client(api_key="AIzaSyA7ORNSE_3Ma1E4dygw8ZhTWo8EWoez2IQ")
gemma_model = "gemma-3-27b-it"

def get_news_articles(stock_symbol, num_articles):
    gn = GoogleNews(lang='en', country='US')
    search_result = gn.search(stock_symbol)
    entries = search_result.get('entries', [])

    articles = []
    for entry in entries[:num_articles]:
        articles.append({
            'title': entry.get('title', 'No title'),
            'description': entry.get('summary', ''),
            'link': entry.get('link', ''),
            'publishedAt': entry.get('published', '')
        })
    return articles

def analyze_sentiment_gemma(article_title, article_description, company_name):
    prompt = (
        f"Analyze the sentiment (positive, negative, or neutral) of this news article strictly in reference "
        f"to the company {company_name}.\n\n"
        f"Title: {article_title}\n"
        f"Description: {article_description}\n\n"
        f"Respond with only one word: positive, negative, or neutral."
    )

    response = client.models.generate_content(
        model=gemma_model,
        contents=prompt
    )

    return response.text.strip().lower()

articles = get_news_articles(stock_symbol, num_articles)

summary = {"positive": 0, "negative": 0, "neutral": 0}
sentiment_results = []

for i, article in enumerate(articles):
    sentiment = analyze_sentiment_gemma(article['title'], article['description'], company_name)
    sentiment_results.append({"title": article['title'], "sentiment": sentiment})
    summary[sentiment] += 1

    print(f"\nArticle {i+1}:")
    print(f"Title: {article['title']}")
    print(f"Description: {article['description']}")
    print(f"Published: {article['publishedAt']}")
    print(f"Link: {article['link']}")
    print(f"Sentiment: {sentiment.upper()}")

most_common_sentiment = max(summary, key=summary.get)
print("\nSummary:")
print(f"POSITIVE: {summary['positive']}")
print(f"NEGATIVE: {summary['negative']}")
print(f"NEUTRAL: {summary['neutral']}")
print(f"Overall Sentiment: {most_common_sentiment.upper()}")
