import requests
import html

API_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks"

def fetch_top_stocks():
    resp = requests.get(API_URL)
    resp.raise_for_status()
    data = resp.json()
    stocks_list = data.get("results", [])
    top10 = sorted(stocks_list, key=lambda x: x.get("rank", 999))[:10]
    return top10

def analyze_trending(top10):
    results = []
    for rec in top10:
        ticker = rec.get("ticker")
        name   = html.unescape(rec.get("name", ""))
        mentions_now = rec.get("mentions", 0)
        mentions_24h = rec.get("mentions_24h_ago", 0)
        rank_24h     = rec.get("rank_24h_ago", None)
        rank_now     = rec.get("rank", None)

        # percent increase
        if mentions_24h > 0:
            pct_increase = ((mentions_now - mentions_24h) / mentions_24h) * 100
        else:
            pct_increase = 0

        tag = ""
        if mentions_24h > 0 and pct_increase > 50:
            tag = "FAST-RISING"
        elif rank_24h is not None and rank_now is not None and rank_now < rank_24h - 5:
            tag = "FAST-RISING"

        results.append({
            "ticker": ticker,
            "name": name,
            "pct_increase": pct_increase,
            "tag": tag
        })
    return results

def main():
    top10 = fetch_top_stocks()
    analyzed = analyze_trending(top10)
    print("Top 10 Trending Stocks:")
    for i, rec in enumerate(analyzed, 1):
        ticker = rec["ticker"]
        name   = rec["name"]
        pct    = rec["pct_increase"]
        tag    = rec["tag"]
        line = f"{i}. {ticker} — {name}: +{pct:.0f}% mentions"
        if tag:
            line += f" [{tag}]"
        print(line)

if __name__ == "__main__":
    main()
