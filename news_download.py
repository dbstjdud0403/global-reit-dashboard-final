import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# Paths and settings
# --------------------------------------------------
os.makedirs("data", exist_ok=True)
DATA_DIR = Path("data")
TICKER_UNIVERSE_FILE = DATA_DIR / "sector_ticker_universe.csv"
OUTPUT_FILE = DATA_DIR / "sector_news.csv"

LOOKBACK_DAYS = 30
MAX_ARTICLES_PER_QUERY = 3
MAX_TICKER_QUERIES_PER_SECTOR = 5

# --------------------------------------------------
# Sector keyword queries
# --------------------------------------------------
SECTOR_KEYWORDS = {
    "Data Center": ["data center REIT", "AI data center demand", "digital infrastructure REIT"],
    "Healthcare": ["healthcare REIT", "senior housing REIT", "medical office REIT"],
    "Industrial": ["industrial REIT", "logistics REIT", "warehouse real estate"],
    "Industrial / Office": ["industrial office REIT", "business park REIT"],
    "Lodging / Resorts": ["hotel REIT", "lodging REIT", "travel demand REIT"],
    "Office": ["office REIT", "office vacancy", "office leasing"],
    "Residential": ["residential REIT", "apartment REIT", "multifamily REIT"],
    "Retail": ["retail REIT", "shopping center REIT", "mall REIT"],
    "Self Storage": ["self storage REIT"],
    "Specialty": ["specialty REIT", "gaming REIT", "net lease REIT"],
    "Diversified": ["diversified REIT"],
}

# --------------------------------------------------
# Issue tagging rules
# --------------------------------------------------
ISSUE_KEYWORDS = {
    "AI / Data Center Demand": ["ai", "artificial intelligence", "data center", "cloud", "hyperscale", "power demand", "digital infrastructure"],
    "Rates / Treasury Yield": ["rate", "rates", "treasury", "yield", "fed", "interest rate", "bond yield", "monetary policy"],
    "Earnings / Guidance": ["earnings", "guidance", "results", "revenue", "profit", "ffo", "affo", "same-store", "outlook"],
    "M&A / Transaction": ["acquisition", "merger", "deal", "transaction", "ipo", "sale", "joint venture", "buyout"],
    "Office Vacancy Risk": ["office vacancy", "vacancy", "remote work", "hybrid work", "leasing demand", "office demand"],
    "Housing / Rent Growth": ["rent growth", "apartment", "multifamily", "housing", "rental demand", "single-family rental"],
    "Healthcare / Senior Housing": ["senior housing", "healthcare", "medical office", "assisted living", "aging population"],
    "Industrial / Logistics Demand": ["industrial", "logistics", "warehouse", "supply chain", "e-commerce", "distribution center"],
    "Retail / Consumer Demand": ["retail sales", "consumer spending", "mall", "shopping center", "tenant sales", "foot traffic"],
    "Hotels / Travel Demand": ["hotel", "lodging", "travel demand", "occupancy", "revpar", "resort"],
    "Self Storage Demand": ["self storage", "storage", "move-in", "move out"],
    "Net Lease / Tenant Credit": ["net lease", "triple net", "tenant credit", "realty income", "lease term"],
}

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def clean_text(text):
    if text is None:
        return ""
    return " ".join(str(text).replace("\n", " ").replace("\r", " ").split())


def classify_issue(title, summary):
    text = f"{title} {summary}".lower()
    matched = []
    for issue_tag, keywords in ISSUE_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            matched.append(issue_tag)
    if not matched:
        return "General"
    return ", ".join(matched[:2])


def make_google_news_url(query):
    query_with_period = f"{query} when:{LOOKBACK_DAYS}d"
    encoded_query = urllib.parse.quote(query_with_period)
    return (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}"
        "&hl=en-US&gl=US&ceid=US:en"
    )


def fetch_google_news_rss(query, max_articles=3):
    url = make_google_news_url(query)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        xml_data = response.read()
    root = ET.fromstring(xml_data)
    channel = root.find("channel")
    if channel is None:
        return []
    items = channel.findall("item")
    articles = []
    for item in items[:max_articles]:
        title = clean_text(item.findtext("title"))
        link = clean_text(item.findtext("link"))
        published = clean_text(item.findtext("pubDate"))
        summary = clean_text(item.findtext("description"))
        source_node = item.find("source")
        source = clean_text(source_node.text if source_node is not None else "")
        articles.append({
            "title": title,
            "source": source,
            "published": published,
            "summary": summary,
            "link": link,
            "issue_tag": classify_issue(title, summary),
        })
    return articles

# --------------------------------------------------
# Build query universe
# --------------------------------------------------
def build_news_queries():
    rows = []

    # 1) Sector keyword queries
    for sector, keywords in SECTOR_KEYWORDS.items():
        for keyword in keywords:
            rows.append({
                "sector": sector,
                "query_type": "Sector Keyword",
                "query": keyword,
                "source_ticker": "",
                "source_index": "",
                "rank": "",
                "index_weight": "",
            })

    # 2) Index member ticker queries. Use top constituents by index weight/rank to avoid noisy duplicate results.
    if TICKER_UNIVERSE_FILE.exists():
        ticker_df = pd.read_csv(TICKER_UNIVERSE_FILE)
        ticker_df.columns = [str(c).strip().lower() for c in ticker_df.columns]
        required_cols = ["sector", "sector_index", "rank", "bloomberg_ticker", "search_ticker"]
        missing_cols = [c for c in required_cols if c not in ticker_df.columns]
        if missing_cols:
            raise ValueError(f"sector_ticker_universe.csv에 필요한 컬럼이 없습니다: {missing_cols}")

        ticker_df["rank"] = pd.to_numeric(ticker_df["rank"], errors="coerce")
        if "index_weight" in ticker_df.columns:
            ticker_df["index_weight"] = pd.to_numeric(ticker_df["index_weight"], errors="coerce")
        else:
            ticker_df["index_weight"] = pd.NA

        ticker_df = ticker_df[ticker_df["rank"] <= MAX_TICKER_QUERIES_PER_SECTOR].copy()
        ticker_df = ticker_df.sort_values(["sector", "rank"])

        for _, row in ticker_df.iterrows():
            search_ticker = str(row.get("search_ticker", "")).strip()
            if search_ticker == "" or search_ticker.lower() == "nan":
                continue
            rows.append({
                "sector": row.get("sector", ""),
                "query_type": "Index Member Ticker",
                "query": f"{search_ticker} REIT",
                "source_ticker": row.get("bloomberg_ticker", ""),
                "source_index": row.get("sector_index", ""),
                "rank": row.get("rank", ""),
                "index_weight": row.get("index_weight", ""),
            })
    else:
        print(f"Warning: {TICKER_UNIVERSE_FILE} 파일이 없습니다. 섹터 키워드만 사용합니다.")

    query_df = pd.DataFrame(rows)
    query_df = query_df.drop_duplicates(subset=["sector", "query_type", "query", "source_ticker"])
    return query_df

# --------------------------------------------------
# Main collection
# --------------------------------------------------
def build_sector_news():
    query_df = build_news_queries()
    print("News query universe:")
    print(query_df.head(20))
    print(f"Total queries: {len(query_df)}")

    rows = []
    for _, qrow in query_df.iterrows():
        sector = qrow["sector"]
        query_type = qrow["query_type"]
        query = qrow["query"]
        print(f"Fetching news for {sector} | {query_type} | {query}")
        try:
            articles = fetch_google_news_rss(query=query, max_articles=MAX_ARTICLES_PER_QUERY)
            for article in articles:
                rows.append({
                    "sector": sector,
                    "query_type": query_type,
                    "query": query,
                    "source_ticker": qrow.get("source_ticker", ""),
                    "source_index": qrow.get("source_index", ""),
                    "rank": qrow.get("rank", ""),
                    "index_weight": qrow.get("index_weight", ""),
                    "title": article["title"],
                    "source": article["source"],
                    "published": article["published"],
                    "summary": article["summary"],
                    "link": article["link"],
                    "issue_tag": article["issue_tag"],
                })
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching {sector} / {query}: {e}")

    news_df = pd.DataFrame(rows)
    if news_df.empty:
        print("No news collected.")
        news_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        return news_df

    news_df = news_df.drop_duplicates(subset=["title", "link"])
    news_df["published_datetime"] = pd.to_datetime(news_df["published"], errors="coerce", utc=True)
    news_df = news_df.sort_values("published_datetime", ascending=False)
    ordered_cols = [
        "sector", "query_type", "query", "source_ticker", "source_index", "rank", "index_weight",
        "title", "source", "published", "published_datetime", "summary", "link", "issue_tag",
    ]
    news_df = news_df[[c for c in ordered_cols if c in news_df.columns]]
    news_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_FILE}")
    print(news_df.head(20))
    return news_df

if __name__ == "__main__":
    build_sector_news()
