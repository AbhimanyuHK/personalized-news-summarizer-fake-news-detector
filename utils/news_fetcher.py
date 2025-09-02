# utils/news_fetcher.py
import requests
import os


class NewsFetcher:
    def __init__(self, api_key=None):
        """
        Initialize NewsFetcher with NewsAPI key.
        Args:
            api_key (str): NewsAPI key, can also be set via NEWSAPI_KEY env variable
        """
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError("⚠️ Missing NewsAPI key. Set NEWSAPI_KEY env variable or pass api_key.")

        self.base_url = "https://newsapi.org/v2/everything"

    def fetch(self, query: str, language: str = "en", page_size: int = 5):
        """
        Fetch news articles from NewsAPI.

        Args:
            query (str): Keywords or topics (e.g., "AI", "economy")
            language (str): Language of news (default English)
            page_size (int): Number of articles to fetch (max 100)

        Returns:
            list[dict]: List of articles with title, description, content, url
        """
        params = {
            "q": query,
            "language": language,
            "sortBy": "relevancy",
            "pageSize": page_size,
            "apiKey": self.api_key
        }

        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"❌ Failed to fetch news: {response.status_code}, {response.text}")

        articles = response.json().get("articles", [])
        results = []
        for a in articles:
            results.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "content": a.get("content"),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name")
            })

        return results


# Example usage
if __name__ == "__main__":
    # Make sure to set NEWSAPI_KEY in your environment
    fetcher = NewsFetcher()
    articles = fetcher.fetch("Artificial Intelligence", page_size=3)
    for art in articles:
        print(f"📰 {art['title']} ({art['source']})\n{art['url']}\n")
