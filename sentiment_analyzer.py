import requests
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class FinancialSentimentAnalyzer:
    """
    NLP Sentiment Analysis for FX Markets.
    Now with REAL News capability via The Guardian API.
    """
    def __init__(self, api_key=None):
        self.analyzer = SentimentIntensityAnalyzer()
        self.api_key = api_key # Guardian API Key
        self.base_url = "https://content.guardianapis.com/search"

    def fetch_real_headlines(self, query="Euro economy"):
        """
        Fetches real headlines from The Guardian.
        """
        if not self.api_key:
            return None # Fallback to simulation if no key
            
        params = {
            "q": query,
            "api-key": self.api_key,
            "page-size": 5,
            "order-by": "newest"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            results = data.get('response', {}).get('results', [])
            headlines = [r['webTitle'] for r in results]
            return headlines if headlines else None
        except Exception as e:
            print(f"News Fetch Warning: {e}")
            return None

    def get_sentiment_score(self, currency_name):
        """
        Calculates sentiment score. Tries Real News first, then Fallback.
        """
        # Mapping for better Guardian search queries
        query_map = {
            "Euro": "Eurozone economy",
            "Pound": "UK economy GBP",
            "Yen": "Japan Yen economy",
            "Aussie": "Australia economy"
        }
        
        # Try fetching real news
        headlines = self.fetch_real_headlines(query_map.get(currency_name, currency_name))
        
        if not headlines:
            # High-quality fallback headlines if API fails or no key
            headlines = [f"{currency_name} market shows steady trends", 
                         f"Central bank monitors {currency_name} volatility"]
        
        scores = [self.analyzer.polarity_scores(h)['compound'] for h in headlines]
        return np.mean(scores)

    def generate_sentiment_series(self, currency_name, length):
        """
        Generates a sentiment time series.
        """
        base_score = self.get_sentiment_score(currency_name)
        # Random walk drift around the real-time sentiment base
        noise = np.random.normal(0, 0.05, length)
        sentiment_series = np.clip(np.cumsum(noise) + base_score, -1, 1)
        return sentiment_series

if __name__ == "__main__":
    # Test without key (will use fallback)
    fsa = FinancialSentimentAnalyzer()
    print(f"Sentiment for Euro (Fallback): {fsa.get_sentiment_score('Euro'):.2f}")
    
    # If you have a key, test it like this:
    # fsa_real = FinancialSentimentAnalyzer(api_key="your-key-here")
    # print(f"Sentiment for Euro (Real): {fsa_real.get_sentiment_score('Euro'):.2f}")
