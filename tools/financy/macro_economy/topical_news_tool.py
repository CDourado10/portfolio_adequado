from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel
import requests
import os
import time
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

# Mantido para referência dos tópicos suportados
SUPPORTED_TOPICS = {
    'blockchain': 'Blockchain technology and developments',
    'earnings': 'Company earnings reports and updates',
    'ipo': 'Initial Public Offerings',
    'mergers_and_acquisitions': 'Mergers & Acquisitions activities',
    'financial_markets': 'Financial Markets overview and analysis',
    'economy_fiscal': 'Economy - Fiscal Policy (e.g., tax reform, government spending)',
    'economy_monetary': 'Economy - Monetary Policy (e.g., interest rates, inflation)',
    'economy_macro': 'Economy - Macro/Overall trends and indicators',
    'energy_transportation': 'Energy & Transportation sector news',
    'finance': 'General finance news and developments',
    'life_sciences': 'Life Sciences sector news',
    'manufacturing': 'Manufacturing sector news',
    'real_estate': 'Real Estate & Construction sector news',
    'retail_wholesale': 'Retail & Wholesale sector news',
    'technology': 'Technology sector news'
}

class MarketNewsInput(BaseModel):
    """Input schema for Topical News Analyzer.
    
    This schema defines the input parameters for fetching financial news
    using Alpha Vantage's NEWS_SENTIMENT endpoint. The tool provides
    detailed news coverage with sentiment analysis and relevance scoring
    across all supported financial topics.
    """
    pass

class TopicalNewsAnalyzer(BaseTool):
    """Financial Topics & Sentiment Analyzer.
    
    This tool uses Alpha Vantage's NEWS_SENTIMENT endpoint to provide:
    - Real-time financial news across all topics
    - Article summaries and URLs
    - Sentiment analysis for each article
    - Relevance scoring
    - Topic classification and distribution
    
    Note: Requires a premium Alpha Vantage API key.
    """
    name: str = "Financial Topics & Sentiment Analyzer"
    description: str = (
        "Advanced news analysis tool that provides comprehensive financial news with "
        "sentiment analysis and detailed summaries across all market topics. Uses "
        "Alpha Vantage's premium NEWS_SENTIMENT endpoint. Automatically fetches "
        "news from the last 7 days."
    )
    args_schema: Type[BaseModel] = MarketNewsInput

    def _run(self) -> str:
        """
        Fetch and analyze financial market news from the last 7 days.
        
        Returns:
            Formatted string containing news articles with summaries,
            sentiment analysis, and metadata, sorted by relevance.
        
        Note: Requires a premium Alpha Vantage API key.
        """
        # Load environment variables
        load_dotenv()
         
        base_url = 'https://www.alphavantage.co/query'
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        if not api_key:
            return """Error: API Key not found. Add ALPHA_VANTAGE_API_KEY to .env file
            
Note: The NEWS_SENTIMENT endpoint requires a premium API key.
Visit https://www.alphavantage.co/premium/ for more information.

Alternative solutions:
1. Use a free news API like NewsAPI.org
2. Use the basic Alpha Vantage endpoints for market data
3. Upgrade to a premium Alpha Vantage plan"""
        
        # Calculate time_from (7 days ago)
        time_from = (datetime.now() - timedelta(days=7)).strftime('%Y%m%dT%H%M')

        
        # Request parameters
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': api_key,
            'sort': 'RELEVANCE',  # Ordenação por relevância
            'limit': '50',  # 50 resultados
            'time_from': time_from
        }
        
        try:
            response = requests.get(base_url, params=params)
            
            response.raise_for_status()
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                return f"""Error decoding JSON response: {str(e)}
                
Note: This error often occurs when using a free API key with premium endpoints.
Please check your API key type and permissions."""
            
            if isinstance(data, dict) and 'Note' in data:
                return f"""API Rate Limit Message: {data['Note']}
                
Note: The NEWS_SENTIMENT endpoint requires a premium API key.
Visit https://www.alphavantage.co/premium/ for more information."""
                
            if isinstance(data, dict) and 'Error Message' in data:
                return f"""API Error Message: {data['Error Message']}
                
Note: This error might occur if:
1. You're using a free API key with premium endpoints
2. The endpoint is not available in your current plan
3. The API key doesn't have the necessary permissions

Please check your API key type and permissions."""
            
            if not isinstance(data, dict) or 'feed' not in data:
                return f"""Unexpected API response format: {json.dumps(data, indent=2)[:500]}...
                
Note: This error often occurs when:
1. Using a free API key with premium endpoints
2. The endpoint is not available in your current plan
3. The API response format has changed

Please check your API key type and permissions."""
            
            # Format news in structured text
            news_text = []
            
            # Header
            news_text.append("")
            news_text.append("FINANCIAL TOPICS & SENTIMENT ANALYSIS")
            news_text.append("=" * 80)
            news_text.append("")
            
            # Summary Section
            news_text.append("SUMMARY")
            news_text.append("-" * 80)
            news_text.append(f"Total Articles: {len(data['feed'])}")
            news_text.append(f"Time Range: {time_from} to present")
            news_text.append("")
            
            # Topic Analysis
            topic_stats = {}
            for item in data['feed']:
                if 'topics' in item:
                    for topic_info in item['topics']:
                        topic = topic_info.get('topic', '')
                        score = float(topic_info.get('relevance_score', 0))
                        if topic not in topic_stats:
                            topic_stats[topic] = {'count': 0, 'total_score': 0}
                        topic_stats[topic]['count'] += 1
                        topic_stats[topic]['total_score'] += score
            
            if topic_stats:
                news_text.append("TOPIC DISTRIBUTION")
                news_text.append("-" * 80)
                # Ordenar tópicos por contagem e score médio
                sorted_topics = sorted(
                    topic_stats.items(),
                    key=lambda x: (x[1]['count'], x[1]['total_score']/x[1]['count']),
                    reverse=True
                )
                for topic, stats in sorted_topics:
                    avg_score = stats['total_score'] / stats['count']
                    news_text.append(
                        f"- {topic.upper()}: {stats['count']} articles "
                        f"(Avg. Relevance: {avg_score:.2f})"
                    )
                news_text.append("")
            
            # Articles Section (já ordenados por relevância pela API)
            news_text.append("ARTICLES BY RELEVANCE")
            news_text.append("=" * 80)
            news_text.append("")
            
            for idx, item in enumerate(data['feed'], 1):
                news_text.append(f"[Article {idx}]")
                news_text.append("-" * 80)
                
                # Title and Source
                news_text.append(f"Title: {item.get('title', 'No title')}")
                news_text.append(
                    f"Source: {item.get('source', 'Unknown Source')} | "
                    f"Date: {item.get('time_published', 'N/A')}"
                )
                
                # Topics Section
                if 'topics' in item:
                    news_text.append("\nTopics & Relevance:")
                    topics_list = sorted(
                        item['topics'],
                        key=lambda x: float(x.get('relevance_score', 0)),
                        reverse=True
                    )
                    for topic in topics_list:
                        news_text.append(
                            f"  • {topic.get('topic', '').upper()}: "
                            f"Score {float(topic.get('relevance_score', 0)):.2f}"
                        )
                
                # Sentiment Section
                sentiment = item.get('overall_sentiment_label', 'N/A').title()
                score = item.get('overall_sentiment_score', 'N/A')
                news_text.append(f"\nSentiment: {sentiment} (Score: {score})")
                
                # Content Section
                if item.get('summary'):
                    news_text.append("\nSummary:")
                    news_text.append(item['summary'])
                
                news_text.append("\nRead More:")
                news_text.append(item.get('url', 'N/A'))
                news_text.append("")
                news_text.append("=" * 80)
                news_text.append("")
            
            return "\n".join(news_text)
            
        except requests.exceptions.RequestException as e:
            return f"""Error fetching news: {str(e)}
            
Please check:
1. Your internet connection
2. API endpoint availability
3. API key validity and permissions"""

if __name__ == '__main__':
    try:
        # Initialize the tool
        news_analyzer = TopicalNewsAnalyzer()
        
        # Buscar todas as notícias ordenadas por relevância
        print("\n=== Financial News Analysis ===")
        result = news_analyzer.run()
        print(result)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting Steps:")
        print("1. Check if .env file exists with ALPHA_VANTAGE_API_KEY")
        print("2. Confirm if your API key is valid")
        print("3. Check your internet connection")
        print("4. Verify if you have reached the API rate limit")
        print("5. Make sure your API key has access to the News API endpoint")
