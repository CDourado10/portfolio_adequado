from datetime import datetime, timedelta
import os
import time
from typing import List, Optional, Type

from crewai.tools import BaseTool
from dotenv import load_dotenv
from newsapi import NewsApiClient
from pydantic import BaseModel, Field

class MarketNewsInput(BaseModel):
    """Input parameters for the Market News Tool.
    
    This schema defines the input parameters for fetching financial market news.
    The tool uses NewsAPI.org to gather news from various trusted financial sources
    across different market sectors including traditional finance, cryptocurrencies,
    commodities, and emerging markets.
    """
    topics: Optional[List[str]] = Field(
        None,
        description="List of topics to search for in financial news. If not provided, "
                   "the tool will fetch general market news covering various sectors "
                   "including stocks, bonds, cryptocurrencies, and commodities."
    )

class MultiSourceMarketNews(BaseTool):
    """Advanced Market News Tool using NewsAPI.
    
    This tool provides comprehensive financial market news coverage by aggregating
    information from multiple trusted sources across different sectors. It covers:
    
    - Traditional financial markets (stocks, bonds, forex)
    - Cryptocurrencies and blockchain
    - Emerging markets and BRICS
    - Commodities and energy
    - FinTech and financial innovation
    - ESG and sustainable investments
    - Market analysis and research
    
    The tool automatically filters for relevant content and presents it in a
    structured format with source attribution and direct links to full articles.
    """
    name: str = "Global Market News Hub"
    description: str = (
        "Advanced financial market news aggregator that provides comprehensive coverage "
        "from trusted sources across multiple sectors including traditional markets, "
        "cryptocurrencies, commodities, and emerging markets. Returns structured news "
        "data with titles, descriptions, sources, and direct links to full articles."
    )
    args_schema: Type[BaseModel] = MarketNewsInput

    def _run(self, topics: Optional[List[str]] = None) -> str:
        """Fetch and process financial market news from multiple trusted sources.

        Args:
            topics: Optional list of topics to filter news by. If not provided,
                   returns general market news from various sectors.

        Returns:
            Formatted string containing relevant news articles with titles,
            descriptions, sources, and links to full content.

        Raises:
            Exception: If there are issues with API access or data processing.
        """
        try:
            # Load environment variables
            load_dotenv()
            api_key = os.getenv('NEWS_API_KEY')
            
            if not api_key:
                return "Error: NEWS_API_KEY not found in environment variables"
            
            # Initialize NewsApiClient
            newsapi = NewsApiClient(api_key=api_key)
            
            # Trusted financial news domains across sectors
            financial_domains = [
                # Major financial portals
                'reuters.com',
                'bloomberg.com',
                'cnbc.com',
                'ft.com',
                'wsj.com',
                'investing.com',
                'marketwatch.com',
                'finance.yahoo.com',
                'fool.com',
                'barrons.com',
                'seekingalpha.com',
                'thestreet.com',
                
                # Cryptocurrency and blockchain
                'coindesk.com',
                'cointelegraph.com',
                'decrypt.co',
                'bitcoinmagazine.com',
                'cryptonews.com',
                
                # Emerging markets and BRICS
                'business-standard.com',
                'economictimes.indiatimes.com',
                'scmp.com',
                'chinadaily.com.cn',
                'themoscowtimes.com',
                'businesstech.co.za',
                'zawya.com',
                'asiatimes.com',
                
                # Commodities and energy
                'oilprice.com',
                'mining.com',
                'agweb.com',
                'platts.com',
                
                # FinTech and innovation
                'fintechnews.org',
                'thefintechtimes.com',
                'tearsheet.co',
                'pymnts.com',
                
                # ESG and sustainable investments
                'esgtoday.com',
                'responsible-investor.com',
                'environmentalfinance.com',
                
                # Market analysis and research
                'morningstar.com',
                'tradingeconomics.com',
                'zacks.com',
                'benzinga.com',
                'investopedia.com'
            ]
            
            # Build search query
            query_parts = []
            
            # Add topics to query
            if topics:
                for topic in topics:
                    query_parts.append(f'"{topic}"')
            
            # Combine query parts
            if query_parts:
                query = ' OR '.join(query_parts)
            else:
                # Default query for comprehensive market coverage
                query = (
                    '"financial markets" OR '
                    '"market analysis" OR '
                    '"investment news" OR '
                    '"economic news" OR '
                    '"market trends"'
                )
                
            # Set date range (last 7 days)
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Fetch news using the official API
            all_articles = newsapi.get_everything(
                q=query,
                from_param=from_date,
                domains=','.join(financial_domains),
                language='en',
                sort_by='relevancy',
                page_size=20
            )
            
            if not all_articles or 'articles' not in all_articles or not all_articles['articles']:
                return "No news found for the specified criteria."
            
            # Sort articles by date (newest first)
            articles = sorted(
                all_articles['articles'],
                key=lambda x: x.get('publishedAt', ''),
                reverse=True
            )
            
            # Format news in structured text
            news_text = []
            
            # Header
            news_text.append("")
            news_text.append("FINANCIAL MARKET NEWS SUMMARY")
            news_text.append("-" * 70)
            news_text.append("")
            
            # Add result count
            total_results = all_articles.get('totalResults', 0)
            news_text.append(f"Total Results: {total_results}")
            if total_results > 20:
                news_text.append("(Showing top 20 most relevant articles)")
            news_text.append("")
            
            # Format date helper function
            def format_date(date_str):
                try:
                    dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
                    return dt.strftime('%Y-%m-%d %H:%M')
                except:
                    return date_str
            
            # Add articles
            for idx, article in enumerate(articles, 1):
                source = article.get('source', {}).get('name', 'Unknown Source')
                title = article.get('title', 'No title')
                if title:
                    title = title.strip()
                date = format_date(article.get('publishedAt', 'N/A'))
                description = article.get('description', 'No description available')
                if description:
                    description = description.strip()
                url = article.get('url', 'N/A')
                
                news_text.append(f"[Article {idx}]")
                news_text.append("-" * 70)
                news_text.append(f"Source: {source}")
                news_text.append(f"Title: {title}")
                news_text.append(f"Date: {date}")
                if description and description.lower() != 'none':
                    # Limit description length
                    if len(description) > 200:
                        description = description[:197] + "..."
                    news_text.append(f"Summary: {description}")
                news_text.append(f"Full Article: {url}")
                news_text.append("")
            
            # Add search parameters footer
            news_text.append("-" * 70)
            news_text.append("SEARCH PARAMETERS")
            news_text.append("-" * 70)
            if topics:
                news_text.append(f"Topics: {', '.join(topics)}")
            news_text.append(f"Date Range: {from_date} to present")
            news_text.append("-" * 70)
            
            # Handle special characters
            return "\n".join(news_text).encode('utf-8', errors='ignore').decode('utf-8')
            
        except Exception as e:
            error_msg = str(e)
            return f"""Error fetching market news: {error_msg}

Error Details:
-------------
{error_msg}

Troubleshooting Steps:
1. Verify NEWS_API_KEY in .env file
2. Check API key validity at newsapi.org
3. Confirm internet connectivity
4. Check API rate limits
5. Try reducing the number of domains if experiencing timeout issues"""

if __name__ == '__main__':
    try:
        # Initialize the tool
        market_news_tool = MultiSourceMarketNews()
        
        # Example 1: Search news by specific topics
        print("\n=== Example 1: Search by Topics ===")
        result = market_news_tool.run(
            topics=["ECONOMY", "FISCAL_POLICY"]
        )
        print(result)
        
        # Wait to avoid rate limiting
        time.sleep(2)
        
        # Example 2: Search general market news
        print("\n=== Example 2: General Market News ===")
        result = market_news_tool.run()
        print(result)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
