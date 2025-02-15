import requests
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

def get_market_news(tickers=None, topics=None, time_from=None):
    """
    Obtém notícias e sentimentos do mercado usando a Alpha Vantage API
    """
    base_url = 'https://www.alphavantage.co/query'
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not api_key:
        raise ValueError("API Key não encontrada. Adicione ALPHA_VANTAGE_API_KEY ao arquivo .env")
    
    # Parâmetros da requisição
    params = {
        'function': 'NEWS_SENTIMENT',
        'apikey': api_key,
        'sort': 'RELEVANCE'
    }
    
    # Adiciona tickers se fornecidos
    if tickers:
        params['tickers'] = ','.join(tickers)
    
    # Adiciona tópicos se fornecidos
    if topics:
        params['topics'] = ','.join(topics)
    
    # Adiciona data inicial se fornecida
    if time_from:
        params['time_from'] = time_from
    
    try:
        print("\nParâmetros da requisição:")
        safe_params = params.copy()
        safe_params['apikey'] = '***'
        print(safe_params)
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        print(f"Status code: {response.status_code}")
        
        data = response.json()
        print("Chaves na resposta:", list(data.keys()) if isinstance(data, dict) else "Resposta não é um dicionário")
        
        if 'feed' not in data:
            print("\nAviso: Nenhuma notícia encontrada ou erro na resposta")
            print("Resposta da API:", data)
            return None
            
        print(f"Número de notícias encontradas: {len(data['feed'])}")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {str(e)}")
        return None
    except Exception as e:
        print(f"Erro ao processar dados: {str(e)}")
        return None

def format_news(news_data):
    if not news_data.get('feed'):
        print("Nenhuma notícia encontrada.")
        return

    # Ordena as notícias por sentimento
    articles = sorted(news_data['feed'], key=lambda x: float(x.get('overall_sentiment_score', 0)), reverse=True)
    
    # Pega a notícia com maior e menor sentimento
    highest_sentiment = articles[0]
    lowest_sentiment = articles[-1]
    
    for article in [highest_sentiment, lowest_sentiment]:
        # Formata a data
        time_published = datetime.strptime(
            article['time_published'],
            '%Y%m%dT%H%M%S'
        ).strftime('%d/%m/%Y %H:%M')
        
        # Define o emoji baseado no sentimento
        sentiment_score = float(article.get('overall_sentiment_score', 0))
        if sentiment_score > 0.35:
            sentiment_emoji = "🟢"  # Muito positivo
        elif sentiment_score > 0:
            sentiment_emoji = "🟡"  # Levemente positivo
        elif sentiment_score > -0.35:
            sentiment_emoji = "🟠"  # Levemente negativo
        else:
            sentiment_emoji = "🔴"  # Muito negativo
        
        # Formata os tópicos
        topics = []
        for topic in article.get('topics', []):
            if isinstance(topic, dict):
                topics.append(topic.get('topic', ''))
            else:
                topics.append(topic)
        
        print("\n" + "="*100)
        print(f"\nSentimento: {sentiment_emoji} ({sentiment_score:.2f})")
        print(f"Data: {time_published}")
        print(f"Título: {article['title']}")
        print(f"Tópicos: {', '.join(topics)}")
        print("\nConteúdo da notícia:")
        print("-"*50)
        print(article['summary'])
        print("-"*50)
        print(f"URL: {article['url']}")
        print("="*100 + "\n")

# Configuração dos parâmetros de busca
search_params = [
    {
        "description": "Política Monetária e Fiscal",
        "tickers": None,
        "topics": ["economy_monetary", "economy_fiscal"],
    },
    {
        "description": "Macroeconomia",
        "tickers": None,
        "topics": ["economy_macro"],
    },
    {
        "description": "Mercados Financeiros",
        "tickers": None,
        "topics": ["financial_markets"],
    }
]

# Calcula a data inicial (7 dias atrás)
time_from = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y%m%dT%H%M')
print(f"\n📅 Buscando notícias desde: {datetime.strptime(time_from, '%Y%m%dT%H%M').strftime('%d/%m/%Y %H:%M')} UTC")

print("\n📊 Monitoramento de Notícias e Sentimentos (Alpha Vantage):")

for params in search_params:
    print(f"\n📌 {params['description']}:")
    print(f"🔍 Tópicos: {', '.join(params['topics'])}")
    
    try:
        news_data = get_market_news(
            tickers=params['tickers'],
            topics=params['topics'],
            time_from=time_from
        )
        
        if news_data:
            print("\n📰 Notícias encontradas:")
            format_news(news_data)
        else:
            print("\nℹ️ Nenhuma notícia encontrada para os parâmetros especificados")
            
    except Exception as e:
        print(f"\n❌ Erro ao processar busca: {str(e)}")
    
    print("\n" + "═" * 80)