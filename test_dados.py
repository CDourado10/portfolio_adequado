from yahooquery import search, get_market_summary, get_trending, get_currencies, get_exchanges
from yahooquery.screener import Screener
from yahooquery.utils.screeners import SCREENERS
from yahooquery.research import Research
from yahooquery.ticker import Ticker
from datetime import datetime

# Instanciando a classe Ticker com um símbolo de exemplo
symbols = 'AAPL'
ticker = Ticker(symbols)
print("\nTicker:")
print(ticker.Ticker)

# Obtendo e imprimindo o perfil do ativo
asset_profile = ticker.asset_profile
print("Perfil do Ativo:")
print(asset_profile)

# Obtendo e imprimindo dados financeiros
financial_data = ticker.financial_data
print("\nDados Financeiros:")
print(financial_data)

ticker = Ticker("Bitcoin")  # Exemplo com a Apple
modules = ticker.get_modules(["financialData", "assetProfile"])
try:
    try:
        news_data = ticker.news(count=5)
        print("\nNotícias:")
        print(news_data)
    except Exception as e:
        print("\nErro ao buscar notícias:", str(e))
except Exception as e:
    print("\nErro ao buscar notícias:", str(e))

financial_data = ticker.all_financial_data(frequency="q")
print("\nDados Financeiros (Todos):")
print(financial_data)

print("\nModulos:")
print(modules)