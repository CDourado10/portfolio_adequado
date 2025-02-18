from crewai.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field
from yahooquery import search
from datetime import datetime
import json


class YahooNewsToolInput(BaseModel):
    query: str = Field(..., description="Query to search for news articles.")
    quotes_count: int = Field(10, description="Number of quotes to retrieve.")
    news_count: int = Field(10, description="Number of news articles to retrieve.")
    first_quote: bool = Field(False, description="Whether to return the first quote.")


class YahooNewsTool(BaseTool):
    name: str = "Yahoo News Tool"
    description: str = (
        "Fetches news articles from Yahoo Finance based on a search query."
    )
    args_schema: Type[BaseModel] = YahooNewsToolInput

    def _run(self, query: str, quotes_count: int = 10, news_count: int = 10, first_quote: bool = False) -> str:
        """Fetch and format news articles based on a search query."""
        search_result = search(query, quotes_count=quotes_count, news_count=news_count, first_quote=first_quote)
        output = ""

        # Iterar sobre as notícias buscadas
        for news in search_result['news']:
            title = news['title']
            publisher = news['publisher']
            link = news['link']
            publish_time = datetime.fromtimestamp(news['providerPublishTime']).strftime('%d/%m/%Y %H:%M:%S')

            output += f" **{title}** - {publisher} (Published on: {publish_time})\n{link}\n\n"

        return output.strip()  # Retorna a saída formatada

# Executando como script independente
if __name__ == "__main__":
    # Criar instância da ferramenta
    yahoo_news_tool = YahooNewsTool()

    # Definir a consulta de notícias
    news_query = "macroeconomy"

    # Executar a ferramenta
    result = yahoo_news_tool._run(query=news_query)

    # Exibir resultado
    print(result)
