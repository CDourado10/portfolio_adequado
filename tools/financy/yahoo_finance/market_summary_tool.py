from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from yahooquery import get_market_summary


class MarketSummaryToolInput(BaseModel):
    """Input schema for MarketSummaryTool."""
    country: str = Field("United States", description="Country for the market summary.")


class MarketSummaryTool(BaseTool):
    name: str = "Market Summary Tool"
    description: str = (
        "Fetches and formats market summary data from Yahoo Finance."
    )
    args_schema: Type[BaseModel] = MarketSummaryToolInput

    def _run(self, country: str = "United States") -> str:
        """Fetch and format market summary data."""
        market_summary_result = get_market_summary(country=country)
        output = ""

        # Iterar sobre os resultados do resumo do mercado
        for market in market_summary_result:
            full_exchange_name = market['fullExchangeName']
            symbol = market['symbol']
            regular_market_price = market['regularMarketPrice']['fmt']
            regular_market_change = market['regularMarketChange']['fmt']
            regular_market_change_percent = market['regularMarketChangePercent']['fmt']
            regular_market_previous_close = market['regularMarketPreviousClose']['fmt']
            regular_market_time = market['regularMarketTime']['fmt']

            output += f"📈 **{full_exchange_name} ({symbol})**\n"
            output += f"💰 **Current Price:** {regular_market_price}\n"
            output += f"🔼 **Change:** {regular_market_change} ({regular_market_change_percent})\n"
            output += f"📅 **Last Market Time:** {regular_market_time}\n"
            output += f"📉 **Previous Close:** {regular_market_previous_close}\n\n"

        return output.strip()  # Retorna a saída formatada

# 🔥 Executando como script independente
if __name__ == "__main__":
    # Criar instância da ferramenta
    market_summary_tool = MarketSummaryTool()
    
    # Executar a ferramenta
    result = market_summary_tool._run()
    
    # Exibir resultado
    print(result)
