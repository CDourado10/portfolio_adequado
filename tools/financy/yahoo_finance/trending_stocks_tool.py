from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from yahooquery import get_trending


class TrendingStocksToolInput(BaseModel):
    country: str = Field("United States", description="Country for trending stocks.")


class TrendingStocksTool(BaseTool):
    name: str = "Trending Stocks Tool"
    description: str = (
        "Fetches and formats trending stocks data from Yahoo Finance."
    )
    args_schema: Type[BaseModel] = TrendingStocksToolInput

    def _run(self, country: str = "United States") -> str:
        """Fetch and format trending stocks data."""
        trending_stocks_result = get_trending(country=country)
        output = ""

        # Iterar sobre os resultados das ações em alta
        for stock in trending_stocks_result['quotes']:
            symbol = stock['symbol']
            output += f"📈 **Stock Trending:** {symbol}\n"

        return output.strip()  # Retorna a saída formatada

# 🔥 Executando como script independente
if __name__ == "__main__":
    # Criar instância da ferramenta
    trending_stocks_tool = TrendingStocksTool()
    
    # Executar a ferramenta
    result = trending_stocks_tool._run("United States")
    
    # Exibir resultado
    print(result)
