from typing import List, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from yahooquery import Screener


class ScreenerToolInput(BaseModel):
    """Input schema for ScreenerTool."""

    screeners: List[str] = Field(
        ..., description="List of screener names to fetch from Yahoo Finance."
    )
    count: int = Field(
        3, description="Number of top companies to retrieve from each screener."
    )


class ScreenerTool(BaseTool):
    name: str = "Yahoo Finance Screener Tool"
    description: str = (
        "Fetches the top companies from specified Yahoo Finance screeners, "
        "including details such as stock prices, market capitalization, and performance."
    )
    args_schema: Type[BaseModel] = ScreenerToolInput

    def _run(self, screeners: List[str], count: int = 3) -> str:
        """Fetch and format data for multiple screeners from Yahoo Finance."""

        # Criar instância do Screener
        s = Screener()

        # Obter os dados de todos os screeners solicitados
        screeners_data = s.get_screeners(screeners, count=count)

        def safe_format(value, default="N/A"):
            """ Converts numeric values to formatted string or returns 'N/A' if not numeric. """
            try:
                return f"{float(value):.2f}" if value is not None else default
            except ValueError:
                return default

        output = ""

        # Iterar sobre cada screener buscado
        for screener_name in screeners:
            data = screeners_data.get(screener_name, {})

            # Informações gerais do screener
            sector_title = data.get("title", "N/A")
            sector_description = data.get("description", "N/A")
            total_stocks = data.get("total", "N/A")

            # Obter lista de empresas
            companies = data.get("quotes", [])

            # Construir saída formatada para cada screener
            output += f"""
📊 **Sector:** {sector_title}
🔹 **Description:** {sector_description}
🔹 **Total Stocks in Sector:** {total_stocks}

---\n"""

            # Iterar sobre as empresas listadas
            for company in companies:
                company_name = company.get("longName", "N/A")
                ticker = company.get("symbol", "N/A")
                exchange = company.get("fullExchangeName", "N/A")
                currency = company.get("currency", "N/A")

                # Preços de mercado
                last_price = safe_format(company.get("regularMarketPrice"))
                change_percent = safe_format(company.get("regularMarketChangePercent"))
                day_high = safe_format(company.get("regularMarketDayHigh"))
                day_low = safe_format(company.get("regularMarketDayLow"))
                day_open = safe_format(company.get("regularMarketOpen"))
                previous_close = safe_format(company.get("regularMarketPreviousClose"))
                volume = company.get("regularMarketVolume", "N/A")

                # Desempenho em 52 semanas
                fifty_two_week_high = safe_format(company.get("fiftyTwoWeekHigh"))
                fifty_two_week_low = safe_format(company.get("fiftyTwoWeekLow"))
                fifty_two_week_change_percent = safe_format(company.get("fiftyTwoWeekChangePercent"))

                # Indicadores financeiros
                market_cap = company.get("marketCap", "N/A")
                pe_ratio = safe_format(company.get("forwardPE"))
                dividend_yield = safe_format(company.get("dividendYield"))

                output += f"""🏢 **Company**
- **Name:** {company_name}
- **Ticker:** {ticker}
- **Exchange:** {exchange}
- **Currency:** {currency}

💰 **Market Prices**
- **Last Price:** ${last_price}
- **Change in day:** {change_percent}%
- **Open:** ${day_open}
- **Day High:** ${day_high}
- **Day Low:** ${day_low}
- **Previous Close:** ${previous_close}
- **Volume:** {volume}

📈 **52 weeks performance**
- **High:** ${fifty_two_week_high}
- **Low:** ${fifty_two_week_low}
- **Percent Change:** {fifty_two_week_change_percent}%

📊 **Financial Indicators**
- **Market Capitalization:** ${market_cap}
- **Forward P/E (Price/Earnings):** {pe_ratio}
- **Dividend Yield:** {dividend_yield}%

---
"""
        return output

# 🔥 Executando como script independente
if __name__ == "__main__":
    # Criar instância da ferramenta
    screener_tool = ScreenerTool()

    # Definir os screeners a serem buscados
    screeners_to_fetch = ["most_actives", "day_gainers", "agricultural_inputs"]

    # Definir o número de empresas por screener
    companies_count = 5

    # Executar a ferramenta
    result = screener_tool._run(screeners=screeners_to_fetch, count=companies_count)

    # Exibir resultado
    print(result)