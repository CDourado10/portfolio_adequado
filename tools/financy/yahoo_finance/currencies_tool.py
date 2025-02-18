from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from yahooquery import get_currencies


class CurrenciesToolInput(BaseModel):
    """Input schema for CurrenciesTool."""
    pass


class CurrenciesTool(BaseTool):
    name: str = "Currencies Tool"
    description: str = (
        "Fetches and formats a list of currencies from Yahoo Finance, providing both short and long names along with their symbols."
    )
    args_schema: Type[BaseModel] = CurrenciesToolInput

    def _run(self) -> str:
        """Fetch and format currency data."""
        currencies_result = get_currencies()
        output = ""

        # Iterar sobre os resultados das moedas
        for currency in currencies_result:
            short_name = currency['shortName']
            long_name = currency['longName']
            symbol = currency['symbol']
            local_long_name = currency['localLongName']

            output += f"💱 **{long_name}** ({short_name}) - Symbol: {symbol} - Local Name: {local_long_name}\n"

        return output.strip()  # Retorna a saída formatada

# 🔥 Executando como script independente
if __name__ == "__main__":
    # Criar instância da ferramenta
    currencies_tool = CurrenciesTool()
    
    # Executar a ferramenta
    result = currencies_tool._run()
    
    # Exibir resultado
    print(result)
