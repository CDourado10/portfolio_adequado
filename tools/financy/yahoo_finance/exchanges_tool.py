from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel
from yahooquery import get_exchanges


class ExchangesToolInput(BaseModel):
    pass  # Não há parâmetros de entrada necessários


class ExchangesTool(BaseTool):
    name: str = "Exchanges Tool"
    description: str = (
        "Fetches and formats a list of available exchanges and their suffixes from Yahoo Finance."
    )
    args_schema: Type[BaseModel] = ExchangesToolInput

    def _run(self) -> str:
        """Fetch and format exchanges data."""
        exchanges_result = get_exchanges()
        output = ""

        # Verificar as colunas disponíveis
        print(exchanges_result.columns)

        # Iterar sobre os resultados das bolsas de valores
        for index, row in exchanges_result.iterrows():
            country = row['Country']
            market = row['Market, or Index']
            suffix = row['Suffix']

            output += (f"🌍 **Country:** {country} - **Exchange:** {market} - **Suffix:** {suffix}\n")

        return output.strip()  # Retorna a saída formatada

# 🔥 Executando como script independente
if __name__ == "__main__":
    # Criar instância da ferramenta
    exchanges_tool = ExchangesTool()
    
    # Executar a ferramenta
    result = exchanges_tool._run()
    
    # Exibir resultado
    print(result)
