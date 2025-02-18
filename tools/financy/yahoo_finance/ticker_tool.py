from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from yahooquery.ticker import Ticker


class TickerToolInput(BaseModel):
    symbol: str = Field(..., description="The stock symbol for the asset to fetch information about.")


class TickerTool(BaseTool):
    name: str = "Asset Information Tool"
    description: str = (
        "Fetches detailed information about a specific asset, including profile data and executives of the company."
    )
    args_schema = TickerToolInput

    def _run(self, symbol: str) -> str:
        """Fetch and format ticker data."""
        try:
            ticker = Ticker(symbol)
            output = f"\nTicker: {ticker.symbols}\n\n"

            # Obtendo e imprimindo o perfil do ativo
            asset_profile = ticker.asset_profile
            output += f"\n\n**Asset Information for Ticker {symbol}:**\n"
            for key, value in asset_profile[symbol].items():
                if key == 'companyOfficers':
                    if 'companyOfficers' in asset_profile[symbol]:
                        output += f"\n**Executives:**\n"
                        for officer in value:
                            name = officer['name']
                            title = officer['title']
                            age = officer.get('age', 'N/A')  # Usar 'N/A' se a idade não estiver disponível
                            output += f"- **{name}** (Idade: {age}) - {title}\n"
                else:
                    output += f"- **{key.replace('_', ' ').capitalize()}:** {value}\n"

            return output.strip()  # Retorna a saída formatada
        except Exception as e:
            return f"Erro ao obter dados do ticker: {str(e)}"

# 🔥 Executando como script independente
if __name__ == "__main__":
    # Criar instância da ferramenta
    ticker_tool = TickerTool()
    
    # Executar a ferramenta com um símbolo de exemplo
    result = ticker_tool._run('AAPL')
    
    # Exibir resultado
    print(result)
