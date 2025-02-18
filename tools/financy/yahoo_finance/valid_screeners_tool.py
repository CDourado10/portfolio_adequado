from crewai.tools import BaseTool
from pydantic import BaseModel
from yahooquery.utils.screeners import SCREENERS


class ValidScreenersToolInput(BaseModel):
    pass  # Não há parâmetros de entrada necessários


class ValidScreenersTool(BaseTool):
    name: str = "Valid Screeners Tool"
    description: str = (
        "Fetches and displays valid screener names from Yahoo Finance."
    )
    args_schema = ValidScreenersToolInput

    def _run(self) -> str:
        """Fetch and format valid screener names."""
        valid_screeners = list(SCREENERS.keys())
        output = "Available screeners:\n" + ', '.join(valid_screeners) + '\n'
        return output.strip()  # Retorna a saída formatada

# 🔥 Executando como script independente
if __name__ == "__main__":
    # Criar instância da ferramenta
    valid_screeners_tool = ValidScreenersTool()
    
    # Executar a ferramenta
    result = valid_screeners_tool._run()
    
    # Exibir resultado
    print(result)
