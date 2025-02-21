from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging
import os
import sys

# Adicionando o diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from crews.asset_exploration.asset_exploration_crew import AssetExplorationCrew

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetExplorationInput(BaseModel):
    """Input schema para a ferramenta de exploração de ativos."""
    risk_level: str = Field(..., description="Desired risk level (low, medium, high).")
    investment_horizon: str = Field(..., description="Investment horizon (short, medium, long).")
    asset_types: list[str] = Field(..., description="List of asset types for analysis (e.g. stocks, ETFs, cryptocurrencies, fixed income).")

class AssetExplorationTool(BaseTool):
    """Ferramenta para explorar e filtrar ativos de investimento com base nos inputs de risco e prazo."""

    name: str = Field(default="AssetExplorationTool")
    description: str = Field(default=(
        "Explores investment assets based on the provided criteria:\n"
        "- Risk level\n"
        "- Investment horizon\n"
        "- Asset type\n"
        "Returns a structured list of recommended assets."
    ))
    args_schema: Type[BaseModel] = AssetExplorationInput

    def _run(self, risk_level: str, investment_horizon: str, asset_types: list[str]) -> str:
        """Executa a exploração de ativos com base nos inputs da task."""
        try:
            logger.info(f"Starting asset exploration for risk={risk_level}, investment horizon={investment_horizon}, asset types={asset_types}")

            # Criando e executando a Crew de exploração de ativos com os parâmetros fornecidos
            crew = AssetExplorationCrew()
            results = crew.crew().kickoff(inputs={
                "risk_level": risk_level,
                "investment_horizon": investment_horizon,
                "asset_types": asset_types
            })
            
            logger.info("Asset exploration completed successfully.")
            return results
            
        except Exception as e:
            error_msg = f"Error exploring assets: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = AssetExplorationTool()
    print(tool._run(risk_level="medium", investment_horizon="medium", asset_types=["stocks", "etfs", "cryptocurrencies"]))
