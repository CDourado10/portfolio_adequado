from typing import Type, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging
import os
import sys

# Adicionando o diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from crews.portfolio_optimization.portfolio_optimization_crew import PortfolioOptimizationCrew

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizationInput(BaseModel):
    """Input schema for the portfolio optimization tool."""
    risk_level: str = Field(..., description="Portfolio risk level (low, medium, high).")
    investment_horizon: str = Field(..., description="Investment horizon (short, medium, long).")
    selected_assets: List[str] = Field(..., description="List of selected assets to compose the portfolio.")

class PortfolioOptimizationTool(BaseTool):
    """Tool for optimizing portfolios based on risk, time horizon, and selected assets."""

    name: str = Field(default="PortfolioOptimizationTool")
    description: str = Field(default=(
        "Optimizes portfolio allocation using advanced risk and return optimization methodologies.\n"
        "- Utilizes different optimization models depending on the risk {risk_level}.\n"
        "- Considers the investment horizon {investment_horizon} for appropriate balancing.\n"
        "- Works only with the selected assets {selected_assets}."
    ))
    args_schema: Type[BaseModel] = PortfolioOptimizationInput

    def _run(self, risk_level: str, investment_horizon: str, selected_assets: List[str]) -> str:
        """Executes portfolio optimization based on the provided inputs."""
        try:
            logger.info(f"Starting portfolio optimization with risk={risk_level}, horizon={investment_horizon}, assets={selected_assets}")

            # Creating and executing the portfolio optimization Crew
            crew = PortfolioOptimizationCrew()
            results = crew.crew().kickoff(inputs={
                "risk_level": risk_level,
                "investment_horizon": investment_horizon,
                "selected_assets": selected_assets
            })
            
            logger.info("Portfolio optimization completed successfully.")
            return results
            
        except Exception as e:
            error_msg = f"Error in portfolio optimization: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = PortfolioOptimizationTool()
    print(tool._run(risk_level="medium", investment_horizon="long", selected_assets=["AAPL", "TSLA", "MSFT", "GOOGL", "BTC-USD", "^GSPC"]))
