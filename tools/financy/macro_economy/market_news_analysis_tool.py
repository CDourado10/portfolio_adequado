from typing import List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging
import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from crews.market_intelligence.market_intelligence_crew import MarketNewsAnalysisCrew

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketNewsAnalysisInput(BaseModel):
    """Input schema for MarketNewsAnalysisTool."""

class MarketNewsAnalysisTool(BaseTool):
    """Tool for market news analysis and intelligence."""
    
    name: str = Field(default="MarketNewsAnalysisTool")
    description: str = Field(default=(
        "Performs comprehensive market news analysis using a specialized crew. "
        "Processes and analyzes market news from multiple sources, identifying "
        "trends, patterns, and potential market impacts. Includes data cleaning, "
        "categorization, trend identification, impact assessment, and insight "
        "generation. Returns a detailed analysis report with actionable insights "
        "derived from market news and events."
    ))
    args_schema: Type[BaseModel] = MarketNewsAnalysisInput

    def _run(self) -> str:
        """Run the tool."""
        try:
            logger.info("Starting market news analysis")
            
            # Create and run the crew
            crew = MarketNewsAnalysisCrew()
            results = crew.run_crew()
            
            logger.info("Market news analysis completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Error executing market news analysis: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = MarketNewsAnalysisTool()
    print(tool._run())
