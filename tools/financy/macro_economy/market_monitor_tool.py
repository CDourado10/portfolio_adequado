from typing import List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging
import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from crews.market_monitor.market_monitor_crew import MarketMonitorCrew

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketMonitorInput(BaseModel):
    """Input schema for MarketMonitorTool."""

class MarketMonitorTool(BaseTool):
    """Tool for integrated market monitoring and analysis."""
    
    name: str = Field(default="MarketMonitorTool")
    description: str = Field(default=(
        "Performs comprehensive market monitoring and analysis using a specialized crew. "
        "Monitors and analyzes 207 assets across different markets: "
        "23 macroeconomic indicators, 21 government bonds, 27 currency pairs, "
        "93 commodities, 23 stock indices, and 20 cryptocurrencies. "
        "Returns an integrated analysis report with market insights, correlations, "
        "risks, and actionable recommendations."
    ))
    args_schema: Type[BaseModel] = MarketMonitorInput

    def _run(self) -> str:
        """Run the tool."""
        try:
            logger.info("Starting market monitoring and analysis")
            
            # Create and run the crew
            crew = MarketMonitorCrew()
            results = crew.run_crew()
            
            logger.info("Market monitoring and analysis completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Error executing market monitoring: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = MarketMonitorTool()
    print(tool._run())
