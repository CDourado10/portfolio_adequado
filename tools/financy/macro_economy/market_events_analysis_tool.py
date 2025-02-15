from typing import List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging
import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from crews.market_events_analysis.market_events_analysis_crew import MarketEventsAnalysisCrew

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketEventsAnalysisInput(BaseModel):
    """Input schema for MarketEventsAnalysisTool."""

class MarketEventsAnalysisTool(BaseTool):
    """Tool for analyzing economic and corporate market events."""
    
    name: str = Field(default="MarketEventsAnalysisTool")
    description: str = Field(default=(
        "Performs comprehensive analysis of market events using a specialized crew. "
        "Processes and analyzes both economic calendar events and corporate earnings "
        "releases. Includes economic calendar collection, earnings calendar collection, "
        "macro analysis, trend identification, earnings analysis, sector impact "
        "assessment, and integrated market analysis. Returns a detailed report with "
        "insights about market events and their potential impacts."
    ))
    args_schema: Type[BaseModel] = MarketEventsAnalysisInput

    def _run(self) -> str:
        """Run the tool."""
        try:
            logger.info("Starting market events analysis")
            
            # Create and run the crew
            crew = MarketEventsAnalysisCrew()
            results = crew.run_crew()
            
            logger.info("Market events analysis completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Error executing market events analysis: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = MarketEventsAnalysisTool()
    print(tool._run())
