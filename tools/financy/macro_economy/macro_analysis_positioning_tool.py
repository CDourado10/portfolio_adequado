from typing import List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging
import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from crews.macro_analysis_and_positioning_strategy.macro_analysis_and_positioning_strategy_crew import MarketMonitorCrew

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroAnalysisPositioningInput(BaseModel):
    """Input schema for MacroAnalysisPositioningTool."""
    pass

class MacroAnalysisPositioningTool(BaseTool):
    """Tool for comprehensive market analysis and strategic positioning."""
    
    name: str = Field(default="MacroAnalysisPositioningTool")
    description: str = Field(default=(
        "Conducts comprehensive market analysis and develops positioning strategies. "
        "Utilizes a specialized team for: \n"
        "1. Impact analysis between sectors\n"
        "2. Market event monitoring\n"
        "3. Analysis of relevant news\n"
        "4. Generation of strategic reports\n"
        "Returns a detailed report with market insights and "
        "strategic positioning recommendations."
    ))
    args_schema: Type[BaseModel] = MacroAnalysisPositioningInput

    def _run(self) -> str:
        """Runs the tool."""
        try:
            logger.info("Starting macro analysis and strategic positioning")
            
            # Create and execute the crew
            crew = MarketMonitorCrew()
            results = crew.run_crew()
            
            logger.info("Macro analysis and strategic positioning completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Error executing macro analysis and positioning: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = MacroAnalysisPositioningTool()
    print(tool._run())
