from typing import List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging
import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from crews.global_economic_analysis.global_economic_analysis_crew import GlobalEconomicAnalysisCrew

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlobalEconomicAnalysisInput(BaseModel):
    """Input schema for GlobalEconomicAnalysisTool."""

class GlobalEconomicAnalysisTool(BaseTool):
    """Tool for global economic analysis."""
    
    name: str = Field(default="GlobalEconomicAnalysisTool")
    description: str = Field(default=(
        "Performs a detailed global economic analysis using a specialized crew. "
        "Analyzes high-importance economic events, including monetary policy decisions, "
        "macroeconomic indicators and their impacts on financial markets. "
        "Returns a comprehensive report with analyses and projections."
    ))
    args_schema: Type[BaseModel] = GlobalEconomicAnalysisInput

    def _run(self) -> str:
        """Run the tool."""
        try:
            logger.info("Starting global economic analysis")
            
            # Create and run the crew
            crew = GlobalEconomicAnalysisCrew()
            results = crew.run_crew()
            
            return results
            
        except Exception as e:
            error_msg = f"Error executing global economic analysis: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = GlobalEconomicAnalysisTool()
    print(tool._run())
