from typing import List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging
import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from crews.macro_analysis_and_positioning_strategy.macro_analysis_and_positioning_strategy_crew_alt import MarketMonitorCrew

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
        "Realiza análise abrangente do mercado e desenvolve estratégias de posicionamento. "
        "Utiliza uma equipe especializada para: \n"
        "1. Análise de impacto entre setores\n"
        "2. Monitoramento de eventos de mercado\n"
        "3. Análise de notícias relevantes\n"
        "4. Geração de relatórios estratégicos\n"
        "Retorna um relatório detalhado com insights sobre o mercado e recomendações "
        "estratégicas de posicionamento."
    ))
    args_schema: Type[BaseModel] = MacroAnalysisPositioningInput

    def _run(self) -> str:
        """Executa a ferramenta."""
        try:
            logger.info("Iniciando análise macro e posicionamento estratégico")
            
            # Cria e executa a crew
            crew = MarketMonitorCrew()
            results = crew.run_crew()
            
            logger.info("Análise macro e posicionamento estratégico concluídos com sucesso")
            return results
            
        except Exception as e:
            error_msg = f"Erro ao executar análise macro e posicionamento: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = MacroAnalysisPositioningTool()
    print(tool._run())
