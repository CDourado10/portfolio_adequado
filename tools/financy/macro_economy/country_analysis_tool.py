from typing import List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging
import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from crews.country_analysis.country_analysis_crew import CountryAnalysisCrew

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CountryAnalysisInput(BaseModel):
    """Schema de entrada para CountryAnalysisTool."""
    pass

class CountryAnalysisTool(BaseTool):
    """Ferramenta para análise abrangente de países e seus mercados."""
    
    name: str = Field(default="CountryAnalysisTool")
    description: str = Field(default=(
        "Realiza uma análise completa de países utilizando uma equipe especializada que executa: \n"
        "1. Análise Macroeconômica: avaliação de indicadores macro e posicionamento\n"
        "2. Análise Setorial: estudo detalhado dos principais setores econômicos\n"
        "3. Análise de Comércio: avaliação de fluxos comerciais e oportunidades\n"
        "4. Análise Financeira: estudo dos mercados financeiros e indicadores\n"
        "5. Análise Estrutural: avaliação da estrutura econômica e institucional\n"
        "6. Consolidação Estratégica: integração das análises e recomendações\n\n"
        "Retorna um relatório consolidado com insights estratégicos e recomendações "
        "baseadas em múltiplas perspectivas de análise."
    ))
    args_schema: Type[BaseModel] = CountryAnalysisInput

    def _run(self) -> str:
        """Executa a ferramenta."""
        try:
            logger.info("Iniciando análise abrangente do país")
            
            # Cria e executa a crew
            crew = CountryAnalysisCrew()
            results = crew.run_crew()
            
            logger.info("Análise do país concluída com sucesso")
            return results
            
        except Exception as e:
            error_msg = f"Erro ao executar análise do país: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = CountryAnalysisTool()
    print(tool._run())
