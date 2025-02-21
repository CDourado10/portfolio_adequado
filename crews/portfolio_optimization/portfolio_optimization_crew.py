#!/usr/bin/env python
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os
import sys

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Import financial tools for different risk levels
from tools.financy.low_risk_short_term_tool import LowRiskShortTermTool
from tools.financy.low_risk_medium_term_tool import LowRiskMediumTermTool
from tools.financy.low_risk_long_term_tool import LowRiskLongTermTool
from tools.financy.medium_risk_short_term_tool import MediumRiskShortTermTool
from tools.financy.medium_risk_medium_term_tool import MediumRiskMediumTermTool
from tools.financy.medium_risk_long_term_tool import MediumRiskLongTermTool
from tools.financy.high_risk_short_term_tool import HighRiskShortTermTool
from tools.financy.high_risk_medium_term_tool import HighRiskMediumTermTool
from tools.financy.high_risk_long_term_tool import HighRiskLongTermTool
#from tools.financy.performance_analysis_tool import PerformanceAnalysisTool  # Nova ferramenta adicionada

# Load environment variables
load_dotenv()

@CrewBase
class PortfolioOptimizationCrew:
    """Crew for optimizing investment portfolios based on different risk levels"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def portfolio_crew_manager(self) -> Agent:
        """Gerente da Crew de otimização, supervisiona todo o processo."""
        return Agent(
            config=self.agents_config['portfolio_crew_manager'],
            verbose=True,
            llm="azure/gpt-4o-mini"
        )

    @agent
    def portfolio_optimization_strategist(self) -> Agent:
        """Escolhe a estratégia de otimização de acordo com o risco e prazo."""
        return Agent(
            config=self.agents_config['portfolio_optimization_strategist'],
            verbose=True,
            llm="azure/gpt-4o-mini"
        )

    @agent
    def portfolio_constructor(self) -> Agent:
        """Constrói o portfólio aplicando a estratégia definida."""
        return Agent(
            config=self.agents_config['portfolio_constructor'],
            verbose=True,
            tools=[
                LowRiskShortTermTool(), LowRiskMediumTermTool(), LowRiskLongTermTool(),
                MediumRiskShortTermTool(), MediumRiskMediumTermTool(), MediumRiskLongTermTool(),
                HighRiskShortTermTool(), HighRiskMediumTermTool(), HighRiskLongTermTool()
            ],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def risk_performance_analyst(self) -> Agent:
        """Analisa métricas de risco e desempenho do portfólio otimizado."""
        return Agent(
            config=self.agents_config['risk_performance_analyst'],
            verbose=True,
            allow_code_execution=True,
            llm="azure/gpt-4o-mini"
        )

    @agent
    def portfolio_report_manager(self) -> Agent:
        """Compila os resultados e gera o relatório final."""
        return Agent(
            config=self.agents_config['portfolio_report_manager'],
            verbose=True,
            llm="azure/gpt-4o-mini"
        )


    @task
    def define_optimization_strategy_task(self) -> Task:
        """Escolhe a melhor estratégia de otimização com base nos inputs do investidor."""
        return Task(
            config=self.tasks_config['define_optimization_strategy_task'],
            agent=self.portfolio_optimization_strategist(),
            output_file="optimization_strategy.txt"
        )

    @task
    def portfolio_construction_task(self) -> Task:
        """Constrói o portfólio aplicando a estratégia escolhida."""
        return Task(
            config=self.tasks_config['portfolio_construction_task'],
            agent=self.portfolio_constructor(),
            context=[self.define_optimization_strategy_task()],
            output_file="optimized_portfolio.txt"
        )

    @task
    def risk_return_evaluation_task(self) -> Task:
        """Avalia métricas de risco e retorno do portfólio otimizado."""
        return Task(
            config=self.tasks_config['risk_return_evaluation_task'],
            agent=self.risk_performance_analyst(),
            context=[self.portfolio_construction_task()],
            output_file="risk_performance_analysis.txt"
        )

    @task
    def final_portfolio_report_task(self) -> Task:
        """Compila o relatório final do portfólio otimizado."""
        return Task(
            config=self.tasks_config['final_portfolio_report_task'],
            agent=self.portfolio_report_manager(),
            context=[
                self.define_optimization_strategy_task(),
                self.portfolio_construction_task(),
                self.risk_return_evaluation_task()
            ],
            output_file="final_portfolio_report.txt"
        )

    @crew
    def crew(self) -> Crew:
        """Run the portfolio optimization crew"""
        return Crew(
            name="Portfolio Optimization Crew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            manager_agent=self.portfolio_crew_manager(),
            verbose=True,
            planning=True,
            output_log_file="portfolio_optimization_log.txt"
        )

if __name__ == "__main__":
    crew = PortfolioOptimizationCrew()
    results = crew.crew().kickoff()
    print(results)
