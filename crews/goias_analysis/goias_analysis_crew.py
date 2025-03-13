#!/usr/bin/env python
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task, tool
from dotenv import load_dotenv

import sys
import os

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Import tools
from tools.financy.macro_economy.goias_data_tool import GoiasDataTool
from tools.financy.macro_economy.country_analysis_tool import CountryAnalysisTool
from tools.financy.macro_economy.macro_analysis_positioning_tool import MacroAnalysisPositioningTool

# Load environment variables
load_dotenv()

@CrewBase
class GoiasAnalysisCrew:
    """Crew para análise econômica do estado de Goiás usando CrewAI"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def regional_economist(self) -> Agent:
        return Agent(
            config=self.agents_config['regional_economist'],
            verbose=True,
            tools=[GoiasDataTool(), CountryAnalysisTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def sector_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['sector_specialist'],
            verbose=True,
            tools=[GoiasDataTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def public_finance_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['public_finance_analyst'],
            verbose=True,
            tools=[GoiasDataTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def strategy_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['strategy_coordinator'],
            verbose=True,
            tools=[MacroAnalysisPositioningTool(), CountryAnalysisTool(), GoiasDataTool()],
            llm="azure/gpt-4o-mini"
        )

    @task
    def regional_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['regional_analysis'],
            agent=self.regional_economist(),
            output_file='regional_analysis.txt'
        )

    @task
    def sector_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['sector_analysis'],
            agent=self.sector_specialist(),
            context=[self.regional_analysis()],
            output_file='sector_analysis.txt'
        )

    @task
    def public_finance_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['public_finance_analysis'],
            agent=self.public_finance_analyst(),
            context=[self.regional_analysis()],
            output_file='public_finance_analysis.txt'
        )

    @task
    def strategic_consolidation(self) -> Task:
        return Task(
            config=self.tasks_config['strategic_consolidation'],
            agent=self.strategy_coordinator(),
            context=[
                self.regional_analysis(),
                self.sector_analysis(),
                self.public_finance_analysis()
            ],
            output_file='strategic_consolidation.txt'
        )

    @crew
    def run_crew(self):
        """Executa a análise completa da economia goiana"""
        crew = Crew(
            name="GoiasAnalysisCrew",
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            process=Process.sequential,
            planning=True,
            output_log_file="crew_goias_analysis_log.txt"
        )
        return crew.kickoff()

if __name__ == "__main__":
    crew = GoiasAnalysisCrew()
    results = crew.run_crew()
