#!/usr/bin/env python
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task, tool
from dotenv import load_dotenv

import sys
import os

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Import economic calendar tool
from tools.financy.macro_economy.country_data_tool import CountryDataTool
from tools.financy.macro_economy.macro_analysis_positioning_tool import MacroAnalysisPositioningTool
from tools.financy.analise_bovespa import BovespaDataTool


# Load environment variables
load_dotenv()

@CrewBase
class CountryAnalysisCrew:
    """Crew for market events analysis using CrewAI"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def macro_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['macro_analyst'],
            verbose=True,
            tools=[MacroAnalysisPositioningTool(), CountryDataTool()],
            llm="gpt-4.1"
        )

    @agent
    def sector_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sector_analyst'],
            verbose=True,
            tools=[CountryDataTool(), BovespaDataTool()],
            llm="gpt-4.1"
        )

    @agent
    def trade_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['trade_specialist'],
            verbose=True,
            tools=[BovespaDataTool()],
            llm="gpt-4.1"
        )

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_analyst'],
            verbose=True,
            llm="gpt-4.1"
        )

    @agent
    def strategic_coordinator   (self) -> Agent:
        return Agent(
            config=self.agents_config['strategic_coordinator'],
            verbose=True,
            llm="gpt-4.1"
        )

    @task
    def macro_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['macro_analysis'],
            agent=self.macro_analyst(),
            output_file='macro_analysis.txt'
        )

    @task
    def sector_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['sector_analysis'],
            agent=self.sector_analyst(),
            context=[self.macro_analysis()],
            output_file='sector_analysis.txt'
        )

    @task
    def trade_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['trade_analysis'],
            agent=self.trade_specialist(),
            context=[self.sector_analysis()],
            output_file='trade_analysis.txt'
        )

    @task
    def financial_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['financial_analysis'],
            agent=self.financial_analyst(),
            context=[self.macro_analysis(), self.sector_analysis()],
            output_file='financial_analysis.txt'
        )

    @task
    def structural_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['structural_analysis'],
            agent=self.strategic_coordinator(),
            context=[self.financial_analysis(), self.trade_analysis()],
            output_file='structural_analysis.txt'
        )

    @task
    def strategic_consolidation(self) -> Task:
        return Task(
            config=self.tasks_config['strategic_consolidation'],
            agent=self.strategic_coordinator(),
            context=[self.structural_analysis()],
            output_file='strategic_consolidation.txt'
        )

    @crew
    def run_crew(self) -> Crew:
        """Run the market monitor crew"""
        crew = Crew(
            name="Market Monitor Crew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=True,
            output_log_file="crew_market_monitor_log.txt"
        )

        # Execute crew and return results
        results = crew.kickoff()
        return results

if __name__ == "__main__":
    crew = CountryAnalysisCrew()
    results = crew.run_crew()