#!/usr/bin/env python
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task, tool
from crewai_tools import SerperDevTool, WebsiteSearchTool
from dotenv import load_dotenv

import sys
import os
import yaml

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now we can correctly import the tools
from tools.financy.analise_macro import MacroDataTool
from tools.financy.portfolio_optm import PortfolioOptimizerTool
from tools.financy.analise_ativo import AssetDataTool
from tools.financy.macro_economy.global_economic_analysis_tool import GlobalEconomicAnalysisTool

# Load environment variables
load_dotenv()

@CrewBase
class FinancialAdvisor:
    """Financial Advisor using CrewAI"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def macro_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['macro_analyst'],
            verbose=True,
            tools=[MacroDataTool(), GlobalEconomicAnalysisTool()],
            llm="gpt-4.1"
        )

    @agent
    def asset_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['asset_researcher'],
            verbose=True,
            tools=[SerperDevTool()],
            llm="gpt-4.1"
        )

    @agent
    def portfolio_optimizer(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_optimizer'],
            verbose=True,
            tools=[PortfolioOptimizerTool()],
            llm="gpt-4.1"
        )

    @agent
    def long_short_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['long_short_analyst'],
            verbose=True,
            tools=[AssetDataTool()],
            llm="gpt-4.1"
        )

    @agent
    def report_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['report_specialist'],
            verbose=True,
            llm="gpt-4.1"
        )

    @task
    def macro_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['macro_analysis'],
            agent=self.macro_analyst(),
            output_file='macro_analysis.md'
        )

    @task
    def asset_research(self) -> Task:
        return Task(
            config=self.tasks_config['asset_research'],
            agent=self.asset_researcher(),
            context=[self.macro_analysis()],
            output_file='asset_research.md'
        )

    @task
    def portfolio_optimization(self) -> Task:
        return Task(
            config=self.tasks_config['portfolio_optimization'],
            agent=self.portfolio_optimizer(),
            context=[self.asset_research()],
            output_file='portfolio_optimization.md'
        )

    @task
    def long_short_strategy(self) -> Task:
        return Task(
            config=self.tasks_config['long_short_strategy'],
            agent=self.long_short_analyst(),
            context=[self.portfolio_optimization()],
            output_file='long_short_strategy.md'
        )

    @task
    def final_report(self) -> Task:
        return Task(
            config=self.tasks_config['final_report'],
            agent=self.report_specialist(),
            context=[self.long_short_strategy(), self.asset_research(), self.portfolio_optimization(), self.macro_analysis()],
            output_file='final_report.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            name="Financial Advisor",
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            process=Process.sequential,
            memory=True,
            planning=True,
            output_log_file="crew_log.md"
        )

def main():
    advisor = FinancialAdvisor()
    result = advisor.crew().kickoff()
    print("\nFinal Report:")
    print(result)

if __name__ == "__main__":
    main()