#!/usr/bin/env python
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os
import sys

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Import financial tools
from tools.financy.yahoo_finance.screener_tool import ScreenerTool
from tools.financy.yahoo_finance.valid_screeners_tool import ValidScreenersTool
from tools.financy.yahoo_finance.trending_stocks_tool import TrendingStocksTool
from tools.financy.yahoo_finance.ticker_tool import TickerTool
from tools.financy.portfolio_reduction_tool import PortfolioReductionTool
from tools.financy.yahoo_finance.exchanges_tool import ExchangesTool

# Load environment variables
load_dotenv()

@CrewBase
class AssetExplorationCrew:
    """Crew for discovering and analyzing potential investment assets"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


    def crew_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['crew_manager'],
            verbose=True,
            llm="gpt-4.1"
        )

    @agent
    def screener_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['screener_researcher'],
            verbose=True,
            tools=[ScreenerTool(), ValidScreenersTool()],
            llm="gpt-4.1"
        )

    @agent
    def trending_stocks_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['trending_stocks_researcher'],
            verbose=True,
            tools=[TrendingStocksTool()],
            llm="gpt-4.1"
        )

    @agent
    def asset_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['asset_analyst'],
            verbose=True,
            tools=[TickerTool(), ExchangesTool()],
            llm="gpt-4.1"
        )

    @agent
    def portfolio_reduction_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_reduction_specialist'],
            verbose=True,
            tools=[PortfolioReductionTool(), ScreenerTool(), ValidScreenersTool()],
            llm="gpt-4.1",
            max_iter=250
        )

    @task
    def screener_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['screener_research_task'],
            agent=self.screener_researcher(),
            output_file="screener_results.txt"
        )

    @task
    def trending_stocks_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['trending_stocks_research_task'],
            agent=self.trending_stocks_researcher(),
            output_file="trending_stocks_results.txt"
        )

    @task
    def asset_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['asset_analysis_task'],
            agent=self.asset_analyst(),
            context=[self.screener_research_task(), self.trending_stocks_research_task()],
            output_file="asset_analysis_results.txt"
        )

    @task
    def portfolio_reduction_task(self) -> Task:
        return Task(
            config=self.tasks_config['portfolio_reduction_task'],
            agent=self.portfolio_reduction_specialist(),
            context=[self.asset_analysis_task()],
            output_file="reduced_portfolio.txt"
        )

    @crew
    def crew(self) -> Crew:
        """Run the asset exploration crew"""
        return Crew(
            name="Asset Exploration Crew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            manager_agent=self.crew_manager(),
            planning=True,
            output_log_file="asset_exploration_log.txt"
        )


if __name__ == "__main__":
    crew = AssetExplorationCrew()
    results = crew.crew().kickoff()
    print(results)
