#!/usr/bin/env python
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task, tool
from dotenv import load_dotenv

import sys
import os

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Import required tools
from tools.financy.macro_economy.macro_analysis_positioning_tool import MacroAnalysisPositioningTool
from tools.financy.yahoo_finance.screener_tool import ScreenerTool
from tools.financy.yahoo_finance.valid_screeners_tool import ValidScreenersTool
from tools.financy.yahoo_finance.trending_stocks_tool import TrendingStocksTool
from tools.financy.yahoo_finance.ticker_tool import TickerTool
from tools.financy.hierarchical_portfolio_tool import HierarchicalPortfolioTool
from tools.financy.yahoo_finance.exchanges_tool import ExchangesTool

# Load environment variables
load_dotenv()

@CrewBase
class PortfolioOptimizationCrew:
    """Crew for portfolio optimization and diversification using CrewAI"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # 🔹 Define Agents
    @agent
    def macro_analysis_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['macro_analysis_expert'],
            verbose=True,
            tools=[MacroAnalysisPositioningTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def screener_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['screener_researcher'],
            verbose=True,
            tools=[ScreenerTool(), ValidScreenersTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def trending_stocks_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['trending_stocks_researcher'],
            verbose=True,
            tools=[TrendingStocksTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def asset_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['asset_analyst'],
            verbose=True,
            tools=[TickerTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def asset_selection_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['asset_selection_manager'],
            verbose=True,
            llm="azure/gpt-4o-mini"
        )

    @agent
    def portfolio_optimizer(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_optimizer'],
            verbose=True,
            tools=[HierarchicalPortfolioTool(), ExchangesTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def portfolio_validator(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_validator'],
            verbose=True,
            tools=[HierarchicalPortfolioTool(), ExchangesTool()],
            llm="azure/gpt-4o-mini"
        )

    # 🔹 Define Tasks
    @task
    def macro_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['macro_analysis_task'],
            agent=self.macro_analysis_expert(),
            output_file='macro_analysis_report.txt'
        )

    @task
    def screener_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['screener_research_task'],
            agent=self.screener_researcher(),
            context=[self.macro_analysis_task()],
            output_file='screener_research_report.txt'
        )

    @task
    def trending_stocks_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['trending_stocks_research_task'],
            agent=self.trending_stocks_researcher(),
            context=[self.macro_analysis_task()],
            output_file='trending_stocks_report.txt'
        )

    @task
    def asset_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['asset_analysis_task'],
            agent=self.asset_analyst(),
            context=[self.screener_research_task(), self.trending_stocks_research_task()],
            output_file='asset_analysis_report.txt'
        )

    @task
    def asset_selection_task(self) -> Task:
        return Task(
            config=self.tasks_config['asset_selection_task'],
            agent=self.asset_selection_manager(),
            context=[self.asset_analysis_task()],
            output_file='asset_selection_report.txt'
        )

    @task
    def initial_portfolio_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config['initial_portfolio_creation_task'],
            agent=self.portfolio_optimizer(),
            context=[self.asset_selection_task()],
            output_file='initial_portfolios.txt'
        )

    @task
    def portfolio_optimization_task(self) -> Task:
        return Task(
            config=self.tasks_config['portfolio_optimization_task'],
            agent=self.portfolio_optimizer(),
            context=[self.initial_portfolio_creation_task()],
            output_file='optimized_portfolios.txt'
        )

    @task
    def portfolio_validation_task(self) -> Task:
        return Task(
            config=self.tasks_config['portfolio_validation_task'],
            agent=self.portfolio_validator(),
            context=[self.portfolio_optimization_task()],
            output_file='portfolio_validation_report.txt'
        )

    @task
    def portfolio_adjustment_task(self) -> Task:
        return Task(
            config=self.tasks_config['portfolio_adjustment_task'],
            agent=self.portfolio_optimizer(),
            context=[self.portfolio_validation_task()],
            output_file='final_portfolios.txt'
        )

    # 🔹 Run the Crew
    @crew
    def run_crew(self) -> Crew:
        """Run the portfolio optimization and diversification crew"""
        crew = Crew(
            name="Portfolio Optimization and Diversification Crew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,  # Using sequential, but will iterate manually if needed
            verbose=True,
            planning=True,
            output_log_file="portfolio_optimization_log.txt"
        )

        # Execute the Crew and return results
        results = crew.kickoff()
        return results

if __name__ == "__main__":
    crew = PortfolioOptimizationCrew()

    # Run the Crew
    results = crew.run_crew()
    print(results)
