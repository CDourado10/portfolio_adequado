#!/usr/bin/env python
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os
import sys

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Import asset exploration and portfolio optimization tools
from tools.financy.asset_exploration_tool import AssetExplorationTool
from tools.financy.portfolio_optimization_tool import PortfolioOptimizationTool
from tools.financy.macro_economy.macro_analysis_positioning_tool import MacroAnalysisPositioningTool

# Load environment variables
load_dotenv()

@CrewBase
class MasterPortfolioCrew:
    """Master crew that coordinates asset exploration and portfolio optimization"""

    @agent
    def portfolio_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_manager'],
            verbose=True,
            tools=[MacroAnalysisPositioningTool(), AssetExplorationTool(), PortfolioOptimizationTool()],
            llm="azure/gpt-4o-mini"
        )

    @task
    def asset_exploration_task(self) -> Task:
        return Task(
            config=self.tasks_config['asset_exploration_task'],
            agent=self.portfolio_manager(),
            tools=[AssetExplorationTool()],
            output_file="asset_exploration_results.txt"
        )

    @task
    def portfolio_optimization_task(self) -> Task:
        return Task(
            config=self.tasks_config['portfolio_optimization_task'],
            agent=self.portfolio_manager(),
            tools=[PortfolioOptimizationTool()],
            context=[self.asset_exploration_task()],
            output_file="portfolio_optimization_results.txt"
        )

    @task
    def final_portfolio_report_task(self) -> Task:
        return Task(
            config=self.tasks_config['final_portfolio_report_task'],
            agent=self.portfolio_manager(),
            context=[self.portfolio_optimization_task()],
            output_file="final_portfolio_report.txt"
        )

    @crew
    def crew(self) -> Crew:
        """Run the master portfolio management crew"""
        crew = Crew(
            name="Master Portfolio Management Crew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=True,
            output_log_file="master_portfolio_log.txt"
        )
        return crew

if __name__ == "__main__":
    crew = MasterPortfolioCrew()
    results = crew.crew().kickoff(inputs={
        "risk_level": "high",
        "investment_horizon": "short_term",
        "asset_types": "cryptocurrencies"
    })

    
    print(results)
