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
from tools.financy.macro_economy.macro_analysis_positioning_tool import MacroAnalysisPositioningTool
from tools.financy.portfolio_optm import PortfolioOptimizerTool
from tools.financy.analise_ativo import AssetDataTool
from tools.financy.macro_economy.global_economic_analysis_tool import GlobalEconomicAnalysisTool

# Load environment variables
load_dotenv()

@CrewBase
class PortfolioOptimizationAdvisor:
    """Portfolio Optimization Advisor using CrewAI
    
    This crew is responsible for creating 9 optimized investment portfolios based on
    different combinations of risk profiles (low, medium, high) and time horizons
    (short, medium, long term). It uses a team of specialized agents to:
    1. Analyze macroeconomic trends
    2. Research suitable assets
    3. Optimize portfolio allocations
    4. Generate comprehensive reports
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    used_assets = set()  # Track used assets to prevent repetition

    @agent
    def macro_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['macro_analyst'],
            verbose=True,
            tools=[MacroAnalysisPositioningTool(), GlobalEconomicAnalysisTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def asset_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['asset_researcher'],
            verbose=True,
            tools=[AssetDataTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def portfolio_optimizer(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_optimizer'],
            verbose=True,
            tools=[PortfolioOptimizerTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def report_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['report_specialist'],
            verbose=True,
            llm="azure/gpt-4o-mini"
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
    def final_report(self) -> Task:
        return Task(
            config=self.tasks_config['final_report'],
            agent=self.report_specialist(),
            context=[self.macro_analysis(), self.asset_research(), self.portfolio_optimization()],
            output_file='final_report.md'
        )

    @crew
    def run(self) -> Crew:
        """Run the Portfolio Optimization Advisor crew"""
        crew = Crew(
            name="Portfolio Optimization Advisor",
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            process=Process.sequential,
            planning=True,
            output_log_file="crew_portfolio_optimization_log.md"
        )

        # Execute crew and return results
        results = crew.kickoff()
        return results

if __name__ == "__main__":
    advisor = PortfolioOptimizationAdvisor()
    results = advisor.run()
    print("\nFinal Report:")
    print(results)
