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
from tools.financy.macro_economy.macro_monitor_tool import MacroMonitorTool
from tools.financy.macro_economy.bond_monitor_tool import BondMonitorTool
from tools.financy.macro_economy.currency_monitor_tool import CurrencyMonitorTool
from tools.financy.macro_economy.commodity_monitor_tool import CommodityMonitorTool
from tools.financy.macro_economy.stock_market_monitor_tool import StockMarketMonitorTool
from tools.financy.macro_economy.crypto_monitor_tool import CryptoMonitorTool


# Load environment variables
load_dotenv()

@CrewBase
class MarketMonitorCrew:
    """Crew for market events analysis using CrewAI"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def monitor_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['monitor_coordinator'],
            verbose=True,
            tools=[MacroMonitorTool(), BondMonitorTool(), CurrencyMonitorTool(), CommodityMonitorTool(), StockMarketMonitorTool(), CryptoMonitorTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def market_integrator(self) -> Agent:
        return Agent(
            config=self.agents_config['market_integrator'],
            verbose=True,
            llm="azure/gpt-4o-mini"
        )

    @task
    def macro_monitoring(self) -> Task:
        return Task(
            config=self.tasks_config['macro_monitoring'],
            agent=self.monitor_coordinator(),
            output_file='macro_monitoring.txt'
        )

    @task
    def bond_monitoring(self) -> Task:
        return Task(
            config=self.tasks_config['bond_monitoring'],
            agent=self.monitor_coordinator(),
            output_file='bond_monitoring.txt'
        )

    @task
    def currency_monitoring(self) -> Task:
        return Task(
            config=self.tasks_config['currency_monitoring'],
            agent=self.monitor_coordinator(),
            output_file='currency_monitoring.txt'
        )

    @task
    def commodity_monitoring(self) -> Task:
        return Task(
            config=self.tasks_config['commodity_monitoring'],
            agent=self.monitor_coordinator(),
            output_file='commodity_monitoring.txt'
        )

    @task
    def stock_monitoring(self) -> Task:
        return Task(
            config=self.tasks_config['stock_monitoring'],
            agent=self.monitor_coordinator(),
            output_file='stock_monitoring.txt'
        )

    @task
    def crypto_monitoring(self) -> Task:
        return Task(
            config=self.tasks_config['crypto_monitoring'],
            agent=self.monitor_coordinator(),
            output_file='crypto_monitoring.txt'
        )
    
    @task
    def data_consolidation(self) -> Task:
        return Task(
            config=self.tasks_config['data_consolidation'],
            agent=self.market_integrator(),
            context=[self.macro_monitoring(), self.bond_monitoring(), self.currency_monitoring(), self.commodity_monitoring(), self.stock_monitoring(), self.crypto_monitoring()],
            output_file='data_consolidation.txt'
        )

    @task
    def integrated_market_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['integrated_market_analysis'],
            agent=self.market_integrator(),
            context=[self.data_consolidation()],
            output_file='integrated_market_analysis.txt'
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
    crew = MarketMonitorCrew()
    results = crew.run_crew()