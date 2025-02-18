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
from tools.financy.macro_economy.market_news_analysis_tool import MarketNewsAnalysisTool
from tools.financy.macro_economy.market_monitor_tool import MarketMonitorTool
from tools.financy.macro_economy.market_events_analysis_tool import MarketEventsAnalysisTool


# Load environment variables
load_dotenv()

@CrewBase
class MarketMonitorCrew:
    """Crew for market events analysis using CrewAI"""

    # Define paths relative to this file location
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def macro_events_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['macro_events_specialist'],
            verbose=True,
            tools=[MarketEventsAnalysisTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def market_dynamics_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['market_dynamics_analyst'],
            verbose=True,
            tools=[MarketMonitorTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def sentiment_trends_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_trends_analyst'],
            verbose=True,
            tools=[MarketNewsAnalysisTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def cross_sector_impact_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['cross_sector_impact_analyst'],
            verbose=True,
            llm="azure/gpt-4o-mini"
        )

    @agent
    def global_intelligence_director(self) -> Agent:
        return Agent(
            config=self.agents_config['global_intelligence_director'],
            verbose=True,
            llm="azure/gpt-4o-mini"
        )

    @task
    def analyze_economic_events(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_economic_events'],
            agent=self.macro_events_specialist(),
            output_file='analyze_economic_events.txt'
        )

    @task
    def monitor_market_patterns(self) -> Task:
        return Task(
            config=self.tasks_config['monitor_market_patterns'],
            agent=self.market_dynamics_analyst(),
            output_file='monitor_market_patterns.txt'
        )

    @task
    def analyze_market_sentiment(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_market_sentiment'],
            agent=self.sentiment_trends_analyst(),
            output_file='analyze_market_sentiment.txt'
        )

    @task
    def perform_cross_sector_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['perform_cross_sector_analysis'],
            agent=self.cross_sector_impact_analyst(),
            context=[self.analyze_economic_events(), self.monitor_market_patterns(), self.analyze_market_sentiment()],
            output_file='perform_cross_sector_analysis.txt'
        )

    @task
    def strategic_analysis_and_reporting(self) -> Task:
        return Task(
            config=self.tasks_config['strategic_analysis_and_reporting'],
            agent=self.global_intelligence_director(),
            context=[self.analyze_economic_events(), self.monitor_market_patterns(), self.analyze_market_sentiment(), self.perform_cross_sector_analysis()],
            output_file='strategic_analysis_and_reporting.txt'
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