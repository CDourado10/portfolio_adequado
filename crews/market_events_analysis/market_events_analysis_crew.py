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
from tools.financy.macro_economy.economic_calendar_tool import EconomicCalendarTool
from tools.financy.macro_economy.earnings_calendar_tool import EarningsCalendarTool

# Load environment variables
load_dotenv()

@CrewBase
class MarketEventsAnalysisCrew:
    """Crew for market events analysis using CrewAI"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def calendar_data_processor(self) -> Agent:
        return Agent(
            config=self.agents_config['calendar_data_processor'],
            verbose=True,
            tools=[EconomicCalendarTool(), EarningsCalendarTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def economic_events_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['economic_events_analyst'],
            verbose=True,
            llm="azure/gpt-4o-mini"
        )

    @agent
    def corporate_events_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['corporate_events_analyst'],
            verbose=True,
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
    def collect_economic_calendar(self) -> Task:
        return Task(
            config=self.tasks_config['collect_economic_calendar'],
            agent=self.calendar_data_processor(),
            output_file='economic_calendar_collection.txt'
        )

    @task
    def collect_earnings_calendar(self) -> Task:
        return Task(
            config=self.tasks_config['collect_earnings_calendar'],
            agent=self.calendar_data_processor(),
            output_file='earnings_calendar_collection.txt'
        )

    @task
    def macro_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['macro_analysis'],
            agent=self.economic_events_analyst(),
            context=[self.collect_economic_calendar()],
            output_file='macro_analysis.txt'
        )

    @task
    def macro_trends(self) -> Task:
        return Task(
            config=self.tasks_config['macro_trends'],
            agent=self.economic_events_analyst(),
            context=[self.macro_analysis()],
            output_file='macro_trends.txt'
        )

    @task
    def earnings_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['earnings_analysis'],
            agent=self.corporate_events_analyst(),
            context=[self.collect_earnings_calendar()],
            output_file='earnings_analysis.txt'
        )
    
    @task
    def sector_impact(self) -> Task:
        return Task(
            config=self.tasks_config['sector_impact'],
            agent=self.corporate_events_analyst(),
            context=[self.earnings_analysis(), self.macro_analysis()],
            output_file='sector_impact.txt'
        )

    @task
    def analysis_integration(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_integration'],
            agent=self.market_integrator(),
            context=[self.macro_trends(), self.sector_impact(), self.collect_economic_calendar(), self.collect_earnings_calendar()],
            output_file='analysis_integration.txt'
        )

    @crew
    def run_crew(self) -> Crew:
        """Run the market events analysis crew"""
        crew = Crew(
            name="Market Events Analysis Crew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=True,
            output_log_file="crew_market_events_analysis_log.txt"
        )

        # Execute crew and return results
        results = crew.kickoff()
        return results

if __name__ == "__main__":
    crew = MarketEventsAnalysisCrew()
    results = crew.run_crew()