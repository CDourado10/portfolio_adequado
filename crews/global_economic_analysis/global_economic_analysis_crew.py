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

# Load environment variables
load_dotenv()

@CrewBase
class GlobalEconomicAnalysisCrew:
    """Crew for global economic analysis using CrewAI"""

    # Define paths relative to this file location
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def monetary_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['monetary_analyst'],
            verbose=True,
            tools=[EconomicCalendarTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def macro_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['macro_analyst'],
            verbose=True,
            tools=[EconomicCalendarTool()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def impact_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['impact_analyst'],
            verbose=True,
            tools=[EconomicCalendarTool()],
            llm="azure/gpt-4o-mini"
        )

    @task
    def monetary_policy_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['monetary_policy_analysis'],
            agent=self.monetary_analyst(),
            output_file='monetary_policy_analysis.txt'
        )

    @task
    def macroeconomic_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['macroeconomic_analysis'],
            agent=self.macro_analyst(),
            context=[self.monetary_policy_analysis()],
            output_file='macroeconomic_analysis.txt'
        )

    @task
    def market_impact_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['market_impact_analysis'],
            agent=self.impact_analyst(),
            context=[self.monetary_policy_analysis(), self.macroeconomic_analysis()],
            output_file='market_impact_analysis.txt'
        )

    @crew
    def run_crew(self) -> Crew:
        """Run the global economic analysis crew"""
        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=True,
            output_log_file="crew_global_economic_analysis_log.txt"
        )

        # Execute crew and return results
        results = crew.kickoff()
        return results

if __name__ == "__main__":
    crew = GlobalEconomicAnalysisCrew()
    results = crew.run_crew()
