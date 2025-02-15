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
from tools.financy.macro_economy.topical_news_tool import TopicalNewsAnalyzer
from tools.financy.macro_economy.multi_source_market_news_tool import MultiSourceMarketNews

# Load environment variables
load_dotenv()

@CrewBase
class MarketNewsAnalysisCrew:
    """Crew for market news analysis using CrewAI"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def news_preprocessor(self) -> Agent:
        return Agent(
            config=self.agents_config['news_preprocessor'],
            verbose=True,
            tools=[MultiSourceMarketNews(), TopicalNewsAnalyzer()],
            llm="azure/gpt-4o-mini"
        )

    @agent
    def market_news_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['market_news_analyst'],
            verbose=True,
            llm="azure/gpt-4o-mini"
        )

    @task
    def data_cleaning(self) -> Task:
        return Task(
            config=self.tasks_config['data_cleaning'],
            agent=self.news_preprocessor(),
            output_file='news_collection.txt'
        )

    @task
    def initial_categorization(self) -> Task:
        return Task(
            config=self.tasks_config['initial_categorization'],
            agent=self.news_preprocessor(),
            context=[self.data_cleaning()],
            output_file='initial_categorization.txt'
        )

    @task
    def trend_identification(self) -> Task:
        return Task(
            config=self.tasks_config['trend_identification'],
            agent=self.market_news_analyst(),
            context=[self.initial_categorization()],
            output_file='trend_identification.txt'
        )

    @task
    def impact_assessment(self) -> Task:
        return Task(
            config=self.tasks_config['impact_assessment'],
            agent=self.market_news_analyst(),
            context=[self.initial_categorization()],
            output_file='impact_assessment.txt'
        )

    @task
    def insight_generation(self) -> Task:
        return Task(
            config=self.tasks_config['insight_generation'],
            agent=self.market_news_analyst(),
            context=[self.trend_identification(), self.impact_assessment()],
            output_file='insight_generation.txt'
        )

    @crew
    def run_crew(self) -> Crew:
        """Run the market news analysis crew"""
        crew = Crew(
            name="Market News Analysis Crew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=True,
            output_log_file="crew_market_news_analysis_log.txt"
        )

        # Execute crew and return results
        results = crew.kickoff()
        return results

if __name__ == "__main__":
    crew = MarketNewsAnalysisCrew()
    results = crew.run_crew()
