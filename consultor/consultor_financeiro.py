#!/usr/bin/env python
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task, tool
from crewai_tools import SerperDevTool, WebsiteSearchTool
from dotenv import load_dotenv

import sys
import os
import yaml

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Agora podemos importar corretamente as ferramentas
from tools.financeira.analise_macro import MacroDataTool
from tools.financeira.portfolio_optm import PortfolioOptimizerTool
from tools.financeira.analise_ativo import AtivoDataTool

# Carregar variáveis de ambiente
load_dotenv()

@CrewBase
class ConsultorFinanceiro:
	"""Consultor Financeiro usando CrewAI"""

	agents_config_path = "config/agents.yaml"
	tasks_config_path = "config/tasks.yaml"

	@agent
	def analista_macro(self) -> Agent:
		return Agent(
			config=self.agents_config['analista_macro'],
			verbose=True,
			tools=[MacroDataTool()],
			llm="azure/gpt-4o-mini"
		)

	@agent
	def pesquisador_ativos(self) -> Agent:
		return Agent(
			config=self.agents_config['pesquisador_ativos'],
			verbose=True,
			tools=[SerperDevTool(), WebsiteSearchTool(website="https://tradingeconomics.com/stream")],
			llm="azure/gpt-4o-mini"
		)

	@agent
	def otimizador_portfolio(self) -> Agent:
		return Agent(
			config=self.agents_config['otimizador_portfolio'],
			verbose=True,
			tools=[PortfolioOptimizerTool()],
			llm="azure/gpt-4o-mini"
		)

	@agent
	def analista_long_short(self) -> Agent:
		return Agent(
			config=self.agents_config['analista_long_short'],
			verbose=True,
			tools=[AtivoDataTool()],
			llm="azure/gpt-4o-mini"
		)

	@agent
	def especialista_relatorio(self) -> Agent:
		return Agent(
			config=self.agents_config['especialista_relatorio'],
			verbose=True,
			llm="azure/gpt-4o-mini"
		)

	@task
	def analise_macro(self) -> Task:
		return Task(
			config=self.tasks_config['analise_macro'],
			agent=self.analista_macro()
		)

	@task
	def pesquisa_ativos(self) -> Task:
		return Task(
			config=self.tasks_config['pesquisa_ativos'],
			agent=self.pesquisador_ativos(),
			context=[self.analise_macro()]
		)

	@task
	def otimizacao_portfolio(self) -> Task:
		return Task(
			config=self.tasks_config['otimizacao_portfolio'],
			agent=self.otimizador_portfolio(),
			context=[self.pesquisa_ativos()]
		)

	@task
	def estrategia_long_short(self) -> Task:
		return Task(
			config=self.tasks_config['estrategia_long_short'],
			agent=self.analista_long_short(),
			context=[self.otimizacao_portfolio()]
		)

	@task
	def relatorio_final(self) -> Task:
		return Task(
			config=self.tasks_config['relatorio_final'],
			agent=self.especialista_relatorio(),
			context=[self.estrategia_long_short(), self.pesquisa_ativos(), self.otimizacao_portfolio(), self.analise_macro()]
		)

	@crew
	def crew(self) -> Crew:
		return Crew(
			name="Consultor Financeiro",
			agents=self.agents,
			tasks=self.tasks,
			verbose=True,
			process=Process.sequential,
			memory=True,
			planning=True,
			output_log_file="crew_log.md"
		)

def main():
	consultor = ConsultorFinanceiro()
	result = consultor.crew().kickoff()
	print("\nRelatório Final:")
	print(result)

if __name__ == "__main__":
	main()