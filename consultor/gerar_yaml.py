import yaml
import os

# Definir caminhos absolutos
base_dir = r'c:\Users\belac_odysseus\Documents\Projetos\consultor_financeiro\novo_consultor'
agents_path = os.path.join(base_dir, 'config', 'agents.yaml')
tasks_path = os.path.join(base_dir, 'config', 'tasks.yaml')

agents = {
	'analista_mercado': {
		'role': 'Analista de Mercado Financeiro',
		'goal': 'Analisar tendências e dados do mercado financeiro',
		'backstory': 'Analista sênior com 15 anos de experiência em análise de mercado e investimentos',
		'tools': ['SerperDevTool', 'WebsiteSearchTool'],
		'llm': 'openai/gpt-4o-mini'
	},
	'consultor_investimentos': {
		'role': 'Consultor de Investimentos',
		'goal': 'Criar estratégias personalizadas de investimento',
		'backstory': 'Consultor certificado CFP com vasta experiência em planejamento financeiro pessoal',
		'tools': ['SerperDevTool'],
		'llm': 'openai/gpt-4o-mini'
	},
	'analista_risco': {
		'role': 'Analista de Risco',
		'goal': 'Avaliar riscos e propor estratégias de mitigação',
		'backstory': 'Especialista em análise de risco com foco em proteção patrimonial',
		'tools': ['SerperDevTool'],
		'llm': 'openai/gpt-4o-mini'
	}
}

tasks = {
	'analise_mercado': {
		'description': '''
Analise o cenário atual do mercado financeiro brasileiro, considerando:
1. Principais índices (Ibovespa, CDI, etc)
2. Cenário macroeconômico
3. Tendências setoriais
4. Riscos e oportunidades
''',
		'expected_output': 'Relatório detalhado sobre o cenário atual do mercado financeiro',
		'agent': 'analista_mercado'
	},
	'criar_estrategia': {
		'description': '''
Com base na análise de mercado, crie uma estratégia de investimentos considerando:
1. Diversificação entre classes de ativos
2. Horizonte de investimento de longo prazo
3. Perfil moderado de risco
4. Objetivos de preservação e crescimento patrimonial
''',
		'expected_output': 'Estratégia detalhada de investimentos com alocação sugerida',
		'agent': 'consultor_investimentos',
		'context': ['analise_mercado']
	},
	'avaliar_riscos': {
		'description': '''
Avalie os riscos da estratégia proposta, considerando:
1. Riscos de mercado
2. Riscos de liquidez
3. Riscos de concentração
4. Propostas de mitigação
''',
		'expected_output': 'Análise de riscos e recomendações de mitigação',
		'agent': 'analista_risco',
		'context': ['criar_estrategia']
	}
}

# Gerar arquivo agents.yaml
with open(agents_path, 'w', encoding='utf-8') as f:
	yaml.dump(agents, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

# Gerar arquivo tasks.yaml
with open(tasks_path, 'w', encoding='utf-8') as f:
	yaml.dump(tasks, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

print(f"Arquivos gerados com sucesso:\n{agents_path}\n{tasks_path}")