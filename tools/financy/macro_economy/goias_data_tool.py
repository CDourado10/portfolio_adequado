import logging
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Type, Union, Any
from enum import Enum
import pandas as pd
from crewai.tools import BaseTool
import re
import traceback

# Configura o logger
def setup_logger():
    """Configura o logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Cria o diretório de logs se não existir
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configura o arquivo de log
    log_file = os.path.join(log_dir, 'goias_data_tool.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Define o formato do log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Remove handlers existentes
    logger.handlers = []
    
    # Adiciona apenas o file handler
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

class DataCategory(str, Enum):
    """Categorias de dados econômicos."""
    TODOS = "todos"
    PIB = "pib"
    INFLACAO = "inflacao"
    AGRICULTURA = "agricultura"
    PECUARIA = "pecuaria"
    SERVICOS = "servicos"
    COMERCIO = "comercio"
    INDUSTRIA = "industria"
    COMERCIO_EXTERIOR = "comercio_exterior"
    FINANCAS_PUBLICAS = "financas_publicas"
    EMPRESAS = "empresas"

class GoiasIndicator(BaseModel):
    """Modelo para indicadores econômicos de Goiás."""
    categoria: str
    nome: str
    valor_atual: Optional[float] = None
    valor_anterior: Optional[float] = None
    variacao: Optional[float] = None
    periodo: str
    comparacao_nacional: Optional[str] = None
    unidade: Optional[str] = None

class GoiasDataInput(BaseModel):
    """Modelo para entrada de dados da ferramenta GoiasDataTool."""
    categorias: List[DataCategory] = Field(
        default=[DataCategory.TODOS],
        description="""
        Lista de categorias de dados econômicos para análise.
        
        Deve ser uma lista contendo uma ou mais das seguintes strings:
        - "todos": Todos os indicadores disponíveis
        - "pib": PIB total e setorial
        - "inflacao": IPCA regional
        - "agricultura": Produção agrícola
        - "pecuaria": Produção pecuária
        - "servicos": Setor de serviços
        - "comercio": Setor comercial
        - "industria": Produção industrial
        - "comercio_exterior": Comércio internacional
        - "financas_publicas": Finanças do estado
        - "empresas": Setor empresarial
        
        Exemplo: ["pib", "industria", "comercio"]
        """
    )
    analise_comparativa: bool = Field(
        default=True,
        description="""
        Se True, inclui comparações com indicadores nacionais para cada métrica,
        permitindo avaliar o desempenho de Goiás em relação ao Brasil.
        Recomendado manter True para uma análise mais completa.
        
        Exemplo: true
        """
    )

class GoiasDataTool(BaseTool):
    """Ferramenta para análise de dados econômicos de Goiás."""
    name: str = "GoiasDataTool"
    description: str = """
    Ferramenta avançada para análise de dados econômicos do estado de Goiás. Processa dados de diversas fontes e gera insights sobre:

    1. PIB e Crescimento:
       - PIB total e per capita
       - Composição setorial do PIB
       - Ranking nacional e participação no PIB brasileiro
       - Variações trimestrais e anuais

    2. Setores Econômicos:
       - Indústria: produção física, emprego e segmentos
       - Agricultura: principais culturas, área e produtividade
       - Pecuária: rebanhos, produção de leite e carnes
       - Serviços: volume, emprego e segmentação
       - Comércio: vendas, e-commerce e indicadores

    3. Comércio Exterior:
       - Balança comercial (exportações e importações)
       - Principais produtos e commodities
       - Destinos e origens do comércio
       - Variação cambial e competitividade

    4. Finanças Públicas:
       - Arrecadação de impostos (federais e estaduais)
       - Gastos governamentais por função
       - Indicadores fiscais e dívida pública
       - Transferências e investimentos

    5. Análise Comparativa:
       - Comparação com médias nacionais
       - Posicionamento regional (Centro-Oeste)
       - Benchmarking com estados similares
       - Análise temporal (variações entre períodos)

    Indicadores de Tendência:
    - Positiva: crescimento em relação ao período anterior
    - Negativa: queda em relação ao período anterior
    - Estável: variação inferior a zero ponto um por cento
    - Informativo: dado pontual ou sem variação disponível

    Formatação dos Dados:
    - Valores monetários no padrão brasileiro (um mil duzentos e trinta e quatro reais e cinquenta e seis centavos)
    - Percentuais com duas casas decimais (doze ponto trinta e quatro por cento)
    - Números grandes em milhões ou bilhões quando apropriado

    Periodicidade:
    - Dados mensais: indicadores conjunturais
    - Dados trimestrais: PIB e setores
    - Dados anuais: análises estruturais

    Todos os valores são ajustados para inflação e sazonalidade quando aplicável.
    """
    args_schema: Type[BaseModel] = GoiasDataInput
    csv_dir: str = Field(default="")
    
    def __init__(self, csv_dir: Optional[str] = None):
        """
        Inicializa a ferramenta.
        
        Args:
            csv_dir: Caminho para o diretório com os arquivos CSV. Se não fornecido, usa o padrão.
        """
        try:
            # Configura o logger
            log_file = 'goias_data_tool.log'
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, mode='w', encoding='utf-8')
                ]
            )
            
            # Define o caminho do diretório CSV
            if csv_dir is None:
                # Obtém o diretório do arquivo atual
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # Sobe três níveis para chegar na raiz do projeto
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
                # Define o caminho para o diretório de dados
                csv_dir = os.path.join(project_root, 'dados_goias', 'csv_tables')
            
            logger.info(f"Diretório CSV: {csv_dir}")
            if not os.path.exists(csv_dir):
                raise FileNotFoundError(f"Diretório não encontrado: {csv_dir}")
            
            # Inicializa a classe base com o csv_dir
            super().__init__(csv_dir=csv_dir)
                
        except Exception as e:
            logger.error(f"Erro ao inicializar a ferramenta: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _format_currency(self, value: float) -> str:
        """Formata valores monetários no padrão brasileiro."""
        try:
            if pd.isna(value):
                return "N/D"
            return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(value)

    def _format_percentage(self, value: float) -> str:
        """Formata percentuais no padrão brasileiro."""
        try:
            if pd.isna(value):
                return "N/D"
            return f"{value:,.2f}%".replace(".", ",")
        except:
            return str(value)

    def _get_trend_emoji(self, current: float, previous: float) -> str:
        """Retorna emoji indicando tendência."""
        if pd.isna(current) or pd.isna(previous):
            return "ℹ️"
        diff = current - previous
        if abs(diff) < 0.001:
            return "➡️"
        return "🟢" if diff > 0 else "🔴"

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Carrega um arquivo CSV."""
        try:
            filepath = os.path.join(self.csv_dir, filename)
            if not os.path.exists(filepath):
                logger.error(f"Arquivo não encontrado: {filepath}")
                return pd.DataFrame()
                
            # Verifica se o arquivo está vazio
            if os.path.getsize(filepath) == 0:
                logger.error(f"Arquivo vazio: {filepath}")
                return pd.DataFrame()
            
            # Tenta diferentes separadores e encodings
            for sep in [',', ';']:
                for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(filepath, sep=sep, encoding=encoding)
                        if not df.empty:
                            # Limpa nomes das colunas
                            df.columns = df.columns.str.strip().str.replace(' ', '_')
                            return df
                    except:
                        continue
            
            logger.error(f"Não foi possível ler o arquivo {filename} com nenhuma configuração")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Erro ao carregar CSV {filename}: {str(e)}")
            return pd.DataFrame()

    def _clean_numeric(self, value: Any) -> float:
        """Limpa e converte valores numéricos."""
        if pd.isna(value):
            return 0.0
            
        if isinstance(value, (int, float)):
            return float(value)
            
        # Remove caracteres não numéricos, exceto ponto e vírgula
        value = str(value)
        value = re.sub(r'[^\d,.-]', '', value)
        
        # Trata diferentes formatos de número
        try:
            if ',' in value and '.' in value:
                # Formato brasileiro (ex: 1.234,56)
                value = value.replace('.', '').replace(',', '.')
            elif ',' in value:
                # Vírgula como decimal
                value = value.replace(',', '.')
                
            return float(value)
        except:
            return 0.0

    def _format_value(self, value: float, unidade: Optional[str] = None) -> str:
        """Formata um valor de acordo com sua unidade."""
        if value is None:
            return "N/A"
            
        # Se não tiver unidade, retorna o valor formatado
        if not unidade:
            return f"{value:,.2f}"
            
        # Ajusta o valor com base na unidade
        if unidade == "%":
            return f"{value:.2f}%"
        elif "milhões" in unidade.lower():
            return f"{value/1_000_000:,.2f}"
        elif unidade == "R$ mil":
            return f"{value/1_000:,.2f}"
        elif unidade == "R$":
            return f"{value:,.2f}"
        else:
            return f"{value:,.2f}"

    def _process_pib_data(self) -> List[GoiasIndicator]:
        """Processa dados do PIB."""
        indicators = []
        try:
            # PIB Total e per capita
            df_pib = self._load_csv("82-crescimento-trimestral-pib-goias-brasil.csv")
            if not df_pib.empty:
                latest_year = df_pib.columns[-1]
                prev_year = df_pib.columns[-2]
                
                for _, row in df_pib.iterrows():
                    tipo = row['Tipo']
                    valor_atual = self._clean_numeric(row[latest_year])
                    valor_anterior = self._clean_numeric(row[prev_year])
                    
                    if valor_atual != 0 or valor_anterior != 0:
                        indicators.append(GoiasIndicator(
                            categoria="PIB",
                            nome=tipo,
                            valor_atual=valor_atual,
                            valor_anterior=valor_anterior,
                            variacao=((valor_atual / valor_anterior) - 1) * 100 if valor_anterior != 0 else None,
                            periodo=f"{prev_year}-{latest_year}",
                            unidade="R$ milhões" if "per capita" not in tipo.lower() else "R$"
                        ))
            
            # Variação setorial
            df_var = self._load_csv("81-variacao-anual-pib-goiano.csv")
            if not df_var.empty:
                latest_year = df_var.columns[-1]
                prev_year = df_var.columns[-2]
                
                for _, row in df_var.iterrows():
                    setor = row['Setor']
                    valor_atual = self._clean_numeric(row[latest_year])
                    valor_anterior = self._clean_numeric(row[prev_year])
                    
                    if valor_atual != 0 or valor_anterior != 0:
                        indicators.append(GoiasIndicator(
                            categoria="PIB Setorial",
                            nome=f"Setor {setor}",
                            valor_atual=valor_atual,
                            valor_anterior=valor_anterior,
                            variacao=valor_atual - valor_anterior,  # Já é variação percentual
                            periodo=f"{prev_year}-{latest_year}",
                            unidade="%"
                        ))
                        
            # Ranking nacional
            df_rank = self._load_csv("83-ranking-pib-unidade-federacao.csv")
            if not df_rank.empty:
                latest_year = df_rank.columns[-1]
                prev_year = df_rank.columns[-2]
                
                goias_row = df_rank[df_rank['UF'].str.contains('Goiás', case=False, na=False)].iloc[0]
                posicao = int(goias_row['Posição'])
                valor_atual = self._clean_numeric(goias_row[latest_year])
                valor_anterior = self._clean_numeric(goias_row[prev_year])
                
                if valor_atual != 0 or valor_anterior != 0:
                    indicators.append(GoiasIndicator(
                        categoria="PIB",
                        nome="PIB Total",
                        valor_atual=valor_atual,
                        valor_anterior=valor_anterior,
                        variacao=((valor_atual / valor_anterior) - 1) * 100 if valor_anterior != 0 else None,
                        periodo=f"{prev_year}-{latest_year}",
                        unidade="R$ milhões",
                        comparacao_nacional=f"{posicao}º maior PIB do Brasil"
                    ))
                    
        except Exception as e:
            logger.error(f"Erro ao processar dados do PIB: {str(e)}")
        return indicators

    def _process_industry_data(self) -> List[GoiasIndicator]:
        """Processa dados da indústria."""
        indicators = []
        try:
            df = self._load_csv("109-crescimento-producao-industrial-goias.csv")
            if not df.empty:
                latest_year = df.columns[-1]
                prev_year = df.columns[-2]
                
                for _, row in df.iterrows():
                    indicator = GoiasIndicator(
                        categoria="Indústria",
                        nome=row['Segmentos'],
                        valor_atual=row[latest_year],
                        valor_anterior=row[prev_year],
                        variacao=row[latest_year] - row[prev_year],
                        periodo=f"{prev_year}-{latest_year}"
                    )
                    indicators.append(indicator)
                    
        except Exception as e:
            logger.error(f"Erro ao processar dados da indústria: {str(e)}")
        
        return indicators

    def _process_livestock_data(self) -> List[GoiasIndicator]:
        """Processa dados da pecuária."""
        indicators = []
        try:
            # Rebanhos e produção de leite
            df = self._load_csv("98-rebanhos-producao-leite-goias-centro-oeste-brasil.csv")
            if not df.empty:
                # Filtra apenas os dados de Goiás
                df_goias = df[df['Especificação'] == 'Goiás'].sort_values('Ano', ascending=True)
                
                if not df_goias.empty:
                    # Pega os dois últimos anos
                    anos = sorted(df_goias['Ano'].unique())
                    if len(anos) >= 2:
                        ano_atual = anos[-1]
                        ano_anterior = anos[-2]
                        
                        dados_atual = df_goias[df_goias['Ano'] == ano_atual].iloc[0]
                        dados_anterior = df_goias[df_goias['Ano'] == ano_anterior].iloc[0]
                        
                        # Processa cada tipo de rebanho
                        for coluna in ['Bovino', 'Suíno', 'Aves']:
                            valor_atual = self._clean_numeric(dados_atual[coluna])
                            valor_anterior = self._clean_numeric(dados_anterior[coluna])
                            
                            if valor_atual != 0 or valor_anterior != 0:
                                indicators.append(GoiasIndicator(
                                    categoria="Pecuária",
                                    nome=f"Rebanho {coluna}",
                                    valor_atual=valor_atual,
                                    valor_anterior=valor_anterior,
                                    variacao=((valor_atual / valor_anterior) - 1) * 100 if valor_anterior != 0 else None,
                                    periodo=f"{ano_anterior}-{ano_atual}",
                                    unidade="cabeças"
                                ))
                        
                        # Processa produção de leite
                        leite_atual = self._clean_numeric(dados_atual['Produção_de_leite'])
                        leite_anterior = self._clean_numeric(dados_anterior['Produção_de_leite'])
                        
                        if leite_atual != 0 or leite_anterior != 0:
                            indicators.append(GoiasIndicator(
                                categoria="Pecuária",
                                nome="Produção de Leite",
                                valor_atual=leite_atual,
                                valor_anterior=leite_anterior,
                                variacao=((leite_atual / leite_anterior) - 1) * 100 if leite_anterior != 0 else None,
                                periodo=f"{ano_anterior}-{ano_atual}",
                                unidade="mil litros"
                            ))
                            
            # Abate de animais
            df_abate = self._load_csv("101-abate-animais-goias.csv")
            if not df_abate.empty:
                latest_year = df_abate.columns[-1]
                prev_year = df_abate.columns[-2]
                
                for _, row in df_abate.iterrows():
                    tipo = row['Tipo']
                    valor_atual = self._clean_numeric(row[latest_year])
                    valor_anterior = self._clean_numeric(row[prev_year])
                    
                    if valor_atual != 0 or valor_anterior != 0:
                        indicators.append(GoiasIndicator(
                            categoria="Pecuária",
                            nome=f"Abate de {tipo}",
                            valor_atual=valor_atual,
                            valor_anterior=valor_anterior,
                            variacao=((valor_atual / valor_anterior) - 1) * 100 if valor_anterior != 0 else None,
                            periodo=f"{prev_year}-{latest_year}",
                            unidade="cabeças"
                        ))
                        
        except Exception as e:
            logger.error(f"Erro ao processar dados da pecuária: {str(e)}")
        return indicators

    def _process_inflation_data(self) -> List[GoiasIndicator]:
        """Processa dados de inflação."""
        indicators = []
        try:
            # INPC
            df_inpc = self._load_csv("84-variacao-inpc-capitais-brasileiras.csv")
            if not df_inpc.empty:
                # Filtra dados de Goiânia
                goiania_row = df_inpc[df_inpc['Localidade'] == 'Goiânia'].iloc[0]
                
                # Variação mensal
                valor_mensal = self._clean_numeric(goiania_row['Variação_Mensal_%'])
                indicators.append(GoiasIndicator(
                    categoria="Inflação",
                    nome="INPC - Variação Mensal",
                    valor_atual=valor_mensal,
                    valor_anterior=0,  # Não temos o mês anterior
                    variacao=valor_mensal,  # A própria variação é o valor
                    periodo="Último mês",
                    unidade="%",
                    comparacao_nacional=f"Brasil: {df_inpc[df_inpc['Localidade'] == 'Brasil']['Variação_Mensal_%'].iloc[0]}%"
                ))
                
                # Variação acumulada
                valor_acumulado = self._clean_numeric(goiania_row['Variação_acumulada_no_ano_%'])
                indicators.append(GoiasIndicator(
                    categoria="Inflação",
                    nome="INPC - Acumulado no Ano",
                    valor_atual=valor_acumulado,
                    valor_anterior=0,  # Não temos o ano anterior
                    variacao=valor_acumulado,  # A própria variação é o valor
                    periodo="Ano atual",
                    unidade="%",
                    comparacao_nacional=f"Brasil: {df_inpc[df_inpc['Localidade'] == 'Brasil']['Variação_acumulada_no_ano_%'].iloc[0]}%"
                ))
            
            # IPCA
            df_ipca = self._load_csv("85-variacao-ipca-capitais-brasileiras.csv")
            if not df_ipca.empty:
                # Filtra dados de Goiânia
                goiania_row = df_ipca[df_ipca['Localidade'] == 'Goiânia'].iloc[0]
                
                # Variação mensal
                valor_mensal = self._clean_numeric(goiania_row['Variação_Mensal_%'])
                indicators.append(GoiasIndicator(
                    categoria="Inflação",
                    nome="IPCA - Variação Mensal",
                    valor_atual=valor_mensal,
                    valor_anterior=0,  # Não temos o mês anterior
                    variacao=valor_mensal,  # A própria variação é o valor
                    periodo="Último mês",
                    unidade="%",
                    comparacao_nacional=f"Brasil: {df_ipca[df_ipca['Localidade'] == 'Brasil']['Variação_Mensal_%'].iloc[0]}%"
                ))
                
                # Variação acumulada
                valor_acumulado = self._clean_numeric(goiania_row['Variação_acumulada_no_ano_%'])
                indicators.append(GoiasIndicator(
                    categoria="Inflação",
                    nome="IPCA - Acumulado no Ano",
                    valor_atual=valor_acumulado,
                    valor_anterior=0,  # Não temos o ano anterior
                    variacao=valor_acumulado,  # A própria variação é o valor
                    periodo="Ano atual",
                    unidade="%",
                    comparacao_nacional=f"Brasil: {df_ipca[df_ipca['Localidade'] == 'Brasil']['Variação_acumulada_no_ano_%'].iloc[0]}%"
                ))
                        
        except Exception as e:
            logger.error(f"Erro ao processar dados de inflação: {str(e)}")
        return indicators

    def _process_agriculture_data(self) -> List[GoiasIndicator]:
        """Processa dados agrícolas."""
        indicators = []
        try:
            # Produção de grãos
            df_graos = self._load_csv("90-producao-graos-goias-centro-oeste-brasil.csv")
            if not df_graos.empty:
                latest_year = df_graos.columns[-1]
                prev_year = df_graos.columns[-2]
                
                goias_data = df_graos[df_graos['Região'] == 'Goiás'].iloc[0]
                
                indicators.append(GoiasIndicator(
                    categoria="Agricultura",
                    nome="Produção de Grãos",
                    valor_atual=goias_data[latest_year],
                    valor_anterior=goias_data[prev_year],
                    variacao=((goias_data[latest_year] / goias_data[prev_year]) - 1) * 100,
                    periodo=f"{prev_year}-{latest_year}",
                    comparacao_nacional="4º maior produtor nacional",
                    unidade="mil toneladas"
                ))
            
            # Principais produtos
            df_prod = self._load_csv("91-producao-principais-produtos-agricolas-goias.csv")
            if not df_prod.empty:
                latest_year = df_prod.columns[-1]
                prev_year = df_prod.columns[-2]
                
                for _, row in df_prod.iterrows():
                    produto = row['Produto']
                    indicators.append(GoiasIndicator(
                        categoria="Agricultura",
                        nome=f"Produção de {produto}",
                        valor_atual=row[latest_year],
                        valor_anterior=row[prev_year],
                        variacao=((row[latest_year] / row[prev_year]) - 1) * 100,
                        periodo=f"{prev_year}-{latest_year}",
                        unidade="toneladas"
                    ))
        except Exception as e:
            logger.error(f"Erro ao processar dados agrícolas: {str(e)}")
        return indicators

    def _process_trade_data(self) -> List[GoiasIndicator]:
        """Processa dados de comércio exterior."""
        indicators = []
        try:
            # Produtos líderes de exportação
            df_exp = self._load_csv("113-produtos-lideres-exportacao-goias.csv")
            if not df_exp.empty:
                latest_year = df_exp.columns[-1]
                prev_year = df_exp.columns[-2]
                
                # Top 5 produtos
                top_produtos = df_exp.nlargest(5, latest_year)
                for _, row in top_produtos.iterrows():
                    produto = row['Produto']
                    indicators.append(GoiasIndicator(
                        categoria="Comércio Exterior",
                        nome=f"Exportação - {produto}",
                        valor_atual=row[latest_year],
                        valor_anterior=row[prev_year],
                        variacao=((row[latest_year] / row[prev_year]) - 1) * 100,
                        periodo=f"{prev_year}-{latest_year}",
                        unidade="US$ milhões"
                    ))
            
            # Principais destinos
            df_dest = self._load_csv("115-destinos-exportacoes-goias.csv")
            if not df_dest.empty:
                latest_year = df_dest.columns[-1]
                prev_year = df_dest.columns[-2]
                
                # Top 3 destinos
                top_destinos = df_dest.nlargest(3, latest_year)
                for _, row in top_destinos.iterrows():
                    pais = row['País']
                    indicators.append(GoiasIndicator(
                        categoria="Comércio Exterior",
                        nome=f"Exportação para {pais}",
                        valor_atual=row[latest_year],
                        valor_anterior=row[prev_year],
                        variacao=((row[latest_year] / row[prev_year]) - 1) * 100,
                        periodo=f"{prev_year}-{latest_year}",
                        unidade="US$ milhões"
                    ))
        except Exception as e:
            logger.error(f"Erro ao processar dados de comércio exterior: {str(e)}")
        return indicators

    def _process_public_finance_data(self) -> List[GoiasIndicator]:
        """Processa dados de finanças públicas."""
        indicators = []
        try:
            # Arrecadação de impostos federais
            df_fed = self._load_csv("123-arrecadacao-impostos-federais-goias.csv")
            if not df_fed.empty:
                latest_year = str(df_fed.columns[-1])
                prev_year = str(df_fed.columns[-2])
                
                # Top 5 impostos por arrecadação
                df_fed['valor_atual'] = df_fed[latest_year].apply(self._clean_numeric)
                top_impostos = df_fed.nlargest(5, 'valor_atual')
                
                for _, row in top_impostos.iterrows():
                    valor_atual = self._clean_numeric(row[latest_year])
                    valor_anterior = self._clean_numeric(row[prev_year])
                    
                    if valor_atual != 0 or valor_anterior != 0:
                        indicators.append(GoiasIndicator(
                            categoria="Finanças Públicas",
                            nome=f"Arrecadação - {row['RECEITAS']}",
                            valor_atual=valor_atual,
                            valor_anterior=valor_anterior,
                            variacao=((valor_atual / valor_anterior) - 1) * 100 if valor_anterior != 0 else None,
                            periodo=f"{prev_year}-{latest_year}",
                            unidade="R$"
                        ))
            
            # Gastos do governo
            df_gastos = self._load_csv("126-gastos-governo-goias-funcao-categoria.csv")
            if not df_gastos.empty:
                latest_year = df_gastos.columns[-1]
                prev_year = df_gastos.columns[-2]
                
                for _, row in df_gastos.iterrows():
                    valor_atual = self._clean_numeric(row[latest_year])
                    valor_anterior = self._clean_numeric(row[prev_year])
                    
                    if valor_atual != 0 or valor_anterior != 0:
                        indicators.append(GoiasIndicator(
                            categoria="Finanças Públicas",
                            nome=f"Gastos - {row['Função']}",
                            valor_atual=valor_atual,
                            valor_anterior=valor_anterior,
                            variacao=((valor_atual / valor_anterior) - 1) * 100 if valor_anterior != 0 else None,
                            periodo=f"{prev_year}-{latest_year}",
                            unidade="R$ milhões"
                        ))
        except Exception as e:
            logger.error(f"Erro ao processar dados de finanças públicas: {str(e)}")
        return indicators

    def _process_business_data(self) -> List[GoiasIndicator]:
        """Processa dados de empresas e serviços."""
        indicators = []
        try:
            # Serviços não financeiros
            df_serv = self._load_csv("104-servicos-nao-financeiros-goias.csv")
            if not df_serv.empty:
                # Processa receita bruta por setor
                receita_col = [col for col in df_serv.columns if 'receita' in col.lower()][0]
                pessoal_col = [col for col in df_serv.columns if 'pessoal' in col.lower()][0]
                
                for _, row in df_serv.iterrows():
                    if pd.notna(row[receita_col]) and str(row['Atividades_de_serviços']).strip():
                        indicators.append(GoiasIndicator(
                            categoria="Serviços",
                            nome=row['Atividades_de_serviços'],
                            valor_atual=self._clean_numeric(row[receita_col]),
                            valor_anterior=None,  # Dados são apenas do último ano
                            variacao=None,
                            periodo="2023",  # Ajustar conforme o ano dos dados
                            unidade="R$ mil",
                            comparacao_nacional=f"Pessoal ocupado: {row[pessoal_col]:,.0f}"
                        ))
            
            # Indicadores do comércio
            df_com = self._load_csv("106-indicadores-comercio-goias-brasil.csv")
            if not df_com.empty:
                latest_year = df_com.columns[-1]
                prev_year = df_com.columns[-2]
                
                for _, row in df_com.iterrows():
                    indicador = row['Indicador']
                    valor_atual = self._clean_numeric(row[latest_year])
                    valor_anterior = self._clean_numeric(row[prev_year])
                    
                    if valor_atual != 0 or valor_anterior != 0:
                        indicators.append(GoiasIndicator(
                            categoria="Comércio",
                            nome=indicador,
                            valor_atual=valor_atual,
                            valor_anterior=valor_anterior,
                            variacao=((valor_atual / valor_anterior) - 1) * 100 if valor_anterior != 0 else None,
                            periodo=f"{prev_year}-{latest_year}",
                            unidade="índice"
                        ))
        except Exception as e:
            logger.error(f"Erro ao processar dados de empresas e serviços: {str(e)}")
        return indicators

    def _add_national_comparisons(self, indicators: List[GoiasIndicator]) -> List[GoiasIndicator]:
        """Adiciona comparações nacionais aos indicadores."""
        try:
            # Carrega dados nacionais
            df_nacional = self._load_csv("comparacao_nacional.csv")
            if not df_nacional.empty:
                for indicator in indicators:
                    # Busca o indicador nacional correspondente
                    nacional = df_nacional[
                        df_nacional['indicador'].str.contains(indicator.nome, case=False, na=False)
                    ]
                    
                    if not nacional.empty:
                        valor_nacional = self._clean_numeric(nacional.iloc[0]['valor'])
                        if valor_nacional > 0 and indicator.valor_atual is not None:
                            # Calcula a diferença percentual
                            diff = ((indicator.valor_atual / valor_nacional) - 1) * 100
                            indicator.comparacao_nacional = f"{diff:+.1f}% em relação à média nacional"
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao adicionar comparações nacionais: {str(e)}")
            return indicators
            
    def _sanitize_string(self, text: str) -> str:
        """Sanitiza strings com caracteres especiais para evitar problemas de formatação."""
        if not text:
            return ''
        # Escapa caracteres especiais e emojis
        return text.encode('unicode_escape').decode('ascii')

    def _format_report(self, indicators: List[GoiasIndicator]) -> str:
        """Formata o relatório com os indicadores."""

        # Agrupa indicadores por categoria
        categorias = {}
        for ind in indicators:
            if ind.categoria not in categorias:
                categorias[ind.categoria] = []
            categorias[ind.categoria].append(ind)

        # Ordem das categorias
        ordem_categorias = {
            'PIB': 1,
            'PIB Setorial': 2,
            'Indústria': 3,
            'Agricultura': 4,
            'Pecuária': 5,
            'Comércio Exterior': 6,
            'Finanças Públicas': 7,
            'Inflação': 8,
            'Serviços': 9
        }

        # Formata a saída
        output = []
        for categoria, inds in sorted(categorias.items(), key=lambda x: ordem_categorias.get(x[0], 99)):
            # Cabeçalho da categoria
            output.append(self._sanitize_string(f"\n 📊 {categoria}"))
            output.append(self._sanitize_string("=" * (len(categoria) + 2)))

            # Calcula estatísticas da categoria
            valores_atuais = [ind.valor_atual for ind in inds if ind.valor_atual is not None]
            variacoes = [ind.variacao for ind in inds if ind.variacao is not None]

            if valores_atuais and len(valores_atuais) > 1:
                media_atual = sum(valores_atuais) / len(valores_atuais)
                output.append(self._sanitize_string(f"📈 Média: {self._format_value(media_atual, inds[0].unidade)} ({len(valores_atuais)} indicadores)\n"))

            if variacoes and len(variacoes) > 1:
                media_var = sum(variacoes) / len(variacoes)
                output.append(self._sanitize_string(f"📊 Variação média: {media_var:+.2f}%\n"))  # Adicionado %% para evitar erro

            # Indicadores da categoria
            for ind in sorted(inds, key=lambda x: (-(x.variacao or 0))):
                # Define o emoji baseado na variação
                if ind.variacao is None:
                    emoji = "ℹ️"
                elif ind.variacao > 0:
                    emoji = "🟢"
                elif ind.variacao < 0:
                    emoji = "🔴"
                else:
                    emoji = "➡️"

                # Formata o valor atual
                valor_atual = self._format_value(ind.valor_atual, ind.unidade) if ind.valor_atual is not None else "N/D"

                # Formata a variação
                if ind.variacao is not None:
                    variacao = f"{ind.variacao:+.1f}%" if ind.variacao != 0 else "0%"  # Adicionado %% para evitar erro
                else:
                    variacao = "N/D"

                # Linha do indicador
                linha = f"{emoji} {ind.nome}: {valor_atual} ({variacao})"

                # Adiciona comparação nacional se disponível
                if ind.comparacao_nacional:
                    linha += f" | {ind.comparacao_nacional}"

                output.append(self._sanitize_string(linha))

            output.append("")  # Linha em branco entre categorias

        # Junta toda a saída e remove caracteres problemáticos
        output_text = "\n".join(output)
        output_text = output_text.replace("\\", "")  # Remove barras invertidas
        output_text = output_text.encode('latin1').decode('utf-8')
        return output_text



    def process_data(self) -> str:
        """Processa todos os dados disponíveis."""
        try:
            all_indicators = []
            
            # Processa PIB
            all_indicators.extend(self._process_pib_data())
            
            # Processa indústria
            all_indicators.extend(self._process_industry_data())
            
            # Processa agricultura
            all_indicators.extend(self._process_agriculture_data())
            
            # Processa pecuária
            all_indicators.extend(self._process_livestock_data())
            
            # Processa comércio exterior
            all_indicators.extend(self._process_trade_data())
            
            # Processa finanças públicas
            all_indicators.extend(self._process_public_finance_data())
            
            # Processa inflação
            all_indicators.extend(self._process_inflation_data())
            
            # Processa serviços
            all_indicators.extend(self._process_business_data())
            
            all_indicators = all_indicators.encode('latin1').decode('utf-8')
            # Formata e retorna a saída
            return self._format_report(all_indicators)
            
        except Exception as e:
            logger.error(f"Erro ao processar os dados: {str(e)}")
            return f"❌ Erro ao processar os dados: {str(e)}"

    def _run(
        self,
        categorias: List[DataCategory] = [DataCategory.TODOS],
        analise_comparativa: bool = True
    ) -> str:
        """
        Executa a análise dos dados econômicos de Goiás.
        
        Args:
            categorias: Lista de categorias para análise
            analise_comparativa: Se deve incluir comparações nacionais
            
        Returns:
            str: Relatório formatado com os resultados da análise
        """
        try:
            logger.info(f"Iniciando análise para categorias: {categorias}")
            
            all_indicators = []
            
            # Processamento dos indicadores por categoria
            for categoria in categorias:
                if categoria in [DataCategory.TODOS, DataCategory.PIB]:
                    all_indicators.extend(self._process_pib_data())
                    
                if categoria in [DataCategory.TODOS, DataCategory.INFLACAO]:
                    all_indicators.extend(self._process_inflation_data())
                    
                if categoria in [DataCategory.TODOS, DataCategory.AGRICULTURA]:
                    all_indicators.extend(self._process_agriculture_data())
                    
                if categoria in [DataCategory.TODOS, DataCategory.PECUARIA]:
                    all_indicators.extend(self._process_livestock_data())
                    
                if categoria in [DataCategory.TODOS, DataCategory.SERVICOS]:
                    all_indicators.extend(self._process_business_data())
                    
                if categoria in [DataCategory.TODOS, DataCategory.COMERCIO]:
                    all_indicators.extend(self._process_business_data())
                    
                if categoria in [DataCategory.TODOS, DataCategory.INDUSTRIA]:
                    all_indicators.extend(self._process_industry_data())
                    
                if categoria in [DataCategory.TODOS, DataCategory.COMERCIO_EXTERIOR]:
                    all_indicators.extend(self._process_trade_data())
                    
                if categoria in [DataCategory.TODOS, DataCategory.FINANCAS_PUBLICAS]:
                    all_indicators.extend(self._process_public_finance_data())
                    
                if categoria in [DataCategory.TODOS, DataCategory.EMPRESAS]:
                    all_indicators.extend(self._process_business_data())
            
            # Adiciona comparações nacionais se solicitado
            if analise_comparativa:
                all_indicators = self._add_national_comparisons(all_indicators)
            
            # Formata e retorna o relatório
            return self._format_report(all_indicators)
            
        except Exception as e:
            error_msg = f"Erro ao executar a ferramenta: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return f"❌ Erro ao processar os dados: {str(e)}"

if __name__ == "__main__":
    # Exemplo de uso
    tool = GoiasDataTool()
    
    # Exemplo 1: Todas as categorias
    print("\nExemplo 1: Todas as categorias")
    result = tool._run(
        categorias=[DataCategory.TODOS],
        analise_comparativa=True
    )
    print(result)
    
    # Exemplo 2: Categorias específicas
    print("\nExemplo 2: PIB e Indústria")
    result = tool._run(
        categorias=["pib", "industria"],
        analise_comparativa=True
    )
    print(result)
