import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Type, Union
from enum import Enum
import time
import os
from crewai.tools import BaseTool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
log_file = os.path.join('logs', 'country_data.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class CountryIndicator(BaseModel):
    """Data model for country-specific indicators."""
    name: str
    last: str
    previous: str
    highest: str
    lowest: str
    unit: str
    last_update: str
    country: str

class DetailedIndicator(BaseModel):
    """Data model for detailed indicators from specific tabs."""
    category: str
    name: str
    value: str
    reference: str
    previous: str
    range: str
    frequency: str

class CountryDataInput(BaseModel):
    """Input parameters for the CountryDataTool."""
    countries: List[str] = Field(
        description="Lista de países para buscar dados. Use o código ISO do país (ex: brazil, germany)"
    )
    detailed_analysis: bool = Field(
        default=False,
        description="Se True, obtém dados detalhados de todas as tabelas para o primeiro país da lista"
    )

class CountryDataTool(BaseTool):
    """Tool for fetching economic data for specific countries."""
    name: str = "CountryDataTool"
    description: str = (
        "Ferramenta avançada para análise macroeconômica de países usando dados do Trading Economics. "
        "Fornece dois níveis de análise:\n"
        "1. Análise Básica: Indicadores principais como inflação, PIB, desemprego e atividade econômica para todos os países.\n"
        "2. Análise Detalhada (opcional): Dados aprofundados do primeiro país da lista, incluindo:\n"
        "   • Overview: Visão geral dos principais indicadores e suas tendências\n"
        "   • GDP: Análise completa do PIB, incluindo setores, per capita e crescimento\n"
        "   • Labour: Métricas do mercado de trabalho, emprego e demografia\n"
        "   • Prices: Índices de preços, inflação e custos setoriais\n"
        "   • Money: Indicadores monetários, taxas de juros e crédito\n"
        "   • Trade: Balança comercial, exportações e importações\n"
        "   • Government: Dívida pública, orçamento e gastos governamentais\n"
        "   • Business: Confiança empresarial, produção industrial e setorial\n"
        "   • Consumer: Vendas, crédito e confiança do consumidor\n"
        "   • Housing: Mercado imobiliário e preços de residências\n\n"
        "Os dados são apresentados com indicadores de tendência: ,  (positiva),  (negativa),  (estável), "
        "permitindo rápida identificação do cenário macroeconômico e tomada de decisão."
    )
    args_schema: Type[BaseModel] = CountryDataInput

    base_url: str = Field(default="https://tradingeconomics.com/{}/indicators")
    headers: Dict[str, str] = Field(default={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1"
    })

    def _get_country_data(self, country: str) -> List[CountryIndicator]:
        """Get economic data for a specific country."""
        try:
            url = self.base_url.format(country.lower())
            logger.info(f"Buscando dados para {country} em {url}")
            
            # Add delay between requests
            time.sleep(1)
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'table'})
            
            if not table:
                logger.error(f"Tabela de indicadores não encontrada para {country}")
                return []
            
            indicators = []
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 6:
                    indicator = CountryIndicator(
                        name=cols[0].text.strip(),
                        last=cols[1].text.strip(),
                        previous=cols[2].text.strip(),
                        highest=cols[3].text.strip(),
                        lowest=cols[4].text.strip(),
                        unit=cols[5].text.strip(),
                        last_update=datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        country=country
                    )
                    indicators.append(indicator)
            
            return indicators
            
        except requests.RequestException as e:
            logger.error(f"Erro ao buscar dados para {country}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Erro inesperado ao processar dados para {country}: {str(e)}")
            return []

    def _get_detailed_data(self, country: str) -> Dict[str, List[DetailedIndicator]]:
        """Get detailed data from all tabs for a specific country."""
        detailed_data = {}
        url = self.base_url.format(country.lower())
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Lista de todas as abas disponíveis
            tabs = ['overview', 'gdp', 'labour', 'prices', 'money', 'trade', 
                   'government', 'business', 'consumer', 'housing']
            
            for tab in tabs:
                time.sleep(1)  # Delay entre requisições
                tab_url = f"{url}#{tab}"
                tab_response = requests.get(tab_url, headers=self.headers)
                tab_soup = BeautifulSoup(tab_response.text, 'html.parser')
                
                # Encontra a tabela específica da aba
                tab_table = tab_soup.find('div', {'id': tab}).find('table', {'class': 'table'})
                if not tab_table:
                    logger.warning(f"Tabela não encontrada para a aba {tab} do país {country}")
                    continue
                
                indicators = []
                rows = tab_table.find_all('tr')[1:]  # Pula o cabeçalho
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        indicator = DetailedIndicator(
                            category=tab,
                            name=cols[0].get_text(strip=True),
                            value=cols[1].get_text(strip=True),
                            reference=cols[2].get_text(strip=True),
                            previous=cols[3].get_text(strip=True),
                            range=cols[4].get_text(strip=True),
                            frequency=cols[5].get_text(strip=True)
                        )
                        indicators.append(indicator)
                
                detailed_data[tab] = indicators
                
            return detailed_data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados detalhados para {country}: {str(e)}")
            return {}

    def _format_indicator_output(self, indicator: CountryIndicator) -> str:
        """Format the output for an indicator."""
        def get_change_emoji(current: str, previous: str) -> str:
            try:
                current_val = float(current.replace(',', '').replace('%', ''))
                previous_val = float(previous.replace(',', '').replace('%', ''))
                if current_val > previous_val:
                    return "🟢"
                elif current_val < previous_val:
                    return "🔴"
                return "⚪"
            except:
                return "⚪"

        return (
            f" **{indicator.name}**\n"
            f" Current: {get_change_emoji(indicator.last, indicator.previous)} {indicator.last} {indicator.unit}\n"
            f" Previous: {indicator.previous} {indicator.unit}\n"
            f" History:\n"
            f"  • Maximum: {indicator.highest} {indicator.unit}\n"
            f"  • Minimum: {indicator.lowest} {indicator.unit}\n"
            f" Last Update: {indicator.last_update}"
        )

    def _format_country_name(self, country: str) -> str:
        """Format country name for display."""
        return country.upper()

    def _get_change_emoji(self, current: str, previous: str) -> str:
        """Get emoji indicating change direction."""
        try:
            current_val = float(current.replace(',', '').replace('%', ''))
            previous_val = float(previous.replace(',', '').replace('%', ''))
            if current_val > previous_val:
                return "🟢"
            elif current_val < previous_val:
                return "🔴"
            return "⚪"
        except:
            return "⚪"

    def _run(self, countries: List[str], detailed_analysis: bool = False) -> str:
        """Run the tool with the given input."""
        try:
            all_data = {}
            
            for i, country in enumerate(countries):
                country_data = self._get_country_data(country)
                all_data[country] = {"basic_indicators": country_data}
                
                # Obtém dados detalhados apenas para o primeiro país se detailed_analysis for True
                if i == 0 and detailed_analysis:
                    detailed_data = self._get_detailed_data(country)
                    all_data[country]["detailed_data"] = detailed_data

            # Formatação da saída
            output = []
            for country, data in all_data.items():
                output.append(f"\n🌍 Dados para {country.upper()}:")
                
                # Indicadores básicos
                output.append("\n📊 Indicadores Principais:")
                for indicator in data["basic_indicators"]:
                    change_emoji = self._get_change_emoji(indicator.last, indicator.previous)
                    output.append(
                        f"  • {indicator.name}: {change_emoji} {indicator.last} {indicator.unit} "
                        f"(Anterior: {indicator.previous})"
                    )
                
                # Dados detalhados (se disponíveis)
                if "detailed_data" in data:
                    output.append("\n🔍 Análise Detalhada:")
                    for category, indicators in data["detailed_data"].items():
                        output.append(f"\n  📌 {category.upper()}:")
                        for indicator in indicators:
                            change_emoji = self._get_change_emoji(indicator.value, indicator.previous)
                            output.append(
                                f"    • {indicator.name}: {change_emoji} {indicator.value} "
                                f"(Ref: {indicator.reference}, Anterior: {indicator.previous})"
                            )
            
            return "\n".join(output) if output else "Não foi possível obter dados para os países solicitados."

        except Exception as e:
            logger.error(f"Erro ao processar a requisição: {str(e)}")
            return f"Erro ao processar a requisição: {str(e)}"

if __name__ == '__main__':
    # Exemplo de uso
    tool = CountryDataTool()
    result = tool._run(['brazil', 'germany'], detailed_analysis=True)
    print(result)
