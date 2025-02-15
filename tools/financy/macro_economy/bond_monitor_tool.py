import logging
from typing import List, Optional, Type
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from datetime import datetime
import pytz

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BondData(BaseModel):
    """Model for bond data."""
    country: str
    yield_value: float
    daily_change: float
    weekly_change: float
    monthly_change: float
    ytd_change: Optional[float]
    year_change: Optional[float]
    last_update: str

class BondMonitorInput(BaseModel):
    """Input parameters for the BondMonitorTool."""
    pass

class BondMonitorTool(BaseTool):
    """Tool to monitor government bond yields from Trading Economics."""
    name: str = Field(default="BondMonitorTool")
    description: str = Field(default=(
        "Monitors and analyzes government bond yields from Trading Economics for major economies. "
        "Provides real-time data on 10-year government bond yields, including current rates and "
        "variations over different timeframes (daily, weekly, monthly, YTD, and yearly). "
        "Data is presented with clear visual indicators: 🟢 for positive changes, 🔴 for negative changes, "
        "and ⚪ for no change."
    ))
    args_schema: Type[BaseModel] = BondMonitorInput
    headers: dict = Field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    BONDS_URL: str = 'https://tradingeconomics.com/bonds'

    def _get_bond_data(self) -> List[BondData]:
        """Get bond data from Trading Economics."""
        try:
            response = requests.get(self.BONDS_URL, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': lambda x: x and x.startswith('bond-')})
            
            if not table:
                logger.error("Bond table not found")
                return []
            
            bonds = []
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) < 9:
                        continue
                    
                    # Extract country name
                    country = cells[1].find('b').text.strip()
                    
                    # Extract numeric values
                    yield_value = float(cells[2].text.strip().replace(',', ''))
                    daily_change = float(cells[3].text.strip().split()[-1].replace(',', ''))
                    
                    # Extract percentage changes
                    weekly_change = float(cells[4].text.strip().rstrip('%').replace(',', ''))
                    monthly_change = float(cells[5].text.strip().rstrip('%').replace(',', ''))
                    
                    # YTD and YoY are optional
                    try:
                        ytd_change = float(cells[6].text.strip().rstrip('%').replace(',', ''))
                    except (ValueError, AttributeError):
                        ytd_change = None
                        
                    try:
                        year_change = float(cells[7].text.strip().rstrip('%').replace(',', ''))
                    except (ValueError, AttributeError):
                        year_change = None
                    
                    last_update = cells[8].text.strip()
                    
                    bond = BondData(
                        country=country,
                        yield_value=yield_value,
                        daily_change=daily_change,
                        weekly_change=weekly_change,
                        monthly_change=monthly_change,
                        ytd_change=ytd_change,
                        year_change=year_change,
                        last_update=last_update
                    )
                    bonds.append(bond)
                    
                except Exception as e:
                    logger.error(f"Error parsing bond row: {str(e)}")
                    continue
            
            return bonds
            
        except Exception as e:
            logger.error(f"Error fetching bond data: {str(e)}")
            return []

    def _get_performance_emoji(self, value: float) -> str:
        """Get emoji indicator for performance."""
        if value > 0:
            return "🟢"
        elif value < 0:
            return "🔴"
        return "⚪"

    def _format_bond_output(self, bond: BondData) -> str:
        """Format bond data for output."""
        daily_emoji = self._get_performance_emoji(bond.daily_change)
        weekly_emoji = self._get_performance_emoji(bond.weekly_change)
        monthly_emoji = self._get_performance_emoji(bond.monthly_change)
        ytd_emoji = self._get_performance_emoji(bond.ytd_change) if bond.ytd_change is not None else "⚪"
        year_emoji = self._get_performance_emoji(bond.year_change) if bond.year_change is not None else "⚪"

        output = [
            f"📊 **{bond.country}**",
            f"💰 Yield: {bond.yield_value:.4f}%",
            f"📈 Daily: {daily_emoji} {bond.daily_change:+.4f}",
            "📊 Changes:",
            f"  • Weekly: {weekly_emoji} {bond.weekly_change:+.2f}%",
            f"  • Monthly: {monthly_emoji} {bond.monthly_change:+.2f}%"
        ]

        if bond.ytd_change is not None:
            output.append(f"  • YTD: {ytd_emoji} {bond.ytd_change:+.2f}%")
        if bond.year_change is not None:
            output.append(f"  • Year: {year_emoji} {bond.year_change:+.2f}%")

        output.append(f"🕒 Last Update: {bond.last_update}")
        return "\n".join(output)

    def _run(self) -> str:
        """Run the bond monitor tool."""
        bonds = self._get_bond_data()
        
        if not bonds:
            return "❌ No bond data available at the moment."
        
        # Sort bonds by yield value for better visualization
        bonds.sort(key=lambda x: x.yield_value, reverse=True)
        
        # Get current time in São Paulo timezone
        sp_tz = pytz.timezone('America/Sao_Paulo')
        current_time = datetime.now(sp_tz).strftime("%m/%d/%Y %H:%M")
        
        output = [
            "📊 **GOVERNMENT BONDS MONITOR** 📊\n",
            f"🕒 Query Date: {current_time}",
            f"📈 Monitoring {len(bonds)} government bonds\n",
            "=== Bonds ordered by Yield (High to Low) ===\n"
        ]
        
        # Add all bonds to the output
        for bond in bonds:
            output.append(self._format_bond_output(bond))
            output.append("\n" + "=" * 40 + "\n")  # Separator
        
        return "\n".join(output)

if __name__ == '__main__':
    # Configuração de logging para testes
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Criar e executar a ferramenta
        tool = BondMonitorTool()
        
        # Executar e imprimir os resultados
        print("\n=== Teste do Monitor de Títulos Governamentais ===\n")
        results = tool._run()
        print(results)
        
    except Exception as e:
        logger.error(f"Erro durante a execução do teste: {str(e)}")
        raise
