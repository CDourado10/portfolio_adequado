from datetime import datetime
from typing import Optional, Type, List
import logging
from bs4 import BeautifulSoup
import requests
from pydantic import BaseModel, Field
import sys
import os
from crewai.tools import BaseTool

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Configure logging
logger = logging.getLogger(__name__)

class CurrencyData(BaseModel):
    """Model for currency data."""
    symbol: str
    price: float
    daily_change: float
    daily_change_pct: float
    weekly_change_pct: float
    monthly_change_pct: float
    ytd_change_pct: Optional[float]
    yoy_change_pct: Optional[float]
    last_update: str

class CurrencyMonitorInput(BaseModel):
    """Input schema for CurrencyMonitorTool."""
    pass

class CurrencyMonitorTool(BaseTool):
    """Tool to monitor currency rates from Trading Economics."""
    name: str = Field(default="CurrencyMonitorTool")
    description: str = Field(default=(
        "Gets currency rates from Trading Economics. "
        "Returns current rates and variations for major currency pairs."
    ))
    args_schema: Type[BaseModel] = CurrencyMonitorInput
    headers: dict = Field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    def _get_currency_data(self) -> List[CurrencyData]:
        """Get currency data from Trading Economics."""
        url = 'https://tradingeconomics.com/currencies'
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': lambda x: x and x.startswith('currency-')})
            
            if not table:
                logger.error("Currency table not found")
                return []
            
            currencies = []
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) < 9:
                        continue
                    
                    # Extract symbol from the link text
                    symbol_cell = cells[1].find('b')
                    if not symbol_cell:
                        continue
                    
                    symbol = symbol_cell.text.strip()
                    
                    # Extract numeric values
                    price = float(cells[2].text.strip())
                    daily_change = float(cells[3].text.strip().split()[-1])
                    daily_change_pct = float(cells[4].text.strip().rstrip('%'))
                    weekly_change_pct = float(cells[5].text.strip().rstrip('%'))
                    monthly_change_pct = float(cells[6].text.strip().rstrip('%'))
                    
                    # YTD and YoY are optional
                    try:
                        ytd_change_pct = float(cells[7].text.strip().rstrip('%'))
                    except (ValueError, AttributeError):
                        ytd_change_pct = None
                        
                    try:
                        yoy_change_pct = float(cells[8].text.strip().rstrip('%'))
                    except (ValueError, AttributeError):
                        yoy_change_pct = None
                    
                    last_update = cells[-1].text.strip()
                    
                    currency = CurrencyData(
                        symbol=symbol,
                        price=price,
                        daily_change=daily_change,
                        daily_change_pct=daily_change_pct,
                        weekly_change_pct=weekly_change_pct,
                        monthly_change_pct=monthly_change_pct,
                        ytd_change_pct=ytd_change_pct,
                        yoy_change_pct=yoy_change_pct,
                        last_update=last_update
                    )
                    currencies.append(currency)
                
                except Exception as e:
                    logger.error(f"Error parsing currency row: {str(e)}")
                    continue
            
            return currencies
            
        except Exception as e:
            logger.error(f"Error fetching currency data: {str(e)}")
            return []

    def _format_change(self, value: float) -> str:
        """Format change value with color and arrow."""
        if value > 0:
            return f"🟢 +{value:.2f}%"
        elif value < 0:
            return f"🔴 {value:.2f}%"
        return f"⚪ {value:.2f}%"

    def _format_currency_output(self, currency: CurrencyData) -> str:
        """Format currency data for output."""
        output = []
        
        # Symbol and current price
        output.append(f"💱 **{currency.symbol}**")
        output.append(f"💰 Price: {currency.price:.4f}")
        
        # Daily change
        output.append(f"📊 Daily: {self._format_change(currency.daily_change_pct)} ({currency.daily_change:+.4f})")
        
        # Other changes
        output.append(f"📈 Changes:")
        output.append(f"  • Weekly: {self._format_change(currency.weekly_change_pct)}")
        output.append(f"  • Monthly: {self._format_change(currency.monthly_change_pct)}")
        
        if currency.ytd_change_pct is not None:
            output.append(f"  • YTD: {self._format_change(currency.ytd_change_pct)}")
            
        if currency.yoy_change_pct is not None:
            output.append(f"  • Year: {self._format_change(currency.yoy_change_pct)}")
        
        output.append(f"🕒 Last Update: {currency.last_update}")
        output.append("")
        
        return "\n".join(output)

    def _run(self) -> str:
        """Run the tool."""
        try:
            logger.debug("Starting currency monitor tool execution")
            
            # Get currency data
            currencies = self._get_currency_data()
            
            if not currencies:
                return "❌ No currency data found."
            
            output = []
            output.append("💱 **CURRENCY MONITOR** 💱\n")
            
            # Add query timestamp
            output.append(f"🕒 Query Date: {datetime.now().strftime('%m/%d/%Y %H:%M')}")
            output.append(f"📊 Monitoring {len(currencies)} currency pairs\n")
            
            # Add currency information
            for currency in currencies:
                output.append(self._format_currency_output(currency))
            
            return "\n".join(output)
            
        except Exception as e:
            error_msg = f"Error monitoring currencies: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run tool
    tool = CurrencyMonitorTool()
    print(tool._run())
