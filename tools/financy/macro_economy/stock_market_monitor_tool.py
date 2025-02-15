from datetime import datetime
from typing import Optional, Type, List
import logging
from bs4 import BeautifulSoup
import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# Configure logging
logger = logging.getLogger(__name__)

class StockMarketData(BaseModel):
    """Model for stock market data."""
    symbol: str
    country: str
    price: float
    daily_change: float
    daily_change_pct: float
    weekly_change_pct: float
    monthly_change_pct: float
    ytd_change_pct: Optional[float]
    yoy_change_pct: Optional[float]
    last_update: str
    session_status: str

class StockMarketMonitorInput(BaseModel):
    """Input schema for StockMarketMonitorTool."""
    pass

class StockMarketMonitorTool(BaseTool):
    """Tool to monitor stock market indices from Trading Economics."""
    name: str = Field(default="StockMarketMonitorTool")
    description: str = Field(default=(
        "Gets stock market indices data from Trading Economics. "
        "Returns current values and variations for major stock indices."
    ))
    args_schema: Type[BaseModel] = StockMarketMonitorInput
    headers: dict = Field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    def _get_session_status(self, cell) -> str:
        """Get trading session status from clock icon."""
        clock = cell.find('span', class_='bi-clock')
        if not clock:
            return "Unknown"
        
        title = clock.get('title', '').lower()
        color = clock.get('style', '').split('color:')[-1].strip()
        
        if 'pre' in title:
            return "🟡 Pre-market"
        elif 'after' in title:
            return "🟡 After-hours"
        elif 'open' in title or color == 'green':
            return "🟢 Open"
        elif 'closed' in title or color == 'darkred':
            return "🔴 Closed"
        return "⚪ Unknown"

    def _get_stock_data(self) -> List[StockMarketData]:
        """Get stock market data from Trading Economics."""
        url = 'https://tradingeconomics.com/stocks'
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': lambda x: x and x.startswith('index-')})
            
            if not table:
                logger.error("Stock market table not found")
                return []
            
            indices = []
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) < 10:
                        continue
                    
                    # Get country from flag
                    flag = cells[0].find('div', class_='flag')
                    country = flag.get('class')[1].split('-')[1].upper() if flag else "Unknown"
                    
                    # Extract symbol from the link text
                    symbol_cell = cells[1].find('b')
                    if not symbol_cell:
                        continue
                    
                    symbol = symbol_cell.text.strip()
                    
                    # Extract numeric values
                    price = float(cells[2].text.strip().replace(',', ''))
                    daily_change = float(cells[3].text.strip().split()[-1].replace(',', ''))
                    daily_change_pct = float(cells[4].text.strip().rstrip('%').replace(',', ''))
                    weekly_change_pct = float(cells[5].text.strip().rstrip('%').replace(',', ''))
                    monthly_change_pct = float(cells[6].text.strip().rstrip('%').replace(',', ''))
                    
                    # YTD and YoY are optional
                    try:
                        ytd_change_pct = float(cells[7].text.strip().rstrip('%').replace(',', ''))
                    except (ValueError, AttributeError):
                        ytd_change_pct = None
                        
                    try:
                        yoy_change_pct = float(cells[8].text.strip().rstrip('%').replace(',', ''))
                    except (ValueError, AttributeError):
                        yoy_change_pct = None
                    
                    last_update = cells[9].text.strip()
                    session_status = self._get_session_status(cells[-1])
                    
                    index = StockMarketData(
                        symbol=symbol,
                        country=country,
                        price=price,
                        daily_change=daily_change,
                        daily_change_pct=daily_change_pct,
                        weekly_change_pct=weekly_change_pct,
                        monthly_change_pct=monthly_change_pct,
                        ytd_change_pct=ytd_change_pct,
                        yoy_change_pct=yoy_change_pct,
                        last_update=last_update,
                        session_status=session_status
                    )
                    indices.append(index)
                
                except Exception as e:
                    logger.error(f"Error parsing stock market row: {str(e)}")
                    continue
            
            return indices
            
        except Exception as e:
            logger.error(f"Error fetching stock market data: {str(e)}")
            return []

    def _format_change(self, value: float) -> str:
        """Format change value with color and arrow."""
        if value > 0:
            return f"🟢 +{value:.2f}%"
        elif value < 0:
            return f"🔴 {value:.2f}%"
        return f"⚪ {value:.2f}%"

    def _format_stock_output(self, index: StockMarketData) -> str:
        """Format stock market data for output."""
        output = []
        
        # Symbol and country
        output.append(f"📊 **{index.symbol}** ({index.country})")
        output.append(f"💰 Price: {index.price:,.2f}")
        
        # Daily change
        output.append(f"📈 Daily: {self._format_change(index.daily_change_pct)} ({index.daily_change:+,.2f})")
        
        # Other changes
        output.append(f"📊 Changes:")
        output.append(f"  • Weekly: {self._format_change(index.weekly_change_pct)}")
        output.append(f"  • Monthly: {self._format_change(index.monthly_change_pct)}")
        
        if index.ytd_change_pct is not None:
            output.append(f"  • YTD: {self._format_change(index.ytd_change_pct)}")
            
        if index.yoy_change_pct is not None:
            output.append(f"  • Year: {self._format_change(index.yoy_change_pct)}")
        
        # Session status and last update
        output.append(f"⏰ Status: {index.session_status}")
        output.append(f"🕒 Last Update: {index.last_update}")
        output.append("")
        
        return "\n".join(output)

    def _run(self) -> str:
        """Run the tool."""
        try:
            logger.debug("Starting stock market monitor tool execution")
            
            # Get stock market data
            indices = self._get_stock_data()
            
            if not indices:
                return "❌ No stock market data found."
            
            output = []
            output.append("📊 **STOCK MARKET MONITOR** 📊\n")
            
            # Add query timestamp
            output.append(f"🕒 Query Date: {datetime.now().strftime('%m/%d/%Y %H:%M')}")
            output.append(f"📈 Monitoring {len(indices)} major indices\n")
            
            # Add index information
            for index in indices:
                output.append(self._format_stock_output(index))
            
            return "\n".join(output)
            
        except Exception as e:
            error_msg = f"Error monitoring stock markets: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run tool
    tool = StockMarketMonitorTool()
    print(tool._run())
