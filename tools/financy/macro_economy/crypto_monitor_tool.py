import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
from pydantic import BaseModel, Field
from typing import List, Optional, Type, Dict
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

class CryptoData(BaseModel):
    """Data model for cryptocurrencies."""
    name: str
    price: float
    daily_change: float
    daily_change_percent: float
    weekly_change: float
    monthly_change: float
    ytd_change: float
    yearly_change: float
    market_cap: str
    last_update: str

class CryptoMonitorInput(BaseModel):
    """Input parameters for the CryptoMonitorTool."""
    pass

class CryptoMonitorTool(BaseTool):
    """Tool for monitoring cryptocurrency prices and variations."""
    name: str = "CryptoMonitorTool"
    description: str = (
        "Monitors and analyzes crypto prices from Trading Economics for major cryptocurrencies. "
        "Provides real-time data on prices, market caps, and various price changes over different timeframes. "
        "Data is presented with clear visual indicators: 🟢 for positive changes, 🔴 for negative changes, "
        "and ⚪ for no change."
    )
    args_schema: Type[BaseModel] = CryptoMonitorInput

    # Add url and headers as model fields
    url: str = Field(default="https://tradingeconomics.com/crypto")
    headers: Dict[str, str] = Field(default={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    
    def __init__(self, **kwargs):
        """Initialize the cryptocurrency monitoring tool."""
        super().__init__(**kwargs)

    def _get_crypto_data(self) -> List[CryptoData]:
        """Get cryptocurrency data from Trading Economics."""
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': lambda x: x and 'crypto-' in x})
            
            if not table:
                logger.error("Cryptocurrency table not found")
                return []
            
            crypto_data = []
            rows = table.find('tbody').find_all('tr')
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 10:
                    continue
                
                try:
                    name = cells[0].find('b').text.strip()
                    price = float(cells[1].text.strip().replace(',', ''))
                    daily_change = float(cells[2].text.strip().split()[0].replace(',', ''))
                    daily_change_percent = float(cells[3].text.strip().replace('%', '').replace(',', ''))
                    weekly_change = float(cells[4].text.strip().replace('%', '').replace(',', ''))
                    monthly_change = float(cells[5].text.strip().replace('%', '').replace(',', ''))
                    ytd_change = float(cells[6].text.strip().replace('%', '').replace(',', ''))
                    yearly_change = float(cells[7].text.strip().replace('%', '').replace(',', ''))
                    market_cap = cells[8].text.strip()
                    last_update = cells[9].text.strip()
                    
                    crypto_data.append(CryptoData(
                        name=name,
                        price=price,
                        daily_change=daily_change,
                        daily_change_percent=daily_change_percent,
                        weekly_change=weekly_change,
                        monthly_change=monthly_change,
                        ytd_change=ytd_change,
                        yearly_change=yearly_change,
                        market_cap=market_cap,
                        last_update=last_update
                    ))
                except Exception as e:
                    logger.error(f"Error processing cryptocurrency row: {str(e)}")
                    continue
            
            return crypto_data
            
        except Exception as e:
            logger.error(f"Error getting cryptocurrency data: {str(e)}")
            return []

    def _format_crypto_output(self, crypto: CryptoData) -> str:
        """Format the output for a cryptocurrency."""
        def get_change_emoji(value: float) -> str:
            if value > 0:
                return "🟢"
            elif value < 0:
                return "🔴"
            return "⚪"

        return (
            f"📊 **{crypto.name}**\n"
            f"💰 Price: ${crypto.price:,.2f}\n"
            f"📈 Daily: {get_change_emoji(crypto.daily_change_percent)} {crypto.daily_change:+.2f} ({crypto.daily_change_percent:+.2f}%)\n"
            f"📊 Changes:\n"
            f"  • Weekly: {get_change_emoji(crypto.weekly_change)} {crypto.weekly_change:+.2f}%\n"
            f"  • Monthly: {get_change_emoji(crypto.monthly_change)} {crypto.monthly_change:+.2f}%\n"
            f"  • YTD: {get_change_emoji(crypto.ytd_change)} {crypto.ytd_change:+.2f}%\n"
            f"  • Yearly: {get_change_emoji(crypto.yearly_change)} {crypto.yearly_change:+.2f}%\n"
            f"💎 Market Cap: {crypto.market_cap}\n"
            f"🕒 Last Update: {crypto.last_update}"
        )

    def _run(self, *args, **kwargs) -> str:
        """Run cryptocurrency monitoring."""
        cryptos = self._get_crypto_data()
        
        if not cryptos:
            return "❌ No cryptocurrency data available at the moment."
        
        # Sort by market cap (removing 'M' and '$' and converting to float)
        cryptos.sort(key=lambda x: float(x.market_cap.replace('$', '').replace('M', '').replace(',', '')), reverse=True)
        
        # Get current time in São Paulo
        sp_tz = pytz.timezone('America/Sao_Paulo')
        current_time = datetime.now(sp_tz).strftime("%m/%d/%Y %H:%M")
        
        output = [
            "📊 **CRYPTOCURRENCY MONITOR** 📊\n",
            f"🕒 Query Time: {current_time}",
            f"📈 Monitoring {len(cryptos)} cryptocurrencies\n",
            "=== Cryptocurrencies sorted by Market Cap ===\n"
        ]
        
        # Add all cryptocurrencies to output
        for crypto in cryptos:
            output.append(self._format_crypto_output(crypto))
            output.append("\n" + "=" * 40 + "\n")
        
        return "\n".join(output)

if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and run the tool
        tool = CryptoMonitorTool()
        
        # Run and print results
        print("\n=== Cryptocurrency Monitor Test ===\n")
        results = tool._run()
        print(results)
        
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}")
        raise
