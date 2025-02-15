from datetime import datetime
from typing import Optional, Type, List
import logging
from bs4 import BeautifulSoup
import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import re

# Configure logging
logger = logging.getLogger(__name__)

class CommodityData(BaseModel):
    """Model for commodity data."""
    name: str
    sector: str
    unit: str
    price: float
    daily_change: float
    daily_change_pct: float
    weekly_change_pct: float
    monthly_change_pct: float
    ytd_change_pct: Optional[float]
    yoy_change_pct: Optional[float]
    last_update: str

class CommodityMonitorInput(BaseModel):
    """Input parameters for the CommodityMonitorTool."""
    pass

class CommodityMonitorTool(BaseTool):
    """Tool to monitor commodity prices from Trading Economics."""
    name: str = Field(default="CommodityMonitorTool")
    description: str = Field(default=(
        "Monitors and analyzes commodity markets from Trading Economics across multiple sectors "
        "(Energy, Metals, Agricultural, Industrial, Livestock, Index, Electricity). "
        "Provides real-time highlights showing the best and worst performing commodities in each sector, "
        "including current prices, daily changes, and variations over different timeframes "
        "(weekly, monthly, YTD, and yearly). Data is presented with clear visual indicators: "
        "🟢 for positive changes, 🔴 for negative changes, and ⚪ for no change."
    ))
    args_schema: Type[BaseModel] = CommodityMonitorInput
    headers: dict = Field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    COMMODITIES_URL: str = 'https://tradingeconomics.com/commodities'

    def _parse_change(self, text: str) -> float:
        """Parse change value from text."""
        match = re.match(r'([+-]?\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        else:
            return 0.0

    def _parse_percentage(self, text: str) -> float:
        """Parse percentage value from text."""
        match = re.match(r'([+-]?\d+(?:\.\d+)?)%', text)
        if match:
            return float(match.group(1))
        else:
            return 0.0

    def _get_commodity_data(self) -> List[CommodityData]:
        """Get commodity data from Trading Economics."""
        try:
            response = requests.get(self.COMMODITIES_URL, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', {'id': lambda x: x and x.startswith('commodity')})
            
            if not tables:
                logger.error("Commodity tables not found")
                return []
            
            commodities = []
            
            for table in tables:
                try:
                    # Get sector from table header
                    header = table.find('th', class_='te-sort')
                    if not header:
                        continue
                        
                    sector = header.text.strip()
                    rows = table.find_all('tr')[1:]  # Skip header row
                    
                    for row in rows:
                        try:
                            cells = row.find_all('td')
                            if len(cells) < 9:
                                continue
                            
                            # Extract commodity name and unit
                            name_cell = cells[0]
                            name = name_cell.find('b').text.strip()
                            unit = name_cell.find('div').text.strip() if name_cell.find('div') else ""
                            
                            # Extract numeric values
                            price = float(cells[1].text.strip().replace(',', ''))
                            daily_change = float(cells[2].text.strip().split()[-1].replace(',', ''))
                            daily_change_pct = float(cells[3].text.strip().rstrip('%').replace(',', ''))
                            weekly_change_pct = float(cells[4].text.strip().rstrip('%').replace(',', ''))
                            monthly_change_pct = float(cells[5].text.strip().rstrip('%').replace(',', ''))
                            
                            # YTD and YoY are optional
                            try:
                                ytd_change_pct = float(cells[6].text.strip().rstrip('%').replace(',', ''))
                            except (ValueError, AttributeError):
                                ytd_change_pct = None
                                
                            try:
                                yoy_change_pct = float(cells[7].text.strip().rstrip('%').replace(',', ''))
                            except (ValueError, AttributeError):
                                yoy_change_pct = None
                            
                            last_update = cells[8].text.strip()
                            
                            commodity = CommodityData(
                                name=name,
                                sector=sector,
                                unit=unit,
                                price=price,
                                daily_change=daily_change,
                                daily_change_pct=daily_change_pct,
                                weekly_change_pct=weekly_change_pct,
                                monthly_change_pct=monthly_change_pct,
                                ytd_change_pct=ytd_change_pct,
                                yoy_change_pct=yoy_change_pct,
                                last_update=last_update
                            )
                            commodities.append(commodity)
                            
                        except Exception as e:
                            logger.error(f"Error parsing commodity row: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error parsing commodity table: {str(e)}")
                    continue
            
            return commodities
            
        except Exception as e:
            logger.error(f"Error fetching commodity data: {str(e)}")
            return []

    def _format_commodity_output(self, commodity: CommodityData) -> str:
        """Format the output for a specific commodity."""
        # Determine emojis based on variations
        daily_emoji = "🟢" if commodity.daily_change_pct > 0 else "🔴" if commodity.daily_change_pct < 0 else "⚪"
        weekly_emoji = "🟢" if commodity.weekly_change_pct > 0 else "🔴" if commodity.weekly_change_pct < 0 else "⚪"
        monthly_emoji = "🟢" if commodity.monthly_change_pct > 0 else "🔴" if commodity.monthly_change_pct < 0 else "⚪"
        ytd_emoji = "🟢" if commodity.ytd_change_pct > 0 else "🔴" if commodity.ytd_change_pct < 0 else "⚪"
        year_emoji = "🟢" if commodity.yoy_change_pct > 0 else "🔴" if commodity.yoy_change_pct < 0 else "⚪"

        return f"""📊 **{commodity.name}** ({commodity.sector})
💰 Price: {commodity.price} {commodity.unit}
📈 Daily: {daily_emoji} {commodity.daily_change_pct:.2f}% ({'+' if commodity.daily_change >= 0 else ''}{commodity.daily_change:.4f})
📊 Changes:
  • Weekly: {weekly_emoji} {'+' if commodity.weekly_change_pct >= 0 else ''}{commodity.weekly_change_pct:.2f}%
  • Monthly: {monthly_emoji} {'+' if commodity.monthly_change_pct >= 0 else ''}{commodity.monthly_change_pct:.2f}%
  • YTD: {ytd_emoji} {'+' if commodity.ytd_change_pct >= 0 else ''}{commodity.ytd_change_pct:.2f}%
  • Year: {year_emoji} {'+' if commodity.yoy_change_pct >= 0 else ''}{commodity.yoy_change_pct:.2f}%
🕒 Last Update: {commodity.last_update}"""

    def _run(self) -> str:
        """Execute the tool and return formatted data."""
        try:
            commodities = self._get_commodity_data()
            
            # Group commodities by sector
            sector_commodities = {}
            for commodity in commodities:
                if commodity.sector not in sector_commodities:
                    sector_commodities[commodity.sector] = []
                sector_commodities[commodity.sector].append(commodity)

            # Format output
            output = []
            output.append("📊 **COMMODITY MONITOR - HIGHLIGHTS** 📊\n")
            output.append(f"🕒 Query Date: {datetime.now().strftime('%m/%d/%Y %H:%M')}")
            output.append(f"📈 Monitoring {len(commodities)} commodities in {len(sector_commodities)} sectors\n")

            # For each sector, find the best and worst performance based on yearly change
            for sector, commodities_list in sector_commodities.items():
                output.append(f"🏭 **{sector}**")
                output.append("=" * 10)
                
                # Sort by yearly performance
                sorted_commodities = sorted(commodities_list, key=lambda x: x.yoy_change_pct, reverse=True)
                
                # Best performance
                best = sorted_commodities[0]
                output.append("🌟 **BEST PERFORMANCE**")
                output.append(self._format_commodity_output(best))
                
                # Worst performance
                worst = sorted_commodities[-1]
                output.append("\n💫 **WORST PERFORMANCE**")
                output.append(self._format_commodity_output(worst))
                output.append("")

            return "\n".join(output)

        except Exception as e:
            logging.error(f"Error executing CommodityMonitorTool: {str(e)}")
            return f"❌ Error getting commodity data: {str(e)}"

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run tool
    tool = CommodityMonitorTool()
    print(tool._run())
