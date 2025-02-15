import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Type
from enum import Enum
import time
import os
from crewai.tools import BaseTool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
log_file = os.path.join('logs', 'macro_monitor.log')
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

class Region(str, Enum):
    """Available regions for monitoring."""
    USA = "united-states"
    EURO = "euro-area"
    CHINA = "china"

class MacroIndicator(BaseModel):
    """Data model for macroeconomic indicators."""
    name: str
    last: str
    previous: str
    highest: str
    lowest: str
    unit: str
    last_update: str

class MacroMonitorInput(BaseModel):
    """Input parameters for the MacroMonitorTool."""
    pass

class MacroMonitorTool(BaseTool):
    """Tool for monitoring macroeconomic indicators."""
    name: str = "MacroMonitorTool"
    description: str = (
        "Monitors and analyzes macroeconomic indicators from Trading Economics for major economies. "
        "Provides real-time data on indicators such as inflation, GDP, unemployment, and economic activity. "
        "Data is presented with clear visual indicators: 🟢 for positive changes, 🔴 for negative changes, "
        "and ⚪ for no change."
    )
    args_schema: Type[BaseModel] = MacroMonitorInput

    # Adicionar campos base_url e headers como atributos do modelo
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
    
    def __init__(self, **kwargs):
        """Initialize the macro monitoring tool."""
        super().__init__(**kwargs)

    def _get_macro_data(self, region: Region) -> List[MacroIndicator]:
        """Get macroeconomic data for a specific region."""
        try:
            url = self.base_url.format(region.value)
            logger.info(f"Searching for data for {region} at {url}")
            
            # Add a delay between requests
            time.sleep(1)
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the main indicators table
            table = soup.find('table', {'class': 'table'})
            
            if not table:
                logger.error(f"Indicators table not found for {region}")
                return []
            
            indicators = []
            rows = table.find_all('tr')[1:]  # Skip the header
            
            for row in rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) < 7:
                        continue

                    # Extract the indicator name
                    name_cell = cells[0].find('a')
                    if not name_cell:
                        continue
                    name = name_cell.text.strip()

                    # Process the values
                    last = cells[1].text.strip().replace('te-value-negative', '').strip()
                    previous = cells[2].text.strip().replace('te-value-negative', '').strip()
                    highest = cells[3].text.strip().replace('te-value-negative', '').strip()
                    lowest = cells[4].text.strip().replace('te-value-negative', '').strip()
                    unit = cells[5].text.strip()
                    last_update = cells[6].text.strip()

                    logger.info(f"Processing indicator: {name}")

                    indicators.append(MacroIndicator(
                        name=name,
                        last=last,
                        previous=previous,
                        highest=highest,
                        lowest=lowest,
                        unit=unit,
                        last_update=last_update
                    ))
                except Exception as e:
                    logger.error(f"Error processing indicator: {str(e)}")
                    continue
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error getting macro data for {region}: {str(e)}")
            return []

    def _format_indicator_output(self, indicator: MacroIndicator) -> str:
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
            f"📊 **{indicator.name}**\n"
            f"📈 Current: {get_change_emoji(indicator.last, indicator.previous)} {indicator.last} {indicator.unit}\n"
            f"⏮️ Previous: {indicator.previous} {indicator.unit}\n"
            f"📊 History:\n"
            f"  • Maximum: {indicator.highest} {indicator.unit}\n"
            f"  • Minimum: {indicator.lowest} {indicator.unit}\n"
            f"🕒 Last Update: {indicator.last_update}"
        )

    def _format_region_name(self, region: Region) -> str:
        """Format region name for display."""
        region_names = {
            Region.USA: "United States",
            Region.EURO: "Euro Area",
            Region.CHINA: "China"
        }
        return region_names.get(region, str(region))

    def _run(self, *args, **kwargs) -> str:
        """Run macroeconomic indicators monitoring."""
        # Get current time in São Paulo
        sp_tz = pytz.timezone('America/Sao_Paulo')
        current_time = datetime.now(sp_tz).strftime("%m/%d/%Y %H:%M")
        
        output = [
            "🌍 **MACROECONOMIC INDICATORS MONITOR** 🌍\n",
            f"🕒 Query Time: {current_time}\n"
        ]

        # Process each region
        for region in Region:
            indicators = self._get_macro_data(region)
            if not indicators:
                output.append(f"\n❌ Data not available for {self._format_region_name(region)}")
                continue

            output.extend([
                f"\n{'=' * 50}",
                f"🏛️ **{self._format_region_name(region)}**",
                f"📊 {len(indicators)} indicators monitored",
                f"{'=' * 50}\n"
            ])

            # Add region indicators
            for indicator in indicators:
                output.append(self._format_indicator_output(indicator))
                output.append("\n" + "-" * 40 + "\n")

        return "\n".join(output)

if __name__ == '__main__':
    try:
        # Create and run the tool
        tool = MacroMonitorTool()
        
        # Run and print results
        results = tool._run()
        print(results)
        
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}")
        raise
