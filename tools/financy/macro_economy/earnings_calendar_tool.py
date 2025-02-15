from crewai.tools import BaseTool
from typing import Type, Optional, List
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, date
import pandas as pd
import os
import time
import logging
from logging.handlers import RotatingFileHandler

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Format date for log file name
log_date = datetime.now().strftime('%Y%m%d')
log_file = os.path.join('logs', 'earnings_calendar', f'earnings_calendar_{log_date}.log')

# Ensure log directory exists
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# File handler with rotation (max 5MB per file, keep 5 backups)
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=5*1024*1024,  # 5MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler (show only INFO or higher)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# Remove existing handlers
logger.handlers = []

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Set other loggers to WARNING level
for log_name in ['urllib3', 'selenium', 'WDM']:
    logging.getLogger(log_name).setLevel(logging.WARNING)

class EarningsCalendarInput(BaseModel):
    """Input schema for EarningsCalendarTool."""
    pass

class EarningsCalendarTool(BaseTool):
    """Tool to get earnings calendar data from Investing.com."""
    
    name: str = Field(default="EarningsCalendarTool")
    description: str = Field(default=(
        "Gets earnings calendar data from Investing.com. "
        "Returns earnings reports for the top 10 companies by market cap, "
        "including EPS, Revenue, and Market Cap information."
    ))
    args_schema: Type[BaseModel] = EarningsCalendarInput
    headers: dict = Field(default_factory=lambda: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Accept-Encoding": "gzip, deflate, br",
    })
    base_url: str = Field(default='https://www.investing.com/earnings-calendar/')
    max_retries: int = Field(default=3)
    retry_delay: int = Field(default=2)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.headers = self.headers
        self.base_url = self.base_url
        self.max_retries = self.max_retries
        self.retry_delay = self.retry_delay

    def _setup_driver(self):
        """Configure Firefox driver."""
        options = FirefoxOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(f'user-agent={self.headers["User-Agent"]}')
        
        # Use local GeckoDriver
        service = FirefoxService("C:\\geckodriver\\geckodriver.exe")
        
        return webdriver.Firefox(service=service, options=options)

    def _get_earnings_data(self) -> List[BeautifulSoup]:
        """Get earnings calendar data."""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            driver = None
            try:
                logger.debug(f"Attempt {attempt} of {max_retries} to get earnings data")
                
                driver = self._setup_driver()
                url = self.base_url
                logger.debug(f"Accessing URL: {url}")
                
                driver.get(url)
                time.sleep(5)  # Wait for initial load
                
                # List to store results
                results = []
                
                # Wait for the filter container to be present
                filter_container = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "js-tabs-earnings"))
                )
                
                # Timeframes to query
                timeframes = ['timeFrame_thisWeek', 'timeFrame_nextWeek']
                
                for timeframe in timeframes:
                    try:
                        logger.debug(f"Processing timeframe: {timeframe}")
                        
                        # Wait for the specific timeframe button and click it
                        timeframe_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.ID, timeframe))
                        )
                        
                        # Scroll the button into view to ensure it's clickable
                        driver.execute_script("arguments[0].scrollIntoView(true);", timeframe_button)
                        time.sleep(1)  # Small pause after scroll
                        
                        # Click using JavaScript to avoid any potential overlay issues
                        driver.execute_script("arguments[0].click();", timeframe_button)
                        
                        # Wait for table load/update
                        time.sleep(3)
                        
                        # Wait for table to be present and visible
                        table = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "genTbl"))
                        )
                        
                        # Get current page HTML after table is loaded
                        html_content = driver.page_source
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Verify if we got the table with data
                        earnings_table = soup.select_one('table.genTbl')
                        if not earnings_table:
                            raise Exception(f"Earnings table not found for {timeframe}")
                        
                        # Add to results if table is found
                        results.append(soup)
                        logger.debug(f"Data successfully obtained for {timeframe}")
                        
                    except Exception as e:
                        logger.error(f"Error processing timeframe {timeframe}: {str(e)}")
                        # Continue to next timeframe if one fails
                        continue
                
                driver.quit()
                
                if not results:
                    raise Exception("No earnings data was obtained from any timeframe")
                
                return results
                
            except Exception as e:
                logger.error(f"Error getting earnings data: {str(e)}")
                if driver:
                    driver.quit()
                if attempt == max_retries:
                    raise
                time.sleep(2)  # Wait before retrying

    def _parse_market_cap(self, market_cap_str: str) -> float:
        """Parse market cap string to float value in billions."""
        try:
            if not market_cap_str or market_cap_str == '--':
                return 0.0
            
            value = float(market_cap_str.replace('B', '').replace('M', '').replace('K', ''))
            
            if 'B' in market_cap_str:
                return value
            elif 'M' in market_cap_str:
                return value / 1000
            elif 'K' in market_cap_str:
                return value / 1000000
            else:
                return value / 1000000000
        except:
            return 0.0

    def _parse_numeric_value(self, value_str: str) -> Optional[float]:
        """Parse numeric value from string."""
        try:
            if not value_str or value_str == '--' or value_str.strip() == '':
                return None
            
            # Remove B/M/K and any other non-numeric characters except dots and minus
            clean_str = ''.join(c for c in value_str if c.isdigit() or c in '.-')
            if not clean_str:
                return None
                
            value = float(clean_str)
            
            # Apply multiplier based on suffix
            if 'B' in value_str:
                value *= 1000000000
            elif 'M' in value_str:
                value *= 1000000
            elif 'K' in value_str:
                value *= 1000
                
            return value
        except (ValueError, TypeError):
            return None

    def _calculate_surprise(self, actual: Optional[float], forecast: Optional[float]) -> Optional[float]:
        """Calculate surprise percentage safely."""
        try:
            if actual is None or forecast is None or forecast == 0:
                return None
            return ((actual - forecast) / abs(forecast)) * 100
        except (ZeroDivisionError, TypeError):
            return None

    def _parse_earnings_row(self, row) -> Optional[dict]:
        """Parse an earnings row from the table."""
        try:
            # Extract country
            country_elem = row.select_one('td.flag span[title]')
            country = country_elem.get('title', '') if country_elem else 'Unknown'
            
            # Extract company info
            company_elem = row.select_one('td.earnCalCompany')
            if not company_elem:
                return None
                
            company_name = company_elem.select_one('span.earnCalCompanyName')
            company = company_name.text.strip() if company_name else ''
            
            ticker_elem = company_elem.select_one('a.bold')
            ticker = ticker_elem.text.strip() if ticker_elem else ''
            
            if not company or not ticker:
                logger.warning(f"Missing company name or ticker: {company} ({ticker})")
                return None
            
            # Extract all td.leftStrong elements for both EPS and Revenue forecasts
            forecast_elements = row.select('td.leftStrong')
            
            # Extract EPS values
            eps_actual_elem = row.select_one('td[class*="pid-"][class*="-eps_actual"]')
            eps_forecast_elem = forecast_elements[0] if len(forecast_elements) > 0 else None
            
            eps_actual = self._parse_numeric_value(eps_actual_elem.text.strip() if eps_actual_elem else None)
            eps_forecast = self._parse_numeric_value(eps_forecast_elem.text.replace('/', '').strip() if eps_forecast_elem else None)
            
            # Extract Revenue values (next set of actual/forecast)
            revenue_actual_elem = row.select_one('td[class*="pid-"][class*="-rev_actual"]')
            revenue_forecast_elem = forecast_elements[1] if len(forecast_elements) > 1 else None
            
            revenue_actual = self._parse_numeric_value(revenue_actual_elem.text.strip() if revenue_actual_elem else None)
            revenue_forecast = self._parse_numeric_value(revenue_forecast_elem.text.replace('/', '').strip() if revenue_forecast_elem else None)
            
            # Log revenue values for debugging
            if revenue_actual is not None or revenue_forecast is not None:
                logger.debug(f"Revenue data for {company} ({ticker}):")
                logger.debug(f"  Actual: {revenue_actual}")
                logger.debug(f"  Forecast: {revenue_forecast}")
                if revenue_actual_elem:
                    logger.debug(f"  Raw actual text: {revenue_actual_elem.text.strip()}")
                if revenue_forecast_elem:
                    logger.debug(f"  Raw forecast text: {revenue_forecast_elem.text.strip()}")
            
            # Extract Market Cap
            market_cap_elem = row.select_one('td.right:not(.time)')
            market_cap = self._parse_market_cap(market_cap_elem.text.strip() if market_cap_elem else '')
            
            # Extract announcement time
            time_elem = row.select_one('td.time span.marketOpen')
            announcement_time = time_elem.get('data-tooltip', '') if time_elem else ''
            
            # Calculate surprises using safe calculation method
            eps_surprise = self._calculate_surprise(eps_actual, eps_forecast)
            revenue_surprise = self._calculate_surprise(revenue_actual, revenue_forecast)
            
            return {
                'company': company,
                'ticker': ticker,
                'country': country,
                'eps_actual': eps_actual,
                'eps_forecast': eps_forecast,
                'eps_surprise': eps_surprise,
                'revenue_actual': revenue_actual,
                'revenue_forecast': revenue_forecast,
                'revenue_surprise': revenue_surprise,
                'market_cap': market_cap,
                'announcement_time': announcement_time
            }
            
        except Exception as e:
            logger.error(f"Error processing earnings row: {str(e)}")
            return None

    def _format_earnings_output(self, earnings_data: dict) -> str:
        """Format earnings output."""
        try:
            output = []
            
            # Company info with market cap and country
            market_cap_str = f"{earnings_data['market_cap']:.1f}B" if earnings_data.get('market_cap') else 'N/A'
            output.append(f"🏢 **{earnings_data['company']} ({earnings_data['ticker']})** - {earnings_data['country']}")
            output.append(f"💰 Market Cap: ${market_cap_str}")
            
            # Announcement time
            if earnings_data.get('announcement_time'):
                output.append(f"⏰ {earnings_data['announcement_time']}")
            
            # EPS information
            eps_actual = earnings_data.get('eps_actual')
            eps_forecast = earnings_data.get('eps_forecast')
            eps_surprise = earnings_data.get('eps_surprise')
            
            if eps_actual is not None or eps_forecast is not None:
                eps_actual_str = f"${eps_actual:.2f}" if eps_actual is not None else "N/A"
                eps_forecast_str = f"${eps_forecast:.2f}" if eps_forecast is not None else "N/A"
                eps_surprise_str = f" ({eps_surprise:+.1f}%)" if eps_surprise is not None else ""
                
                # Determine color based on surprise (if available)
                eps_color = "🟢" if eps_surprise and eps_surprise > 0 else "🔴" if eps_surprise and eps_surprise < 0 else "⚪"
                
                output.append(f"{eps_color} EPS: {eps_actual_str} vs Forecast: {eps_forecast_str}{eps_surprise_str}")
            
            # Revenue information
            rev_actual = earnings_data.get('revenue_actual')
            rev_forecast = earnings_data.get('revenue_forecast')
            rev_surprise = earnings_data.get('revenue_surprise')
            
            if rev_actual is not None or rev_forecast is not None:
                rev_actual_str = f"${rev_actual/1e9:.2f}B" if rev_actual is not None else "N/A"
                rev_forecast_str = f"${rev_forecast/1e9:.2f}B" if rev_forecast is not None else "N/A"
                rev_surprise_str = f" ({rev_surprise:+.1f}%)" if rev_surprise is not None else ""
                
                # Determine color based on surprise (if available)
                rev_color = "🟢" if rev_surprise and rev_surprise > 0 else "🔴" if rev_surprise and rev_surprise < 0 else "⚪"
                
                output.append(f"{rev_color} Revenue: {rev_actual_str} vs Forecast: {rev_forecast_str}{rev_surprise_str}")
            
            return "\n".join(output) + "\n"
            
        except Exception as e:
            logger.error(f"Error formatting earnings output: {str(e)}")
            # Return basic information even in case of error
            try:
                market_cap_str = f"{earnings_data['market_cap']:.1f}B" if earnings_data.get('market_cap') else 'N/A'
                return (f"🏢 **{earnings_data['company']} ({earnings_data['ticker']})** - {earnings_data['country']}\n"
                       f"💰 Market Cap: ${market_cap_str}\n")
            except:
                return f"Error formatting data for {earnings_data.get('company', 'Unknown Company')}\n"

    def _run(self) -> str:
        """Run the tool."""
        try:
            logger.debug("Starting earnings calendar tool execution")
            
            # Get earnings data
            earnings_data_list = self._get_earnings_data()
            
            # List to store all earnings
            all_earnings = []
            
            # Process each earnings dataset
            for earnings_data in earnings_data_list:
                # Find all earnings rows
                earnings_rows = earnings_data.select('table.genTbl tr:not(.head)')
                
                # Process each row
                for row in earnings_rows:
                    earnings_info = self._parse_earnings_row(row)
                    if earnings_info:
                        all_earnings.append(earnings_info)
            
            if not all_earnings:
                return "❌ No earnings data found."
            
            # Sort earnings by market cap (descending) and get top 10
            all_earnings.sort(key=lambda x: x['market_cap'] if x['market_cap'] else 0, reverse=True)
            top_earnings = all_earnings[:10]
            
            output = []
            output.append("📅 **EARNINGS CALENDAR** 📅\n")
            
            # Add query timestamp
            output.append(f"🕒 Query Date: {datetime.now().strftime('%m/%d/%Y %H:%M')}\n")
            
            # Add total number of companies found
            output.append(f"📊 Total companies found: {len(all_earnings)}")
            output.append(f"📈 Showing top 10 companies by market cap\n")
            
            # Add earnings information for top 10
            for i, earnings in enumerate(top_earnings, 1):
                output.append(f"#{i}")
                output.append(self._format_earnings_output(earnings))
            
            return "\n".join(output)
            
        except Exception as e:
            error_msg = f"Error processing earnings calendar: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = EarningsCalendarTool()
    print(tool._run())
