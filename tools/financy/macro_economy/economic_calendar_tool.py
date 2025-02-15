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
log_file = os.path.join('logs', 'economic_calendar', f'economic_calendar_{log_date}.log')

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

class EconomicCalendarInput(BaseModel):
    """Input schema for EconomicCalendarTool."""

class EconomicCalendarTool(BaseTool):
    """Tool to get economic calendar data from Investing.com."""
    
    name: str = Field(default="EconomicCalendarTool")
    description: str = Field(default=(
        "Gets economic calendar data from Investing.com. "
        "Returns high-importance economic events from last week and next week for all countries. "
        "Returns detailed information including date/time, country, event, importance, actual, forecast, and previous."
    ))
    args_schema: Type[BaseModel] = EconomicCalendarInput
    headers: dict = Field(default_factory=lambda: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Accept-Encoding": "gzip, deflate, br",
    })
    base_url: str = Field(default='https://investing.com/economic-calendar/')
    CALENDAR_URL: str = Field(default='https://investing.com/economic-calendar/')
    max_retries: int = Field(default=3)
    retry_delay: int = Field(default=2)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.headers = self.headers
        self.base_url = self.base_url
        self.CALENDAR_URL = self.CALENDAR_URL
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

    def _get_calendar_data(self) -> List[BeautifulSoup]:
        """Get economic calendar data."""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt} of {max_retries} to get calendar data")
                
                driver = self._setup_driver()
                url = self.base_url
                logger.debug(f"Accessing URL: {url}")
                
                driver.get(url)
                time.sleep(5)  # Wait for initial load
                
                # List to store results
                results = []
                
                # Timeframes to query
                timeframes = ['timeFrame_thisWeek', 'timeFrame_nextWeek']
                
                for timeframe in timeframes:
                    try:
                        # Click timeframe button
                        timeframe_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.ID, timeframe))
                        )
                        timeframe_button.click()
                        time.sleep(3)  # Wait for load
                        
                        # Wait for table to load
                        calendar_element = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.ID, "economicCalendarData"))
                        )
                        
                        # Get current page HTML
                        html_content = driver.page_source
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Check if table was found
                        if not soup.select('#economicCalendarData'):
                            raise Exception(f"Calendar table not found for {timeframe}")
                        
                        results.append(soup)
                        logger.debug(f"Data successfully obtained for {timeframe}")
                        
                    except Exception as e:
                        logger.error(f"Error getting data for {timeframe}: {str(e)}")
                
                driver.quit()
                
                if not results:
                    raise Exception("No calendar data was obtained")
                
                return results
                
            except Exception as e:
                logger.error(f"Error getting calendar data: {str(e)}")
                if driver:
                    driver.quit()
                if attempt == max_retries:
                    raise
                time.sleep(2)  # Wait before retrying

    def _parse_event_row(self, row) -> Optional[dict]:
        """Parse an event row from the table."""
        try:
            # Extract event date and time
            datetime_str = row.get('data-event-datetime')
            if not datetime_str:
                return None
            
            event_datetime = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')
            
            # Extract country
            country_elem = row.select_one('td.flagCur span[title]')
            if not country_elem:
                return None
            country = country_elem.get('title', '').strip()
            
            # Extract event (including period in parentheses)
            event_elem = row.select_one('td.event a')
            if not event_elem:
                return None
            event = event_elem.text.strip()
            
            # Extract importance based on title
            importance_elem = row.select_one('td.left.textNum.sentiment')
            importance = 'low'  # default value
            if importance_elem:
                title = importance_elem.get('title', '').lower()
                if 'high volatility' in title:
                    importance = 'high'
                elif 'moderate volatility' in title:
                    importance = 'medium'
                elif 'low volatility' in title:
                    importance = 'low'
            
            # Extract values using specific classes
            actual = row.select_one('td.act')
            forecast = row.select_one('td.fore')
            previous = row.select_one('td.prev')
            
            def clean_value(elem):
                if not elem:
                    return None
                span = elem.select_one('span[title]')
                if span:
                    value = span.text.strip()
                else:
                    value = elem.text.strip()
                return value if value and value != '&nbsp;' and value != '--' else None
            
            return {
                'datetime': event_datetime,
                'country': country,
                'event': event,
                'importance': importance,
                'actual': clean_value(actual),
                'forecast': clean_value(forecast),
                'previous': clean_value(previous)
            }
            
        except Exception as e:
            logger.error(f"Error processing event row: {str(e)}")
            return None

    def _format_event_output(self, event_data: dict) -> str:
        """Format event output."""
        output = []
        output.append(f"🔴 **{event_data['datetime'].strftime('%d/%m %H:%M')} - {event_data['country']}**")
        output.append(f"📊 Event: {event_data['event']}")
        
        importance_emoji = "🔥" if event_data['importance'] == 'high' else \
                         "⚡" if event_data['importance'] == 'medium' else "ℹ️"
        output.append(f"{importance_emoji} Importance: {event_data['importance'].title()}")
        
        if event_data['actual']:
            output.append(f"📈 Actual: {event_data['actual']}")
        if event_data['forecast']:
            output.append(f"🎯 Forecast: {event_data['forecast']}")
        if event_data['previous']:
            output.append(f"📊 Previous: {event_data['previous']}")
        
        return "\n".join(output) + "\n"

    def _run(self) -> str:
        """Run the tool."""
        try:
            logger.debug("Starting economic calendar tool execution")
            
            # Get calendar data
            calendar_data_list = self._get_calendar_data()
            
            # List to store all events
            all_events = []
            
            # Process each calendar dataset
            for calendar_data in calendar_data_list:
                # Find all event rows
                event_rows = calendar_data.select('#economicCalendarData tr.js-event-item')
                
                # Process each row
                for row in event_rows:
                    event_data = self._parse_event_row(row)
                    if event_data:
                        # Filter only high importance events
                        if event_data['importance'] == 'high':
                            all_events.append(event_data)
            
            if not all_events:
                return "❌ No events found with specified filters."
            
            # Sort events by date and time
            all_events.sort(key=lambda x: x['datetime'])
            
            # Determine actual event period
            first_event_date = all_events[0]['datetime'].date()
            last_event_date = all_events[-1]['datetime'].date()
            
            output = []
            output.append("📅 **ECONOMIC CALENDAR** 📅\n")
            
            # Query period
            output.append("⏰ **Query Period**")
            output.append(f"📅 From: {first_event_date.strftime('%d/%m/%Y')}")
            output.append(f"📅 To: {last_event_date.strftime('%d/%m/%Y')}")
            output.append(f"🕒 Query Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            
            # Events by date
            current_date = None
            for event in all_events:
                event_date = event['datetime'].date()
                if event_date != current_date:
                    current_date = event_date
                    output.append(f"\n📆 **{current_date.strftime('%d/%m/%Y')}**")
                
                output.append(self._format_event_output(event))
            
            return "\n".join(output)
            
        except Exception as e:
            error_msg = f"Error processing economic calendar: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"

if __name__ == '__main__':
    tool = EconomicCalendarTool()
    print(tool._run())
