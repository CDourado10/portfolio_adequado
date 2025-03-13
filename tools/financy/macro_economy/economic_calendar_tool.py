from langchain_openai import AzureChatOpenAI
from browser_use import Agent, BrowserConfig, Browser
from dotenv import load_dotenv
import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel
import asyncio

load_dotenv()

class EconomicCalendarToolInput(BaseModel):
    pass

class EconomicCalendarTool(BaseTool):
    name: str = "EconomicCalendarTool"
    description: str = (
        "Executes an AI agent to retrieve relevant events from next week's economic calendar."
    )
    args_schema: Type[BaseModel] = EconomicCalendarToolInput

    def _run(self) -> str:
            async def main():

                config = BrowserConfig(
                    headless=True,
                    disable_security=True
                )

                browser = Browser(config=config)
                initial_actions = [
                    {'open_tab': {'url': 'https://tradingeconomics.com/calendar'}}
                ]

                agent = Agent(
                    task="Select the 10 most impactful events for the financial market in the next 7 days.",
                    initial_actions=initial_actions,
                    llm=AzureChatOpenAI(model="gpt-4o-mini", api_version="2024-10-21", api_key=os.getenv("AZURE_API_KEY"), azure_endpoint=os.getenv("AZURE_API_BASE")),
                    #planner_llm=AzureChatOpenAI(model="gpt-4o-mini", api_version="2024-10-21", api_key=os.getenv("AZURE_API_KEY"), azure_endpoint=os.getenv("AZURE_API_BASE")),
                    browser=browser,
                    #use_vision=True,
                    save_conversation_path="logs/web_interaction"
                )
                result = await agent.run()
                await browser.close()
                return result

            result = asyncio.run(main())
            resultado_final = result.final_result()
            return resultado_final

if __name__ == "__main__":
    tool = EconomicCalendarTool()
    result = tool._run()
    print("Result:")
    print(result)
