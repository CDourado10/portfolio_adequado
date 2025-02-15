from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd

class PortfolioOptimizerInput(BaseModel):
    """Input schema for PortfolioOptimizer tool."""
    symbols: str = Field(..., description="List of asset symbols to be optimized (comma-separated), in yfinance format. For example: 'AAPL, BTC-USD, ^GSPC, COCO.L'")
    period: str = Field(
        default="1 year",
        description="Analysis period (1 day, 5 days, 1 month, 3 months, 6 months, 1 year, 2 years, 5 years, 10 years, ytd, max)"
    )
    rebalance_period: str = Field(
        default="1M", 
        description="Portfolio rebalancing frequency, e.g.: '1M' for monthly, '1W' for weekly.")

class PortfolioOptimizerTool(BaseTool):
    name: str = "Portfolio Optimization"
    description: str = (
        "Tool that retrieves historical data from Yahoo Finance using VectorBT Pro, "
        "calculates asset returns and optimizes portfolio allocation using the Hierarchical Risk Parity (HRP) model via PyPortfolioOpt, "
        "with periodic rebalancing for better diversification and robustness."
    )
    args_schema: Type[BaseModel] = PortfolioOptimizerInput

    def _run(self, symbols: str, period: str, rebalance_period: str) -> str:
        try:
            # Historical data retrieval
            symbols_list = [s.strip().upper() for s in symbols.split(",")]
            data = vbt.YFData.pull(symbols_list, start=f"{period} ago", tz="UTC")
            close_prices = data.get("Close")
            close_prices = close_prices.resample('1D').last().ffill().bfill()

            # Calculate daily returns
            returns = close_prices.pct_change().dropna()

            # Determine rebalancing points
            rebalance_dates = close_prices.vbt.wrapper.get_index_points(every=rebalance_period)
            
            # Portfolio optimization using Hierarchical Risk Parity (HRP)
            pfo = vbt.PFO.from_pypfopt(
                returns=returns,
                optimizer="hrp",
                target="optimize",
                every=rebalance_period
            )
            
            # Get the latest recommended allocation
            last_allocation = pfo.allocations.iloc[-1]
            
            # Output formatting
            output = "=== Portfolio Optimization ===\n"
            output += f"Analyzed symbols: {', '.join(symbols_list)}\n"
            output += f"Period: {period}\n"
            output += f"Optimization method: Hierarchical Risk Parity (HRP) via PyPortfolioOpt\n"
            output += f"Rebalancing frequency: {rebalance_period}\n\n"
            output += "Recommended allocation:\n"
            for symbol, weight in last_allocation.items():
                output += f"- {symbol}: {weight * 100:.2f}%\n"
            
            return output
        except Exception as e:
            return f"Error optimizing portfolio: {str(e)}"

if __name__ == "__main__":
    # Usage example
    optimizer = PortfolioOptimizerTool()
    symbols = "AAPL, BTC-USD, ^GSPC, COCO.L"
    result = optimizer._run(symbols, period="1 year", rebalance_period="1M")
    print(result)