from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import time
import warnings

class MediumRiskLongTermInput(BaseModel):
    """Input schema for Medium Risk Long Term Portfolio tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for optimization - comma-separated - in yfinance format"
    )
    period: str = Field(
        default="5 years",
        description="Analysis period - 3-10 years recommended for long term"
    )
    rebalance_period: str = Field(
        default="3M", 
        description="Portfolio rebalancing frequency, e.g., '3M' for quarterly, '6M' for semi-annual"
    )
    min_weight: float = Field(
        default=0.05,
        description="Minimum weight for any asset"
    )
    max_weight: float = Field(
        default=0.30,
        description="Maximum weight for any asset"
    )
    target_return: float = Field(
        default=0.12,
        description="Target annual return - e.g., 0.12 for 12 percentage"
    )

class MediumRiskLongTermTool(BaseTool):
    name: str = "Medium Risk Long Term Portfolio Optimization"
    description: str = (
        "Tool specialized in long-term portfolio optimization using Mean-Variance optimization with target return. "
        "Features: "
        "1. Uses Efficient Frontier optimization "
        "2. Quarterly rebalancing "
        "3. Return targeting "
        "4. Long-term stability focus "
        "Key benefits: "
        "- Target return optimization "
        "- Long-term stability metrics "
        "- Return and volatility monitoring "
        "- Volume and liquidity analysis"
    )
    args_schema: Type[BaseModel] = MediumRiskLongTermInput

    def _run(self, symbols: str, period: str = "5 years", rebalance_period: str = "3M",
             min_weight: float = 0.05, max_weight: float = 0.30, 
             target_return: float = 0.12) -> str:
        try:
            # Data preparation
            symbols_list = [s.strip().upper() for s in symbols.split(",")]
            
            # Data retrieval with retry
            max_retries = 3
            retry_delay = 2
            
            for retry in range(max_retries):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data = vbt.YFData.pull(
                            symbols_list,
                            start=f"{period} ago",
                            tz="UTC"
                        )
                        close_prices = data.get("Close")
                        if close_prices is None or close_prices.empty:
                            raise ValueError("Unable to retrieve price data")
                        break
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise ValueError(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
            
            # Data processing
            close_prices = close_prices.resample('1D').last().ffill().bfill()
            returns = close_prices.pct_change().dropna()

            # Efficient Frontier optimization configuration
            optimizer_params = {
                "min_weight": min_weight,
                "max_weight": max_weight,
                "target_return": target_return,
                "risk_free_rate": 0.02
            }

            # Portfolio optimization using target return
            pfo = vbt.PFO.from_pypfopt(
                returns=returns,
                optimizer="efficient_frontier",
                target="efficient_return",
                every=rebalance_period,
                **optimizer_params
            )

            # Portfolio simulation
            pf = pfo.simulate(
                close=close_prices,
                init_cash=100000.0,
                freq="1D",
                direction="longonly"
            )

            # Get portfolio metrics
            total_value = pf.value
            total_returns = pf.returns
            last_allocation = pfo.allocations.iloc[-1]
            
            # Risk metrics calculation
            total_days = len(total_returns)
            total_return = pf.total_return
            annual_return = ((1 + total_return) ** (252/total_days) - 1) * 100
            annual_volatility = pf.annualized_volatility * 100
            sharpe_ratio = pf.sharpe_ratio
            max_drawdown = abs(pf.max_drawdown * 100)
            
            # Calculate turnover
            total_turnover = sum(
                abs(pfo.allocations.iloc[i] - pfo.allocations.iloc[i-1]).sum() / 2
                for i in range(1, len(pfo.allocations))
            )
            avg_turnover = total_turnover / len(pfo.allocations)
            
            # Calculate long-term stability metrics
            rolling_returns = total_returns.rolling(252).mean() * 252 * 100  # Annual rolling returns
            rolling_vol = total_returns.rolling(252).std() * np.sqrt(252) * 100
            return_stability = rolling_returns.std()  # Lower is more stable
            vol_stability = rolling_vol.std()  # Lower is more stable
            
            # Calculate efficiency score
            efficiency = (annual_return / target_return / 100 - 1) * 100  # Percentage above/below target
            
            # Output formatting
            output = "=== Medium Risk Long Term Portfolio (CLA) ===\n"
            output += f"Analysis Period: {period}\n"
            output += f"Rebalancing Frequency: {rebalance_period}\n"
            output += f"Target Annual Return: {target_return*100:.1f}%\n"
            output += f"Number of Assets: {len(close_prices.columns)}\n\n"
            
            output += "Portfolio Metrics:\n"
            output += f"Initial Value: $ 100,000.00\n"
            output += f"Final Value: $ {total_value.iloc[-1]:,.2f}\n"
            output += f"Total Return: {total_returns.sum() * 100:.2f}%\n"
            output += f"Annualized Return: {annual_return:.2f}%\n"
            output += f"Target Efficiency: {efficiency:+.2f}%\n"
            output += f"Annualized Volatility: {annual_volatility:.2f}%\n"
            output += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
            output += f"Maximum Drawdown: {max_drawdown:.2f}%\n"
            output += f"Return Stability: {return_stability:.2f}%\n"
            output += f"Volatility Stability: {vol_stability:.2f}%\n"
            
            output += "\nTurnover Analysis:\n"
            output += f"Average Turnover: {avg_turnover*100:.2f}%\n"
            
            output += "\nRecommended Allocation:\n"
            sorted_allocations = sorted(
                last_allocation.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for symbol, weight in sorted_allocations:
                volume = data.get("Volume").iloc[-1][symbol]
                close = close_prices.iloc[-1][symbol]
                position_value = weight * 100000.0
                daily_volume = volume * close
                emoji = "🟢" if weight > 0.20 else "🟡" if weight > 0.10 else "🔴"
                output += (
                    f"{symbol}: {weight*100:.2f}% {emoji} "
                    f"($ {position_value:,.2f} | "
                    f"Volume: $ {daily_volume:,.2f})\n"
                )
            
            return output
            
        except Exception as e:
            return f"Error in portfolio optimization: {str(e)}"

if __name__ == "__main__":
    optimizer = MediumRiskLongTermTool()
    symbols = "AAPL,MSFT,GOOGL,AMZN,FB,NVDA,JPM,V,UNH,JNJ"  # Balanced stock selection
    result = optimizer._run(
        symbols=symbols,
        period="5 years",
        rebalance_period="3M",
        min_weight=0.05,
        max_weight=0.30,
        target_return=0.05
    )
    print(result)
