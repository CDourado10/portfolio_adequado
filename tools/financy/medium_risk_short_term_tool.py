from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import time
import warnings

class MediumRiskShortTermInput(BaseModel):
    """Input schema for Medium Risk Short Term Portfolio tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for optimization - comma-separated - in yfinance format"
    )
    period: str = Field(
        default="1 year",
        description="Analysis period - 6 months to 1 year recommended for short term"
    )
    rebalance_period: str = Field(
        default="1M", 
        description="Portfolio rebalancing frequency, e.g., '1M' for monthly, '2W' for biweekly"
    )
    risk_free_rate: float = Field(
        default=0.02,
        description="Risk-free rate for optimization"
    )
    min_weight: float = Field(
        default=0.05,
        description="Minimum weight for any asset"
    )
    max_weight: float = Field(
        default=0.25,
        description="Maximum weight for any asset"
    )
    target_volatility: float = Field(
        default=0.15,
        description="Target annualized volatility for the portfolio"
    )

class MediumRiskShortTermTool(BaseTool):
    name: str = "Medium Risk Short Term Portfolio Optimization"
    description: str = (
        "Tool specialized in short-term portfolio optimization for medium risk tolerance using Efficient Frontier. "
        "Features: "
        "1. Uses Mean-Variance optimization with balanced constraints "
        "2. Monthly rebalancing "
        "3. Volatility targeting "
        "4. Risk-return balance "
        "Key benefits: "
        "- Balanced risk-return profile "
        "- Regular rebalancing "
        "- Volatility management "
        "- Efficient diversification"
    )
    args_schema: Type[BaseModel] = MediumRiskShortTermInput

    def _run(self, symbols: str, period: str = "1 year", rebalance_period: str = "1M",
             risk_free_rate: float = 0.02, min_weight: float = 0.05, 
             max_weight: float = 0.25, target_volatility: float = 0.15) -> str:
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
                        # Busca os dados históricos com tratamento robusto de período
                        period_map = {
                            "1 day": "1d",
                            "5 days": "5d",
                            "1 month": "1mo",
                            "3 months": "3mo",
                            "6 months": "6mo",
                            "1 year": "1y",
                            "2 years": "2y",
                            "5 years": "5y",
                            "10 years": "10y",
                            "ytd": "ytd",
                            "max": "max"
                        }
                        user_period = period.strip().lower()
                        yf_period = period_map.get(user_period)
                        import yfinance as yf
                        from datetime import datetime, timedelta
                        if yf_period:
                            data = yf.download(symbols_list, period=yf_period)
                        else:
                            try:
                                n, unidade = user_period.split()
                                n = int(n)
                                if "month" in unidade:
                                    delta = timedelta(days=30 * n)
                                elif "year" in unidade:
                                    delta = timedelta(days=365 * n)
                                else:
                                    delta = timedelta(days=n)
                                start = (datetime.now() - delta).strftime('%Y-%m-%d')
                                data = yf.download(symbols_list, start=start)
                            except Exception as e:
                                raise ValueError(f"Período '{period}' inválido para análise.")
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
                "risk_free_rate": risk_free_rate,
                "min_weight": min_weight,
                "max_weight": max_weight,
                "target_volatility": target_volatility,
                "weight_sum_to_one": True
            }

            # Portfolio optimization
            pfo = vbt.PFO.from_pypfopt(
                returns=returns,
                optimizer="efficient_frontier",
                target="efficient_risk",
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
            annual_return = ((1 + total_return) ** (252/total_days) - 1) * 100  # Properly annualized
            annual_volatility = pf.annualized_volatility * 100
            sharpe_ratio = pf.sharpe_ratio
            max_drawdown = abs(pf.max_drawdown * 100)
            
            # Calculate rolling volatility
            rolling_vol = total_returns.rolling(63).std() * np.sqrt(252) * 100  # ~3 months
            current_vol = rolling_vol.iloc[-1]
            
            # Calculate turnover
            total_turnover = sum(
                abs(pfo.allocations.iloc[i] - pfo.allocations.iloc[i-1]).sum() / 2
                for i in range(1, len(pfo.allocations))
            )
            avg_turnover = total_turnover / len(pfo.allocations)
            
            # Output formatting
            output = "=== Medium Risk Short Term Portfolio (Efficient Frontier) ===\n"
            output += f"Analysis Period: {period}\n"
            output += f"Rebalancing Frequency: {rebalance_period}\n"
            output += f"Target Volatility: {target_volatility*100:.1f}%\n"
            output += f"Number of Assets: {len(close_prices.columns)}\n\n"
            
            output += "Portfolio Metrics:\n"
            output += f"Initial Value: $ 100,000.00\n"
            output += f"Final Value: $ {total_value.iloc[-1]:,.2f}\n"
            output += f"Total Return: {total_returns.sum() * 100:.2f}%\n"
            output += f"Annualized Return: {annual_return:.2f}%\n"
            output += f"Annualized Volatility: {annual_volatility:.2f}%\n"
            output += f"Current Volatility: {current_vol:.2f}%\n"
            output += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
            output += f"Maximum Drawdown: {max_drawdown:.2f}%\n"
            
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
    optimizer = MediumRiskShortTermTool()
    symbols = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,V,UNH,JNJ"  # Balanced stock selection
    result = optimizer._run(
        symbols=symbols,
        period="1 year",
        rebalance_period="1M",
        risk_free_rate=0.02,
        min_weight=0.05,
        max_weight=0.25,
        target_volatility=0.25
    )
    print(result)
