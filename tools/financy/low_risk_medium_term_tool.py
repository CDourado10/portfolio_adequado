from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import time
import warnings

class LowRiskMediumTermInput(BaseModel):
    """Input schema for Low Risk Medium Term Portfolio tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for optimization (comma-separated) in yfinance format"
    )
    period: str = Field(
        default="2 years",
        description="Analysis period (1-3 years recommended for medium term)"
    )
    rebalance_period: str = Field(
        default="3M", 
        description="Portfolio rebalancing frequency, e.g., '3M' for quarterly, '2M' for bimonthly"
    )
    benchmark: float = Field(
        default=0.0,
        description="Benchmark return for semivariance calculation"
    )
    min_weight: float = Field(
        default=0.03,
        description="Minimum weight for any asset"
    )
    max_weight: float = Field(
        default=0.20,
        description="Maximum weight for any asset"
    )

class LowRiskMediumTermTool(BaseTool):
    name: str = "Low Risk Medium Term Portfolio Optimization"
    description: str = (
        "Tool specialized in medium-term portfolio optimization for low risk tolerance using Semivariance. "
        "Features: "
        "1. Uses Efficient Semivariance optimization "
        "2. Focus on downside risk "
        "3. Quarterly rebalancing "
        "4. Conservative position sizing "
        "Key benefits: "
        "- Minimizes downside deviations "
        "- Stable medium-term performance "
        "- Regular risk assessment "
        "- Benchmark-relative optimization"
    )
    args_schema: Type[BaseModel] = LowRiskMediumTermInput

    def _run(self, symbols: str, period: str = "2 years", rebalance_period: str = "3M",
             benchmark: float = 0.0, min_weight: float = 0.03, max_weight: float = 0.20) -> str:
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

            # Semivariance optimization configuration
            optimizer_params = {
                "benchmark": benchmark,
                "risk_free_rate": 0.02,
                "min_weight": min_weight,
                "max_weight": max_weight
            }

            # Portfolio optimization
            pfo = vbt.PFO.from_pypfopt(
                returns=returns,
                optimizer="efficient_semivariance",
                target="min_semivariance",
                every=rebalance_period,
                benchmark=benchmark,
                weight_bounds=(min_weight, max_weight)
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
            sortino_ratio = pf.sortino_ratio
            max_drawdown = abs(pf.max_drawdown * 100)
            
            # Calculate semideviation
            returns_arr = total_returns.to_numpy()
            drops = np.fmin(returns_arr - benchmark, 0)
            semivariance = np.sum(np.square(drops)) / len(returns_arr) * 252
            semideviation = np.sqrt(semivariance) * 100
            
            # Calculate turnover
            total_turnover = sum(
                abs(pfo.allocations.iloc[i] - pfo.allocations.iloc[i-1]).sum() / 2
                for i in range(1, len(pfo.allocations))
            )
            avg_turnover = total_turnover / len(pfo.allocations)
            
            # Output formatting
            output = "=== Low Risk Medium Term Portfolio (Semivariance Optimization) ===\n"
            output += f"Analysis Period: {period}\n"
            output += f"Rebalancing Frequency: {rebalance_period}\n"
            output += f"Benchmark Return: {benchmark*100:.1f}%\n"
            output += f"Number of Assets: {len(close_prices.columns)}\n\n"
            
            output += "Portfolio Metrics:\n"
            output += f"Initial Value: $ 100,000.00\n"
            output += f"Final Value: $ {total_value.iloc[-1]:,.2f}\n"
            output += f"Total Return: {total_returns.sum() * 100:.2f}%\n"
            output += f"Annualized Return: {annual_return:.2f}%\n"
            output += f"Annualized Volatility: {annual_volatility:.2f}%\n"
            output += f"Semideviation: {semideviation:.2f}%\n"
            output += f"Sortino Ratio: {sortino_ratio:.2f}\n"
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
                output += (
                    f"{symbol}: {weight*100:.2f}% "
                    f"($ {position_value:,.2f} | "
                    f"Volume: $ {daily_volume:,.2f})\n"
                )
            
            return output

        except Exception as e:
            return f"Error in portfolio optimization: {str(e)}"

if __name__ == "__main__":
    optimizer = LowRiskMediumTermTool()
    symbols = "AAPL,MSFT,GOOGL,AMZN,PG,JNJ,KO,PEP,WMT,VZ"  # Conservative stock selection
    result = optimizer._run(
        symbols=symbols,
        period="2 years",
        rebalance_period="3M",
        benchmark=0.02/252  # Daily risk-free rate (2% annual)
    )
    print(result)
