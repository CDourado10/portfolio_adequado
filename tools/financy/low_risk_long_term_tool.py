from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import time
import warnings
import riskfolio as rp
from vectorbtpro.portfolio.pfopt.base import resolve_pypfopt_expected_returns

class LowRiskLongTermInput(BaseModel):
    """Input schema for Low Risk Long Term Portfolio tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for optimization (comma-separated) in yfinance format"
    )
    period: str = Field(
        default="5 years",
        description="Analysis period (3-10 years recommended for long term)"
    )
    rebalance_period: str = Field(
        default="6M", 
        description="Portfolio rebalancing frequency, e.g., '6M' for semi-annual, '3M' for quarterly"
    )
    risk_aversion: float = Field(
        default=4.0,
        description="Risk aversion parameter (higher = more conservative, range 1-5)"
    )
    min_weight: float = Field(
        default=0.05,
        description="Minimum weight for any asset"
    )
    max_weight: float = Field(
        default=0.25,
        description="Maximum weight for any asset"
    )
    market_view: str = Field(
        default="neutral",
        description="Market view for Black-Litterman: 'bullish', 'bearish', or 'neutral'"
    )

class LowRiskLongTermTool(BaseTool):
    name: str = "Low Risk Long Term Portfolio Optimization"
    description: str = (
        "Tool specialized in long-term portfolio optimization for low risk tolerance using Black-Litterman. "
        "Features: "
        "1. Uses Black-Litterman model with conservative views "
        "2. Semi-annual rebalancing "
        "3. Market-view integration "
        "4. Long-term risk management "
        "Key benefits: "
        "- Stable long-term performance "
        "- Market wisdom incorporation "
        "- Conservative position sizing "
        "- Regular strategy review"
    )
    args_schema: Type[BaseModel] = LowRiskLongTermInput

    def _run(self, symbols: str, period: str = "5 years", rebalance_period: str = "6M",
             risk_aversion: float = 4.0, min_weight: float = 0.05, 
             max_weight: float = 0.25, market_view: str = "neutral") -> str:
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

            # Calculate market views and parameters
            historical_returns = returns.mean() * 252
            historical_vol = returns.std() * np.sqrt(252)
            market_prices = close_prices.iloc[-1]
            market_caps = market_prices * data.get("Volume").iloc[-1]
            
            # Create views based on market_view parameter
            view_adjustment = {
                'bullish': 0.02,  # Add 2% to expected returns
                'bearish': -0.02,  # Subtract 2% from expected returns
                'neutral': 0.0     # No adjustment
            }[market_view]
            
            # Create absolute views for Black-Litterman
            viewdict = {}
            for symbol in returns.columns:
                expected_return = historical_returns[symbol] + view_adjustment
                viewdict[symbol] = expected_return

            # Portfolio optimization using Black-Litterman and efficient frontier
            weights = vbt.pypfopt_optimize(
                prices=close_prices,
                expected_returns="bl_returns",
                market_prices=market_prices,
                market_caps=market_caps,
                absolute_views=viewdict,
                target="min_volatility",  # Minimize volatility
                weight_bounds=(min_weight, max_weight),
                risk_free_rate=0.02,
                risk_aversion=risk_aversion,  # Adicionando o parâmetro de aversão ao risco
                bl_view_confidence=1.0  # Confiança máxima nas views
            )

            # Convert weights to DataFrame with proper rebalancing
            weights_df = pd.DataFrame(
                [weights] * len(close_prices),
                index=close_prices.index,
                columns=close_prices.columns
            )
            # Apply rebalancing frequency
            weights_df = weights_df.resample(rebalance_period.replace('M', 'ME')).first().ffill()

            # Simulate portfolio
            pf = vbt.Portfolio.from_orders(
                close=close_prices,
                size=weights_df,
                size_type='targetpercent',
                init_cash=100000.0,
                freq="1D",
                direction="longonly",
                group_by=True,
                cash_sharing=True,
                call_seq="auto",
                fees=0.001,  # 0.1% transaction fee
                slippage=0.001  # 0.1% slippage
            )

            # Get portfolio metrics
            total_value = pf.value
            total_returns = pf.returns
            last_allocation = pd.Series(weights, index=close_prices.columns)
            
            # Risk metrics calculation
            total_days = len(total_returns)
            total_return = pf.total_return
            annual_return = ((1 + total_return) ** (252/total_days) - 1) * 100  # Using trading days (252) instead of calendar days
            annual_volatility = pf.annualized_volatility * 100
            sharpe_ratio = pf.sharpe_ratio
            max_drawdown = pf.max_drawdown * 100
            
            # Calculate turnover
            total_turnover = sum(
                abs(weights_df.iloc[i] - weights_df.iloc[i-1]).sum() / 2
                for i in range(1, len(weights_df))
            )
            avg_turnover = total_turnover / len(weights_df)
            
            # Long-term stability metrics
            rolling_vol = pd.Series(total_returns).rolling(252).std() * np.sqrt(252) * 100
            vol_of_vol = rolling_vol.std()
            
            # Output formatting
            output = "=== Low Risk Long Term Portfolio (Black-Litterman) ===\n"
            output += f"Analysis Period: {period}\n"
            output += f"Rebalancing Frequency: {rebalance_period}\n"
            output += f"Market View: {market_view.capitalize()}\n"
            output += f"Risk Aversion: {risk_aversion}\n"
            output += f"View Confidence: {(1 - view_adjustment)*100:.0f}%\n"
            output += f"Number of Assets: {len(close_prices.columns)}\n\n"
            
            output += "Portfolio Metrics:\n"
            output += f"Initial Value: $ 100,000.00\n"
            output += f"Final Value: $ {total_value.iloc[-1]:,.2f}\n"
            output += f"Total Return: {total_returns.sum() * 100:.2f}%\n"
            output += f"Annualized Return: {annual_return:.2f}%\n"
            output += f"Annualized Volatility: {annual_volatility:.2f}%\n"
            output += f"Volatility of Volatility: {vol_of_vol:.2f}%\n"
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
                output += (
                    f"{symbol}: {weight*100:.2f}% "
                    f"($ {position_value:,.2f} | "
                    f"Volume: $ {daily_volume:,.2f})\n"
                )
            
            return output
            
        except Exception as e:
            return f"Error in portfolio optimization: {str(e)}"

if __name__ == "__main__":
    optimizer = LowRiskLongTermTool()
    symbols = "AAPL,MSFT,GOOGL,AMZN,PG,JNJ,KO,PEP,WMT,VZ"
    result = optimizer._run(
        symbols=symbols,
        period="5 years",
        rebalance_period="6M",
        risk_aversion=4.0,
        min_weight=0.05,
        max_weight=0.25,
        market_view="neutral"
    )
    print(result)
