from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import time
import warnings

class LowRiskShortTermInput(BaseModel):
    """Input schema for Low Risk Short Term Portfolio tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for optimization - comma-separated - in yfinance format"
    )
    period: str = Field(
        default="6 months",
        description="Analysis period - 6 months to 1 year recommended for short term"
    )
    rebalance_period: str = Field(
        default="1M", 
        description="Portfolio rebalancing frequency, e.g., '1M' for monthly, '2W' for biweekly"
    )
    beta: float = Field(
        default=0.95,
        description="CVaR confidence level - higher = more conservative"
    )
    min_weight: float = Field(
        default=0.02,
        description="Minimum weight for any asset"
    )
    max_weight: float = Field(
        default=0.15,
        description="Maximum weight for any asset"
    )
    optimization_target: str = Field(
        default="min_cvar",
        description="Optimization method: min_cvar, efficient_return, or efficient_risk"
    )
    target_return: Optional[float] = Field(
        default=None,
        description="Annual target return for efficient_return - optional"
    )
    target_cvar: Optional[float] = Field(
        default=None,
        description="Target CVaR for efficient_risk - optional"
    )
    market_neutral: bool = Field(
        default=False,
        description="If True, creates a market neutral portfolio - requires negative weights"
    )
    init_cash: float = Field(
        default=100000.0,
        description="Initial capital for simulation"
    )
    broker_commission: float = Field(
        default=0.0003,
        description="Broker commission - percentage - per trade"
    )
    slippage: float = Field(
        default=0.0005,
        description="Estimated slippage - percentage - per trade"
    )
    min_liquidity_value: float = Field(
        default=100000.0,
        description="Minimum daily volume in USD to consider an asset liquid"
    )
    min_liquidity_days: int = Field(
        default=21,
        description="Minimum number of days with adequate liquidity in the last 30 days"
    )
    max_position_value: Optional[float] = Field(
        default=None,
        description="Maximum value in USD per position - optional"
    )
    solver: Optional[str] = Field(
        default=None,
        description="Specific CVXPY solver - optional"
    )

class LowRiskShortTermTool(BaseTool):
    name: str = "Low Risk Short Term Portfolio Optimization"
    description: str = (
        "Tool specialized in short-term portfolio optimization for low risk tolerance using CVaR. "
        "Features: "
        "1. Uses Conditional Value at Risk - CVaR - optimization "
        "2. Conservative weight constraints "
        "3. Regular rebalancing "
        "4. Risk-focused optimization "
        "5. Realistic market costs "
        "6. Liquidity constraints "
        "Key benefits: "
        "- Minimizes extreme losses "
        "- Stable short-term performance "
        "- Conservative position sizing "
        "- Regular risk monitoring "
        "- Cost-aware optimization"
    )
    args_schema: Type[BaseModel] = LowRiskShortTermInput

    def _check_liquidity(self, data: vbt.YFData, min_value: float, min_days: int) -> pd.Series:
        """Check assets liquidity based on financial volume."""
        volume = data.get("Volume")
        close = data.get("Close")
        financial_volume = volume * close
        
        # Calculate number of days with adequate liquidity in the last 30 days
        recent_volume = financial_volume.tail(30)
        days_with_liquidity = (recent_volume >= min_value).sum()
        
        # Return True for assets that meet liquidity criteria
        return days_with_liquidity >= min_days

    def _calculate_turnover(self, old_weights: pd.Series, new_weights: pd.Series) -> float:
        """Calculate portfolio turnover between two periods."""
        if old_weights is None:
            return 1.0  # First rebalancing
        return abs(new_weights - old_weights).sum() / 2

    def _run(
        self, 
        symbols: str, 
        period: str = "6 months", 
        rebalance_period: str = "1M",
        beta: float = 0.95, 
        min_weight: float = 0.02, 
        max_weight: float = 0.15,
        optimization_target: str = "min_cvar",
        target_return: Optional[float] = None,
        target_cvar: Optional[float] = None,
        market_neutral: bool = False,
        init_cash: float = 100000.0,
        broker_commission: float = 0.0003,
        slippage: float = 0.0005,
        min_liquidity_value: float = 100000.0,
        min_liquidity_days: int = 21,
        max_position_value: Optional[float] = None,
        solver: Optional[str] = None
    ) -> str:
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
            
            # Liquidity check
            liquid_assets = self._check_liquidity(data, min_liquidity_value, min_liquidity_days)
            if not liquid_assets.any():
                raise ValueError("No assets meet the liquidity criteria")
            
            # Filter out illiquid assets
            close_prices = close_prices.loc[:, liquid_assets]
            if len(close_prices.columns) < 2:
                raise ValueError("Not enough liquid assets for diversification")
            
            # Data processing
            close_prices = close_prices.resample('1D').last().ffill().bfill()
            returns = close_prices.pct_change().dropna()

            # Position value constraints
            if max_position_value is not None:
                max_weights = np.minimum(
                    max_weight,
                    max_position_value / (close_prices.iloc[-1] * init_cash)
                )
            else:
                max_weights = max_weight

            # Basic optimizer configuration
            optimizer_params = {
                "beta": beta,  # CVaR confidence level
                "weight_bounds": (min_weight, max_weights),  # Min/max weight for each asset
                "solver": solver
            }

            # Target-specific configuration
            if optimization_target == "efficient_return" and target_return is not None:
                target_params = {
                    "target": "efficient_return",
                    "target_return": target_return / 100,  # Convert percentage to decimal
                    "market_neutral": market_neutral
                }
            elif optimization_target == "efficient_risk" and target_cvar is not None:
                target_params = {
                    "target": "efficient_risk",
                    "target_cvar": target_cvar / 100,  # Convert percentage to decimal
                    "market_neutral": market_neutral
                }
            else:
                target_params = {
                    "target": "min_cvar",
                    "market_neutral": market_neutral
                }

            # Portfolio optimization
            pfo = vbt.PFO.from_pypfopt(
                returns=returns,
                optimizer="efficient_cvar",
                every=rebalance_period,
                **optimizer_params,
                **target_params
            )

            # Calculate total transaction costs
            total_costs = broker_commission + slippage
            
            # Portfolio simulation with costs
            pf = pfo.simulate(
                close=close_prices,
                init_cash=init_cash,
                freq="1D",
                direction="longonly",
                fees=broker_commission,  # Broker commission
                slippage=slippage,  # Market impact
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
            
            # Calculate CVaR
            returns_arr = total_returns.to_numpy()
            sorted_returns = np.sort(returns_arr[~np.isnan(returns_arr)])  # Remove NaN values
            cvar_cutoff = int(len(sorted_returns) * (1 - beta))
            cvar = sorted_returns[:cvar_cutoff].mean() * 100 if len(sorted_returns) > 0 else 0
            
            # Calculate turnover and costs
            total_turnover = sum(
                self._calculate_turnover(
                    pfo.allocations.iloc[i-1] if i > 0 else None,
                    pfo.allocations.iloc[i]
                )
                for i in range(len(pfo.allocations))
            )
            avg_turnover = total_turnover / len(pfo.allocations)
            
            # Estimate total costs based on turnover
            total_cost = init_cash * avg_turnover * (broker_commission + slippage)
            
            # Output formatting
            output = "=== Low Risk Short Term Portfolio (CVaR Optimization) ===\n"
            output += f"Period of Analysis: {period}\n"
            output += f"Frequency of Rebalancing: {rebalance_period}\n"
            output += f"Method of Optimization: {optimization_target}\n"
            output += f"Market Neutral: {'Yes' if market_neutral else 'No'}\n"
            if target_return is not None:
                output += f"Target Return: {target_return:.2f}%\n"
            if target_cvar is not None:
                output += f"Target CVaR: {target_cvar:.2f}%\n"
            output += f"Level of Confidence of CVaR: {beta}\n"
            output += f"Number of Assets: {len(close_prices.columns)}\n\n"
            
            output += "Portfolio Metrics:\n"
            output += f"Initial Value: $ {init_cash:,.2f}\n"
            output += f"Final Value: $ {total_value.iloc[-1]:,.2f}\n"
            output += f"Total Return: {total_returns.sum() * 100:.2f}%\n"
            output += f"Annualized Return: {annual_return:.2f}%\n"
            output += f"Annualized Volatility: {annual_volatility:.2f}%\n"
            output += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
            output += f"Maximum Drawdown: {max_drawdown:.2f}%\n"
            output += f"CVaR ({beta*100:.0f}%): {cvar:.2f}%\n"
            
            output += "\nAnalysis of Costs:\n"
            output += f"Average Turnover: {avg_turnover*100:.2f}%\n"
            output += f"Total Transaction Costs: $ {total_cost:,.2f}\n"
            output += f"Impact of Transaction Costs: {(total_cost/init_cash)*100:.2f}%\n\n"
            
            output += "Recommended Allocation:\n"
            sorted_allocations = sorted(
                last_allocation.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for symbol, weight in sorted_allocations:
                volume = data.get("Volume").iloc[-1][symbol]
                close = close_prices.iloc[-1][symbol]
                position_value = weight * init_cash
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
    optimizer = LowRiskShortTermTool()
    symbols = "AAPL,MSFT,GOOGL,AMZN,PG,JNJ,KO,PEP,WMT,VZ"  # Conservative stock selection
    result = optimizer._run(
        symbols=symbols,
        period="1 year",
        rebalance_period="1M",
        beta=0.95,
        min_weight=0.02,
        max_weight=0.15
    )
    print(result)
