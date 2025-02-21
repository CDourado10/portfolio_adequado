from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import time
import warnings

class LowRiskPortfolioInput(BaseModel):
    """Input schema for LowRiskPortfolio tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for optimization (comma-separated) in yfinance format"
    )
    period: str = Field(
        default="1 year",
        description="Analysis period (1 year, 2 years, 5 years, 10 years, ytd, max)"
    )
    rebalance_period: str = Field(
        default="3M", 
        description="Portfolio rebalancing frequency, e.g., '1M' for monthly, '1W' for weekly, '3M' for quarterly"
    )
    target_assets: int = Field(
        default=None,
        description="Target number of assets for portfolio reduction. If None, uses all assets."
    )
    investment_horizon: str = Field(
        default="short",
        description="Investment horizon: 'short', 'medium', or 'long'"
    )

class LowRiskPortfolioTool(BaseTool):
    name: str = "Low Risk Portfolio Optimization"
    description: str = (
        "Tool that implements portfolio optimization for low risk tolerance across different time horizons. "
        "Features: "
        "1. Uses CVaR optimization for short-term "
        "2. Uses Semivariance for medium-term "
        "3. Uses Black-Litterman for long-term "
        "4. Conservative weight allocation "
        "5. Regular rebalancing with risk monitoring "
        "Key benefits: "
        "- Focus on capital preservation "
        "- Downside risk protection "
        "- Stable returns over time "
        "- Multiple risk measures"
    )
    args_schema: Type[BaseModel] = LowRiskPortfolioInput

    def _run(self, symbols: str, period: str = "1 year", rebalance_period: str = "3M", 
             target_assets: int = None, investment_horizon: str = "short") -> str:
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

            # Select optimizer based on investment horizon
            if investment_horizon == "short":
                optimizer_name = "efficient_cvar"
                optimizer_params = {
                    "beta": 0.95,  # Conservative CVaR threshold
                    "risk_free_rate": 0.02,
                    "min_weight": 0.02,
                    "max_weight": 0.15
                }
            elif investment_horizon == "medium":
                optimizer_name = "efficient_semivariance"
                optimizer_params = {
                    "benchmark": 0,
                    "min_weight": 0.03,
                    "max_weight": 0.20
                }
            else:  # long-term
                optimizer_name = "black_litterman"
                optimizer_params = {
                    "risk_aversion": 4.0,  # High risk aversion
                    "min_weight": 0.05,
                    "max_weight": 0.25
                }

            # Asset reduction if specified
            if target_assets and target_assets < len(symbols_list):
                # Initial optimization for clustering
                pfo_initial = vbt.PFO.from_pypfopt(
                    returns=returns,
                    optimizer=optimizer_name,
                    target="optimize"
                )
                
                initial_weights = pfo_initial.allocate_asset_weights()
                corr_matrix = returns.corr()
                
                selected_assets = self._reduce_assets(
                    returns.columns,
                    initial_weights,
                    corr_matrix,
                    target_assets
                )
                
                returns = returns[selected_assets]

            # Final optimization
            pfo = vbt.PFO.from_pypfopt(
                returns=returns,
                optimizer=optimizer_name,
                target="optimize",
                every=rebalance_period,
                **optimizer_params
            )

            # Portfolio simulation
            pf = pfo.simulate(data, freq="1D")
            total_value = pf.value
            total_returns = pf.returns
            init_cash = pf.init_cash
            last_allocation = pfo.allocations.iloc[-1]
            
            # Calculate metrics
            annual_return = total_returns.mean() * 252 * 100
            annual_volatility = total_returns.std() * np.sqrt(252) * 100
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
            max_drawdown = abs(pf.drawdown.min() * 100)
            
            # Output formatting
            output = f"=== Low Risk Portfolio ({investment_horizon.capitalize()} Term) ===\n"
            output += f"Optimization Method: {optimizer_name}\n"
            output += f"Analysis Period: {period}\n"
            output += f"Rebalancing Frequency: {rebalance_period}\n"
            output += f"Number of Assets: {len(returns.columns)}\n\n"
            
            output += "Portfolio Metrics:\n"
            output += f"Initial Value: $ {init_cash:,.2f}\n"
            output += f"Final Value: $ {total_value.iloc[-1]:,.2f}\n"
            output += f"Total Return: {total_returns.sum() * 100:.2f}%\n"
            output += f"Annualized Return: {annual_return:.2f}%\n"
            output += f"Annualized Volatility: {annual_volatility:.2f}%\n"
            output += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
            output += f"Maximum Drawdown: {max_drawdown:.2f}%\n\n"
            
            output += "Recommended Allocation:\n"
            sorted_allocations = sorted(
                last_allocation.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for symbol, weight in sorted_allocations:
                emoji = "🟢" if weight > 0.10 else "🟡" if weight > 0.05 else "🔴"
                output += f"- {symbol:<8}: {weight * 100:>6.2f}% {emoji}\n"
            
            return output
            
        except Exception as e:
            return f"Error in portfolio optimization: {str(e)}"

    def _reduce_assets(self, assets, weights, corr_matrix, target_n):
        """
        Reduce number of assets while maintaining diversification
        
        Args:
            assets: List of asset names
            weights: Initial portfolio weights
            corr_matrix: Correlation matrix
            target_n: Target number of assets
        """
        weights = pd.Series(weights) if not isinstance(weights, pd.Series) else weights
        sorted_weights = weights.sort_values(ascending=False)
        
        selected = [sorted_weights.index[0]]
        remaining = list(set(assets) - set(selected))
        
        while len(selected) < target_n and remaining:
            avg_corr = corr_matrix[remaining].loc[selected].mean()
            corr_weight_score = avg_corr - weights[remaining] * 2
            next_asset = corr_weight_score.idxmin()
            
            selected.append(next_asset)
            remaining.remove(next_asset)
        
        return selected

if __name__ == "__main__":
    optimizer = LowRiskPortfolioTool()
    symbols = "AAPL, MSFT, GOOGL, TSLA, BTC-USD, ^GSPC, DOGE-USD"
    result = optimizer._run(
        symbols=symbols,
        period="1 year",
        rebalance_period="3M",
        target_assets=5,
        investment_horizon="short"
    )
    print(result)
