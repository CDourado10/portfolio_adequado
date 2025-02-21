from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import warnings

class HighRiskMediumTermInput(BaseModel):
    """Input schema for High Risk Medium Term Portfolio tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for optimization - comma-separated - in yfinance format"
    )
    period: str = Field(
        default="2 years",
        description="Analysis period - 1-3 years recommended for medium term"
    )
    rebalance_period: str = Field(
        default="1M",
        description="Portfolio rebalancing frequency, e.g., '1M' for monthly"
    )
    risk_free_rate: float = Field(
        default=0.045,
        description="Risk-free rate for Sharpe ratio calculation"
    )
    min_weight: float = Field(
        default=0.08,
        description="Minimum weight per asset - default: 8 percentage",
        ge=0.0,
        le=0.5
    )
    max_weight: float = Field(
        default=0.20,
        description="Maximum weight per asset - default: 20 percentage",
        ge=0.1,
        le=0.5
    )
    max_assets: int = Field(
        default=6,
        description="Maximum number of assets in portfolio",
        ge=3,
        le=15
    )
    target_volatility: float = Field(
        default=0.40,
        description="Target annual volatility - default: 40 percentage",
        ge=0.25,
        le=0.60
    )

class HighRiskMediumTermTool(BaseTool):
    name: str = "High Risk Medium Term Portfolio Optimization"
    description: str = (
        "Tool specialized in high-risk portfolio optimization for medium term using multi-factor strategy.\n"
        "Features:\n"
        "1. Multi-factor selection - Momentum, Volatility, Growth\n"
        "2. Monthly rebalancing\n"
        "3. Factor-based constraints\n"
        "4. Dynamic risk allocation\n"
        "Benefits:\n"
        "- Factor premium capture\n"
        "- Systematic strategy\n"
        "- High growth potential\n"
        "- Risk factor balancing"
    )
    args_schema: Type[BaseModel] = HighRiskMediumTermInput

    def _calculate_factor_metrics(self, close_prices: pd.DataFrame, returns: pd.DataFrame,
                               momentum_window: int = 126, volatility_window: int = 63) -> dict:
        """Calculate factor metrics for portfolio optimization."""
        metrics = {}
        
        # Momentum metrics (more aggressive)
        momentum = returns.rolling(momentum_window).mean() * 252
        momentum_vol = returns.rolling(momentum_window).std() * np.sqrt(252)
        momentum_z = momentum.iloc[-1]
        momentum_sharpe = momentum / momentum_vol
        momentum_score = 0.7 * momentum_z + 0.3 * momentum_sharpe.iloc[-1]  # Combines return and quality
        
        # Volatility metrics (preference for moderate/high volatility)
        volatility = returns.rolling(volatility_window).std() * np.sqrt(252)
        volatility_z = volatility.iloc[-1]
        vol_score = pd.Series(
            np.where(
                volatility_z > 0.8,
                volatility_z * 0.5,  # Reduces score for very high volatility
                volatility_z
            ),
            index=volatility_z.index
        )
        
        # Growth and quality with more weight in recent periods
        growth_rates = {}
        growth_quality = {}
        
        for col in close_prices.columns:
            prices = close_prices[col].dropna()
            if len(prices) > 0:
                # Growth in different periods
                growth_3m = (prices.iloc[-1] / prices.iloc[-min(63, len(prices))] - 1)
                growth_6m = (prices.iloc[-1] / prices.iloc[-min(126, len(prices))] - 1)
                growth_1y = (prices.iloc[-1] / prices.iloc[-min(252, len(prices))] - 1)
                
                # Weighted average with more weight on medium term
                growth_rates[col] = (0.2 * growth_3m + 0.5 * growth_6m + 0.3 * growth_1y)
                
                # Growth quality considering consistency
                returns_std = returns[col].rolling(volatility_window).std()
                growth_quality[col] = growth_rates[col] / (returns_std.iloc[-1] if returns_std.iloc[-1] > 0 else 1)
        
        # Trend with more emphasis on medium term
        sma_20 = close_prices.rolling(20).mean()
        sma_50 = close_prices.rolling(50).mean()
        sma_200 = close_prices.rolling(200).mean()
        
        trend_short = ((close_prices - sma_20) / sma_20).iloc[-1]
        trend_medium = ((close_prices - sma_50) / sma_50).iloc[-1]
        trend_long = (sma_50 > sma_200).iloc[-1]
        
        trend_strength = 0.3 * trend_short + 0.7 * trend_medium
        
        metrics['momentum'] = momentum_score
        metrics['volatility'] = vol_score
        metrics['growth_rates'] = pd.Series(growth_rates)
        metrics['growth_quality'] = pd.Series(growth_quality)
        metrics['trend_strength'] = trend_strength
        metrics['trend_direction'] = trend_long
        
        return metrics

    def _run(self, symbols: str, period: str = "2 years", rebalance_period: str = "1M",
             risk_free_rate: float = 0.045, min_weight: float = 0.08, max_weight: float = 0.20, max_assets: int = 6, target_volatility: float = 0.40) -> str:
        """Executes the high-risk medium-term portfolio optimization strategy."""
        try:
            print("Starting portfolio optimization...")
            
            # Configurations for high-risk with control
            momentum_window = 126  # ~6 months
            volatility_window = 63  # ~3 months
            
            # Data preparation
            symbols_list = [s.strip().upper() for s in symbols.split(",")]
            print(f"Processing {len(symbols_list)} symbols: {', '.join(symbols_list)}")
            
            # Data collection
            all_data = []
            valid_symbols = []
            
            for symbol in symbols_list:
                try:
                    print(f"Fetching data for {symbol}...")
                    symbol_data = vbt.YFData.pull(
                        [symbol],
                        start=f"-{period}",
                        tz="UTC",
                        silence_warnings=True
                    )
                    
                    if symbol_data is not None:
                        close = symbol_data.get('Close')
                        if close is not None and not close.empty:
                            if close.isnull().sum().sum() / len(close) < 0.1:
                                all_data.append(close)
                                valid_symbols.append(symbol)
                            else:
                                print(f"Warning: {symbol} has too many missing values")
                        else:
                            print(f"Warning: No price data found for {symbol}")
                    else:
                        print(f"Warning: Data not found for {symbol}")
                        
                except Exception as e:
                    print(f"Error fetching {symbol}: {str(e)}")
                    continue
            
            if len(valid_symbols) < 2:
                raise ValueError(f"Insufficient valid symbols (only {len(valid_symbols)} remaining)")
            
            print(f"Valid symbols after filtering: {', '.join(valid_symbols)}")
            
            # Data processing
            close_prices = pd.concat(all_data, axis=1)
            close_prices.columns = valid_symbols
            close_prices = close_prices.ffill().bfill()
            
            print(f"Final data format: {close_prices.shape}")
            
            returns = close_prices.pct_change().dropna()
            
            # Factor calculation
            factor_metrics = self._calculate_factor_metrics(
                close_prices, 
                returns,
                momentum_window, 
                volatility_window
            )
            
            # Composite score calculation
            composite_score = pd.Series(0.0, index=valid_symbols)
            composite_score += 0.45 * factor_metrics['momentum']  # Increased
            composite_score += 0.20 * factor_metrics['volatility']  # Reduced
            composite_score += 0.35 * (factor_metrics['growth_rates'] * factor_metrics['growth_quality'])  # Combined
            
            # Trend adjustment
            trend_mask = factor_metrics['trend_direction']
            trend_strength = factor_metrics['trend_strength']
            composite_score[trend_mask] *= (1 + trend_strength[trend_mask] * 0.3)
            
            # Asset selection
            min_score = composite_score.quantile(0.4)  # Top 60% (more selective)
            selected_assets = composite_score[composite_score >= min_score].index.tolist()
            
            if len(selected_assets) < 2:
                print("Warning: Few assets passed the filter, relaxing criteria...")
                selected_assets = composite_score.nlargest(min(5, len(composite_score))).index.tolist()
            elif len(selected_assets) > max_assets:
                print(f"Limiting to top {max_assets} assets...")
                selected_assets = composite_score[selected_assets].nlargest(max_assets).index.tolist()
            
            print(f"Selected assets: {', '.join(selected_assets)}")
            
            # Portfolio optimization
            print("Running portfolio optimization...")
            
            try:
                # First try with target_volatility
                pfo = vbt.PFO.from_pypfopt(
                    returns=returns[selected_assets],
                    optimizer="efficient_frontier",
                    target="efficient_risk",
                    target_volatility=target_volatility,
                    every=rebalance_period,
                    weight_bounds=(min_weight, max_weight),
                    risk_free_rate=risk_free_rate,
                    market_neutral=False,
                    risk_aversion=0.5
                )
            except Exception as e:
                if "minimum volatility" in str(e):
                    print("Warning: Portfolio minimum volatility is higher than the target. Using max_sharpe...")
                    # Fallback to max_sharpe if unable to reach target volatility
                    pfo = vbt.PFO.from_pypfopt(
                        returns=returns[selected_assets],
                        optimizer="efficient_frontier",
                        target="max_sharpe",
                        every=rebalance_period,
                        weight_bounds=(min_weight, max_weight),
                        risk_free_rate=risk_free_rate,
                        market_neutral=False,
                        risk_aversion=0.5
                    )
                else:
                    raise e
            
            # Simulation
            print("Simulating portfolio performance...")
            pf = pfo.simulate(
                close=close_prices[selected_assets],
                init_cash=100000.0,
                freq="1D",
                direction="longonly"
            )
            
            # Portfolio metrics
            print("Calculating portfolio metrics...")
            total_return = pf.total_return
            annual_return = pf.annualized_return
            annual_volatility = pf.annualized_volatility
            sharpe_ratio = pf.sharpe_ratio
            max_drawdown = pf.max_drawdown
            
            # Factor exposure
            last_allocation = pfo.allocations.iloc[-1]
            factor_exposures = {
                'Momentum': (factor_metrics['momentum'] * last_allocation).sum(),
                'Volatility': (factor_metrics['volatility'] * last_allocation).sum(),
                'Growth': (factor_metrics['growth_rates'] * last_allocation).sum()
            }
            
            # Concentration
            concentration = (last_allocation ** 2).sum()
            top_3_weight = pd.Series(last_allocation).nlargest(3).sum() * 100
            
            # Output formatting
            output = "=== High Risk Medium Term Portfolio ===\n"
            output += f"Analysis Period: {period}\n"
            output += f"Rebalancing Frequency: {rebalance_period}\n"
            output += f"Number of Assets: {len(selected_assets)}\n\n"
            
            output += "Portfolio Metrics:\n"
            output += f"Total Return: {total_return * 100:.2f}%\n"
            output += f"Annual Return: {annual_return * 100:.2f}%\n"
            output += f"Annual Volatility: {annual_volatility * 100:.2f}%\n"
            output += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
            output += f"Max Drawdown: {abs(max_drawdown) * 100:.2f}%\n\n"
            
            output += "Factor Exposure:\n"
            for factor, exposure in factor_exposures.items():
                indicator = "🔥" if exposure > 0.5 else "✨" if exposure > 0 else "❄️"
                output += f"{factor}: {exposure:+.2f} {indicator}\n"
            
            output += f"\nConcentration (HHI): {concentration:.3f}\n"
            output += f"Top 3 Weight: {top_3_weight:.1f}%\n\n"
            
            output += "Recommended Allocation:\n"
            sorted_allocations = sorted(
                last_allocation.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for symbol, weight in sorted_allocations:
                momentum = factor_metrics['momentum'][symbol]
                growth = factor_metrics['growth_rates'][symbol]
                trend = "📈" if factor_metrics['trend_direction'][symbol] else "📉"
                quality = "⭐" if factor_metrics['growth_quality'][symbol] > 0.8 else "✨"
                
                output += (
                    f"{symbol}: {weight*100:.2f}% {trend}{quality} "
                    f"(Mom: {momentum:+.2f}, "
                    f"Growth: {growth*100:+.1f}%)\n"
                )
            
            print("Portfolio optimization completed successfully!")
            return output
            
        except Exception as e:
            return f"Error in portfolio optimization: {str(e)}"

if __name__ == "__main__":
    # Example usage of the tool
    optimizer = HighRiskMediumTermTool()
    
    # List of semiconductors (high-growth sector)
    symbols = "NVDA,AMD,ASML,TSM,AVGO,MRVL,QCOM,AMAT,KLAC,ON"
    
    # Running optimization with adjusted parameters
    result = optimizer._run(
        symbols=symbols,
        period="2 years",
        rebalance_period="1M",
        risk_free_rate=0.045,
        min_weight=0.08,
        max_weight=0.20,
        max_assets=6,
        target_volatility=0.40
    )
    
    # Displaying results
    print(result)
