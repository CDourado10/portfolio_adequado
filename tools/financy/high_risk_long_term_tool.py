from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import time
import warnings
from scipy import stats

class HighRiskLongTermInput(BaseModel):
    """Input schema for High Risk Long Term Portfolio tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for optimization - comma-separated - in yfinance format"
    )
    period: str = Field(
        default="5 years",
        description="Analysis period - 3-10 years recommended for long term growth"
    )
    rebalance_period: str = Field(
        default="3M", 
        description="Portfolio rebalancing frequency, e.g., '3M' for quarterly"
    )
    min_weight: float = Field(
        default=0.05,
        description="Minimum weight for any asset"
    )
    max_weight: float = Field(
        default=0.45,
        description="Maximum weight for any asset"
    )
    growth_threshold: float = Field(
        default=0.15,
        description="Minimum annual growth rate threshold"
    )
    min_r2: float = Field(
        default=0.4,
        description="Minimum R² - quality - threshold for growth trend"
    )
    min_trend: float = Field(
        default=-0.1,
        description="Minimum trend strength threshold"
    )
    min_volume: float = Field(
        default=100000,
        description="Minimum average daily volume"
    )
    risk_free_rate: float = Field(
        default=0.02,
        description="Risk-free rate for Sharpe ratio calculation"
    )

class HighRiskLongTermTool(BaseTool):
    name: str = "High Risk Long Term Portfolio Optimization"
    description: str = (
        "Tool specialized in long-term high-risk portfolio optimization focusing on aggressive growth. "
        "Features: "
        "1. Growth-focused selection "
        "2. Quarterly rebalancing "
        "3. High concentration allowance "
        "4. Secular trend analysis "
        "Key benefits: "
        "- Long-term growth capture "
        "- Trend following "
        "- High return potential "
        "- Quality filtering"
    )
    args_schema: Type[BaseModel] = HighRiskLongTermInput

    def _calculate_growth_metrics(self, close_prices: pd.DataFrame) -> dict:
        """Calculate growth and trend metrics for portfolio optimization."""
        metrics = {}
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        # Long-term growth rate (exponential regression)
        growth_rates = {}
        growth_r2 = {}
        for col in close_prices.columns:
            prices = close_prices[col].dropna()
            if len(prices) > 0:
                x = np.arange(len(prices))
                y = np.log(prices.values)
                slope, intercept, r_value, _, _ = stats.linregress(x, y)
                growth_rates[col] = np.exp(slope * 252) - 1  # Annualized
                growth_r2[col] = r_value ** 2
        
        # Rolling metrics
        window_252 = returns.rolling(252)
        annual_returns = window_252.mean() * 252
        annual_vol = window_252.std() * np.sqrt(252)
        
        # Trend strength
        sma_200 = close_prices.rolling(200).mean()
        trend_strength = (close_prices - sma_200) / sma_200
        
        metrics['growth_rates'] = pd.Series(growth_rates)
        metrics['growth_r2'] = pd.Series(growth_r2)
        metrics['annual_returns'] = annual_returns.iloc[-1]
        metrics['annual_vol'] = annual_vol.iloc[-1]
        metrics['trend_strength'] = trend_strength.iloc[-1]
        
        return metrics

    def _run(self, symbols: str, period: str = "5 years", rebalance_period: str = "3M",
             min_weight: float = 0.05, max_weight: float = 0.45, 
             growth_threshold: float = 0.15, min_r2: float = 0.4, min_trend: float = -0.1, 
             min_volume: float = 100000, risk_free_rate: float = 0.02) -> str:
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
                        
                        # Verificar dados válidos
                        if close_prices is None or close_prices.empty:
                            raise ValueError("Unable to retrieve price data")
                        
                        # Remover colunas com dados faltantes
                        missing_data = close_prices.isnull().sum()
                        if missing_data.any():
                            bad_symbols = missing_data[missing_data > 0].index.tolist()
                            print(f"Warning: Removing symbols with missing data: {', '.join(bad_symbols)}")
                            close_prices = close_prices.drop(columns=bad_symbols)
                        
                        if len(close_prices.columns) < 2:
                            raise ValueError(f"Insufficient valid symbols. Only {len(close_prices.columns)} remaining.")
                        
                        break
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise ValueError(f"Failed to fetch data after {max_retries} attempts:\n{str(e)}\n\nTry using different symbols or a shorter period.")
            
            # Data processing
            close_prices = close_prices.resample('1D').last().ffill().bfill()
            returns = close_prices.pct_change().dropna()
            
            # Calculate growth metrics
            growth_metrics = self._calculate_growth_metrics(close_prices)
            
            # Asset filtering based on growth metrics
            valid_assets = pd.Series(True, index=symbols_list)
            
            # Growth rate filter (mais flexível para alto risco)
            valid_assets &= growth_metrics['growth_rates'] > growth_threshold * 0.5
            
            # Growth quality filter (R² mais flexível)
            valid_assets &= growth_metrics['growth_r2'] > min_r2
            
            # Trend strength filter (mais flexível)
            valid_assets &= growth_metrics['trend_strength'] > min_trend
            
            # Volume filter
            if data.get("Volume") is not None:
                volume = data.get("Volume")
                avg_volume = volume.mean()
                valid_assets &= avg_volume > min_volume
            
            # Ensure we have enough valid assets
            min_assets = max(3, len(symbols_list) // 3)  # Pelo menos 3 ativos ou 1/3 da lista
            if valid_assets.sum() < min_assets:
                # Relaxar critérios gradualmente
                while valid_assets.sum() < min_assets and growth_threshold > 0.05:
                    growth_threshold *= 0.8
                    valid_assets = pd.Series(True, index=symbols_list)
                    valid_assets &= growth_metrics['growth_rates'] > growth_threshold * 0.5
                    valid_assets &= growth_metrics['growth_r2'] > min_r2 * 0.8
                    valid_assets &= growth_metrics['trend_strength'] > min_trend * 0.8

            # Portfolio optimization using Efficient Frontier
            optimizer_params = {
                "weight_bounds": (min_weight, max_weight),  # Usando os limites fornecidos
                "risk_free_rate": risk_free_rate,  # Taxa livre de risco (pode ser parametrizada se necessário)
                "market_neutral": False,
                "target_volatility": None,  # Permite volatilidade livre para estratégia de alto risco
            }

            # Garantir que estamos usando apenas ativos válidos e seus dados correspondentes
            valid_symbols = [sym for sym, is_valid in valid_assets.items() if is_valid]
            if len(valid_symbols) < 2:
                raise ValueError(f"Insufficient valid symbols for optimization. Only {len(valid_symbols)} remaining after filtering.")

            # Preparar os dados apenas com ativos válidos
            valid_returns = returns[valid_symbols].copy()
            valid_close = close_prices[valid_symbols].copy()

            # Verificar se temos dados suficientes
            if valid_returns.empty or valid_close.empty:
                raise ValueError("No valid data available for optimization after filtering")

            try:
                pfo = vbt.PFO.from_pypfopt(
                    returns=valid_returns,
                    optimizer="efficient_frontier",
                    target="max_sharpe",
                    every=rebalance_period,
                    **optimizer_params
                )

                # Portfolio simulation
                pf = pfo.simulate(
                    close=valid_close,
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
                
                # Calculate growth quality metrics
                portfolio_growth_rate = (growth_metrics['growth_rates'][valid_symbols] * last_allocation).sum() * 100
                portfolio_r2 = (growth_metrics['growth_r2'][valid_symbols] * last_allocation).sum()
                avg_trend_strength = (growth_metrics['trend_strength'][valid_symbols] * last_allocation).sum() * 100
                
                # Calculate concentration metrics
                concentration = (last_allocation ** 2).sum()  # Herfindahl index
                top_3_weight = pd.Series(last_allocation).nlargest(3).sum() * 100
                
                # Calculate volume metrics
                volume = data.get("Volume")
                if volume is not None:
                    avg_volume = volume[valid_symbols].mean().mean()
                    volume_growth = ((volume[valid_symbols].iloc[-252:].mean() / volume[valid_symbols].iloc[:-252].mean() - 1) * 100).mean()
                    liquidity_score = ((volume[valid_symbols].mean() * close_prices[valid_symbols].iloc[-1]) / 1e6).mean()  # em milhões
                else:
                    avg_volume = 0
                    volume_growth = 0
                    liquidity_score = 0

                # Output formatting
                output = "=== High Risk Long Term Portfolio (Growth) ===\n"
                output += f"Analysis Period: {period}\n"
                output += f"Rebalancing Frequency: {rebalance_period}\n"
                output += f"Growth Threshold: {growth_threshold*100:.1f}%\n"
                output += f"Number of Assets: {len(valid_assets[valid_assets])}/{len(symbols_list)}\n\n"
                
                output += "Portfolio Metrics:\n"
                output += f"Initial Value: $ 100,000.00\n"
                output += f"Final Value: $ {total_value.iloc[-1]:,.2f}\n"
                output += f"Total Return: {total_return * 100:.2f}%\n"
                output += f"Annualized Return: {annual_return:.2f}%\n"
                output += f"Annualized Volatility: {annual_volatility:.2f}%\n"
                output += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                output += f"Maximum Drawdown: {max_drawdown:.2f}%\n\n"
                
                output += "Growth Metrics:\n"
                output += f"Portfolio Growth Rate: {portfolio_growth_rate:.1f}%\n"
                output += f"Growth Quality (R²): {portfolio_r2:.3f}\n"
                output += f"Trend Strength: {avg_trend_strength:+.1f}%\n"
                output += f"Concentration (HHI): {concentration:.3f}\n"
                output += f"Top 3 Weight: {top_3_weight:.1f}%\n\n"
                
                output += "Volume Analysis:\n"
                output += f"Average Daily Volume: {avg_volume:,.0f}\n"
                output += f"Volume Growth (YoY): {volume_growth:.1f}%\n"
                output += f"Average Liquidity Score: {liquidity_score:.2f}M\n\n"
                
                output += "Recommended Allocation:\n"
                sorted_allocations = sorted(
                    last_allocation.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                for symbol, weight in sorted_allocations:
                    volume_val = data.get("Volume").iloc[-1][symbol]
                    close_val = close_prices.iloc[-1][symbol]
                    position_value = weight * 100000.0
                    daily_volume = volume_val * close_val
                    growth_emoji = "🚀" if growth_metrics['growth_rates'][symbol] > 0.25 else "📈"
                    quality_emoji = "⭐" if growth_metrics['growth_r2'][symbol] > 0.8 else "✨"
                    output += (
                        f"{symbol}: {weight*100:.2f}% {growth_emoji}{quality_emoji} "
                        f"($ {position_value:,.2f} | "
                        f"Volume: $ {daily_volume:,.2f})\n"
                    )
                
                return output
                
            except Exception as e:
                return f"Error in portfolio optimization: {str(e)}"

        except Exception as e:
            return f"Error in portfolio optimization: {str(e)}"

if __name__ == "__main__":
    optimizer = HighRiskLongTermTool()
    # Using more established growth stocks
    symbols = "AAPL,MSFT,AMZN,GOOGL,META,NVDA,AMD,ADBE,CRM,NFLX"
    result = optimizer._run(
        symbols=symbols,
        period="5 years",
        rebalance_period="3M",
        min_weight=0.05,
        max_weight=0.45,
        growth_threshold=0.15,
        min_r2=0.4,
        min_trend=-0.1,
        min_volume=100000,
        risk_free_rate=0.02
    )
    print(result)
