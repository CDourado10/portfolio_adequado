from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import time
import warnings

class PortfolioReductionInput(BaseModel):
    """Input schema for Portfolio Reduction tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for reduction (comma-separated) in yfinance format"
    )
    lookback: str = Field(
        default="1 year",
        description="Lookback period in yfinance format (e.g., '1 year', '6 months', '3 months', '1 month')"
    )
    target_assets: int = Field(
        ...,
        description="Target number of assets for the reduced portfolio"
    )
    min_volume_percentile: float = Field(
        default=0.2,
        description="Minimum volume percentile (0-1) for filtering low liquidity assets"
    )
    rebalance_period: str = Field(
        default="1M",
        description="Portfolio rebalancing frequency (e.g., 1M for monthly, 1W for weekly)"
    )

class PortfolioReductionTool(BaseTool):
    name: str = "Portfolio Asset Reduction"
    description: str = (
        "Tool specialized in reducing the number of assets in a portfolio using HRP (Hierarchical Risk Parity). "
        "Features: "
        "1. Smart volume-based filtering "
        "2. Risk-adjusted asset selection "
        "3. Hierarchical clustering "
        "4. Performance metrics"
    )
    args_schema: Type[BaseModel] = PortfolioReductionInput

    def _calculate_metrics(self, prices: pd.DataFrame, volume: pd.DataFrame) -> tuple:
        """Calculate key metrics for asset selection."""
        try:
            # Ensure data is properly aligned and has enough data
            prices = prices.resample('1D').last().ffill().bfill()
            volume = volume.resample('1D').last().ffill().bfill()
            
            if len(prices) < 252:  # Precisa de pelo menos 1 ano de dados
                raise ValueError("Insufficient data for analysis. Need at least 1 year.")
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate metrics
            ann_factor = np.sqrt(252)
            metrics = {
                'volatility': returns.std() * ann_factor,
                'sharpe': returns.mean() / returns.std() * ann_factor,
                'volume': volume.mean()
            }
            
            return metrics, returns
            
        except Exception as e:
            raise ValueError(f"Metrics calculation error: {str(e)}")

    def _reduce_portfolio(self, returns: pd.DataFrame, prices: pd.DataFrame, target_n: int, rebalance_period: str) -> tuple:
        """Reduce portfolio using Hierarchical Portfolio optimization."""
        try:
            # Configure optimization using the correct method
            pfo = vbt.PFO.from_pypfopt(
                returns=returns,
                optimizer="hierarchical_portfolio",  # Método correto
                target="optimize",
                every=rebalance_period  # Período de rebalanceamento configurável
            )
            
            # Run simulation with proper parameters
            pf = pfo.simulate(
                prices,
                freq="1D",
                init_cash=1_000_000,
                fees=0.001,
                slippage=0.001
            )
            
            # Get final allocation
            weights = pfo.allocations.iloc[-1]
            
            # Select top assets
            selected_assets = list(weights.nlargest(target_n).index)
            
            return selected_assets, weights, pf
            
        except Exception as e:
            raise ValueError(f"Portfolio optimization error: {str(e)}")

    def _run(self, symbols: str, lookback: str = "1y", target_assets: int = 10,
             min_volume_percentile: float = 0.2, rebalance_period: str = "1M") -> str:
        """Execute portfolio reduction using Hierarchical Portfolio optimization."""
        try:
            # Prepare symbols
            symbols_list = [s.strip().upper() for s in symbols.split(",")]
            
            # Validate inputs
            if len(symbols_list) <= target_assets:
                return f"⚠️ Target assets ({target_assets}) must be less than input assets ({len(symbols_list)})"
            
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
                        user_period = lookback.strip().lower()
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
                                raise ValueError(f"Período '{lookback}' inválido para análise.")
                        
                        # Get price data
                        close_prices = data.get("Close")
                        volume = data.get("Volume")
                        
                        if close_prices is None or close_prices.empty:
                            raise ValueError("No price data available")
                            
                        # Resample e preenche dados faltantes
                        close_prices = close_prices.resample('1D').last().ffill().bfill()
                        volume = volume.resample('1D').last().ffill().bfill()
                            
                        break
                        
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    raise ValueError(f"Data retrieval failed: {str(e)}")
            
            # Calculate metrics
            metrics, returns = self._calculate_metrics(close_prices, volume)
            
            # Reduce portfolio
            selected_assets, weights, pf = self._reduce_portfolio(
                returns, 
                close_prices, 
                target_assets,
                rebalance_period
            )
            
            # Format results
            result_lines = [
                "📊 Portfolio Reduction Results:",
                f"\n📈 Analysis Period: {lookback}",
                f"💰 Total Return: {(pf.total_return - 1) * 100:.1f}%",
                f"📉 Max Drawdown: {pf.max_drawdown * 100:.1f}%",
                "\n🔍 Selected Assets:"
            ]
            
            # Calcular percentis de volume para referência
            volume_percentiles = {
                asset: (metrics['volume'][asset] >= metrics['volume']).mean() * 100
                for asset in selected_assets
            }
            
            for i, asset in enumerate(selected_assets, 1):
                perf_line = []
                
                # Add performance indicators
                if metrics['sharpe'][asset] > 1.0:
                    perf_line.append(f"⭐ Sharpe: {metrics['sharpe'][asset]:.2f}")
                else:
                    perf_line.append(f"Sharpe: {metrics['sharpe'][asset]:.2f}")
                
                vol = metrics['volatility'][asset]
                if vol < metrics['volatility'].mean():
                    perf_line.append(f"🛡️ Vol: {vol:.1%}")
                else:
                    perf_line.append(f"Vol: {vol:.1%}")
                
                # Add weight
                perf_line.append(f"💰 Weight: {weights[asset]:.1%}")
                
                # Add volume metrics
                volume_pct = volume_percentiles[asset]
                volume_daily = metrics['volume'][asset]
                if volume_pct > 80:
                    perf_line.append(f"💫 Volume: {volume_daily:.2e} ({volume_pct:.0f}th pct)")
                elif volume_pct > 50:
                    perf_line.append(f"📊 Volume: {volume_daily:.2e} ({volume_pct:.0f}th pct)")
                else:
                    perf_line.append(f"⚠️ Volume: {volume_daily:.2e} ({volume_pct:.0f}th pct)")
                
                result_lines.append(
                    f"{i}. {asset}\n   {', '.join(perf_line)}"
                )
            
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    reducer = PortfolioReductionTool()
    symbols = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,JPM,JNJ,V"
    result = reducer._run(
        symbols=symbols,
        lookback="1 year",
        target_assets=5,
        min_volume_percentile=0.2,
        rebalance_period="1M"
    )
    print(result)
