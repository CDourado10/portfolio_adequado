from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import time
import warnings

class HighRiskShortTermInput(BaseModel):
    """Input schema for High Risk Short Term Portfolio tool."""
    symbols: str = Field(
        ..., 
        description="List of asset symbols for optimization - comma-separated - in yfinance format"
    )
    period: str = Field(
        default="3 months",
        description="Analysis period - 1-6 months recommended for short term"
    )
    rebalance_period: str = Field(
        default="1W", 
        description="Portfolio rebalancing frequency, e.g., '1W' for weekly"
    )
    min_weight: float = Field(
        default=0.10,  
        description="Minimum weight for any asset"
    )
    max_weight: float = Field(
        default=0.60,  
        description="Maximum weight for any asset"
    )
    momentum_lookback: int = Field(
        default=10,  
        description="Number of days for momentum calculation"
    )
    volatility_target: float = Field(
        default=0.60,
        description="Target annualized volatility - e.g., 0.60 for 60 percentage"
    )
    min_volume: float = Field(
        default=500000,  
        description="Minimum daily volume in dollars"
    )
    rsi_period: int = Field(
        default=10,  
        description="Period for RSI calculation"
    )
    sma_short: int = Field(
        default=5,  
        description="Short-term SMA period"
    )
    sma_long: int = Field(
        default=20,  
        description="Long-term SMA period"
    )
    min_r2: float = Field(
        default=0.25,  
        description="Minimum R-squared for trend quality"
    )
    max_rsi: float = Field(
        default=90,  
        description="Maximum RSI value for filtering"
    )
    cvar_beta: float = Field(
        default=0.90,  
        description="Confidence level for CVaR calculation - 0-1"
    )
    risk_free_rate: float = Field(
        default=0.02,
        description="Risk-free rate for optimization - e.g., 0.02 for 2 percentage"
    )

class HighRiskShortTermTool(BaseTool):
    name: str = "High Risk Short Term Portfolio Optimization"
    description: str = (
        "Tool specialized in short-term portfolio optimization for high risk tolerance. "
        "Features: "
        "1. Uses momentum and volatility targeting "
        "2. Weekly rebalancing "
        "3. High concentration allowance "
        "4. Aggressive position sizing "
        "Key benefits: "
        "- Momentum-driven returns "
        "- Quick adaptation to market changes "
        "- High return potential "
        "- Active risk management"
    )
    args_schema: Type[BaseModel] = HighRiskShortTermInput

    def _calculate_technical_metrics(self, close_prices: pd.DataFrame, volume: pd.DataFrame = None,
                                  momentum_lookback: int = 20, rsi_period: int = 14,
                                  sma_short: int = 20, sma_long: int = 50) -> dict:
        """Calcula métricas técnicas para os ativos."""
        metrics = {}
        
        # Momentum (retorno percentual)
        metrics['momentum'] = close_prices.pct_change(momentum_lookback).iloc[-1]
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        metrics['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Tendência (SMA ratio)
        sma_short_val = close_prices.rolling(window=sma_short).mean()
        sma_long_val = close_prices.rolling(window=sma_long).mean()
        metrics['sma_ratio'] = ((close_prices / sma_short_val - 1) + (close_prices / sma_long_val - 1)).iloc[-1]
        
        # Volume médio diário
        if volume is not None:
            metrics['volume'] = (volume * close_prices).rolling(window=momentum_lookback).mean().iloc[-1]
        
        # R-squared da tendência
        x = np.arange(len(close_prices.iloc[-momentum_lookback:]))
        metrics['r2'] = close_prices.iloc[-momentum_lookback:].apply(lambda y: np.corrcoef(x, y)[0,1]**2)
        
        # Tendência (slope)
        metrics['trend'] = close_prices.pct_change().rolling(momentum_lookback).mean().iloc[-1] * momentum_lookback
        
        return metrics

    def _run(self, symbols: str, period: str = "3 months", rebalance_period: str = "1W",
             min_weight: float = 0.05, max_weight: float = 0.50, momentum_lookback: int = 20,
             volatility_target: float = 0.60, min_volume: float = 1000000, rsi_period: int = 14,
             sma_short: int = 20, sma_long: int = 50, min_r2: float = 0.3,
             max_rsi: float = 80, cvar_beta: float = 0.95, risk_free_rate: float = 0.02) -> str:
        try:
            # Data preparation
            symbols_list = [s.strip().upper() for s in symbols.split(",")]
            
            # Data retrieval with retry
            max_retries = 3
            retry_delay = 2
            
            # Buscar dados individualmente para cada símbolo
            all_close_prices = []
            all_volumes = []
            
            for symbol in symbols_list:
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
                                data = yf.download([symbol], period=yf_period)
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
                                    data = yf.download([symbol], start=start)
                                except Exception as e:
                                    raise ValueError(f"Período '{period}' inválido para análise.")
                            close = data.get("Close")
                            volume = data.get("Volume")
                            
                            if close is not None and not close.empty:
                                all_close_prices.append(close)
                                all_volumes.append(volume)
                                break
                    except Exception as e:
                        if retry < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        else:
                            print(f"Failed to fetch data for {symbol}: {str(e)}")
            
            if not all_close_prices:
                raise ValueError("Failed to fetch data for any symbol")
            
            # Combinar os dados
            close_prices = pd.concat(all_close_prices, axis=1)
            volume = pd.concat(all_volumes, axis=1)
            
            # Remover colunas duplicadas se houver
            close_prices = close_prices.loc[:, ~close_prices.columns.duplicated()]
            volume = volume.loc[:, ~volume.columns.duplicated()]
            
            # Data processing
            close_prices = close_prices.resample('1D').last().ffill().bfill()
            returns = close_prices.pct_change().dropna()
            volume = volume.resample('1D').last().ffill().bfill()

            # Cálculo de métricas técnicas
            metrics = self._calculate_technical_metrics(
                close_prices, volume,
                momentum_lookback=momentum_lookback,
                rsi_period=rsi_period,
                sma_short=sma_short,
                sma_long=sma_long
            )

            # Filtragem de ativos
            valid_assets = pd.Series(True, index=close_prices.columns)
            
            # Volume mínimo
            if 'volume' in metrics:
                valid_assets &= metrics['volume'] > min_volume
            
            # Momentum positivo
            valid_assets &= metrics['momentum'] > 0
            
            # Tendência de curto prazo
            valid_assets &= metrics['sma_ratio'] > -0.02
            
            # RSI não sobrecomprado
            valid_assets &= metrics['rsi'] < max_rsi
            
            # Qualidade da tendência
            valid_assets &= metrics['r2'] > min_r2

            # Garantir número mínimo de ativos
            min_assets = max(3, len(symbols_list) // 3)
            if valid_assets.sum() < min_assets:
                # Relaxar critérios gradualmente
                valid_assets = pd.Series(True, index=close_prices.columns)
                if 'volume' in metrics:
                    valid_assets &= metrics['volume'] > (min_volume * 0.8)  # 20% menos volume mínimo
                valid_assets &= metrics['momentum'] > -0.02
                valid_assets &= metrics['r2'] > (min_r2 * 0.7)

            # Preparar dados apenas com ativos válidos
            valid_symbols = [sym for sym, is_valid in valid_assets.items() if is_valid]
            if len(valid_symbols) < 2:
                raise ValueError(f"Insufficient valid symbols. Only {len(valid_symbols)} remaining.")

            valid_returns = returns[valid_symbols].copy()
            valid_close = close_prices[valid_symbols].copy()

            # Portfolio optimization
            pfo = vbt.PFO.from_pypfopt(
                returns=valid_returns,
                optimizer="efficient_cvar",
                target="min_cvar",
                every=rebalance_period,
                weight_bounds=(min_weight, max_weight),
                beta=cvar_beta,
                risk_free_rate=risk_free_rate,
                market_neutral=False
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
            
            # Calculate turnover
            total_turnover = sum(
                abs(pfo.allocations.iloc[i] - pfo.allocations.iloc[i-1]).sum() / 2
                for i in range(1, len(pfo.allocations))
            )
            avg_turnover = total_turnover / len(pfo.allocations)
            
            # Output formatting
            output = "=== High Risk Short Term Portfolio ===\n"
            output += f"Analysis Period: {period}\n"
            output += f"Rebalancing Frequency: {rebalance_period}\n"
            output += f"Valid Assets: {len(valid_symbols)}/{len(symbols_list)}\n\n"
            
            output += "Portfolio Metrics:\n"
            output += f"Initial Value: $ 100,000.00\n"
            output += f"Final Value: $ {total_value.iloc[-1]:,.2f}\n"
            output += f"Total Return: {total_returns.sum() * 100:.2f}%\n"
            output += f"Annualized Return: {annual_return:.2f}%\n"
            output += f"Annualized Volatility: {annual_volatility:.2f}%\n"
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
                mom = metrics['momentum'][symbol]
                rsi = metrics['rsi'][symbol]
                trend = metrics['trend'][symbol]
                vol = metrics['volume'][symbol] if 'volume' in metrics else 0
                
                trend_emoji = "🚀" if trend > 2 else "📈" if trend > 0 else "📉"
                strength_emoji = "💪" if rsi > 60 else "✨" if rsi > 40 else "⚠️"
                
                output += (
                    f"{symbol}: {weight*100:.2f}% {trend_emoji}{strength_emoji} "
                    f"(Mom: {mom*100:+.1f}%, RSI: {rsi:.0f}"
                )
                if 'volume' in metrics:
                    output += f", Vol: ${vol:,.0f}"
                output += ")\n"
            
            return output
            
        except Exception as e:
            return f"Error in portfolio optimization: {str(e)}"

if __name__ == "__main__":
    optimizer = HighRiskShortTermTool()
    # Using liquid high volatility stocks
    symbols = "AAPL,MSFT,NVDA,AMD,META,TSLA,AMZN,GOOGL,QQQ,SPY"
    result = optimizer._run(
        symbols=symbols,
        period="3 months",
        rebalance_period="1W",
        min_weight=0.05,
        max_weight=0.50,
        momentum_lookback=20,
        volatility_target=0.60,
        min_volume=1000000,
        rsi_period=14,
        sma_short=20,
        sma_long=50,
        min_r2=0.3,
        max_rsi=80,
        cvar_beta=0.95,
        risk_free_rate=0.02
    )
    print(result)
