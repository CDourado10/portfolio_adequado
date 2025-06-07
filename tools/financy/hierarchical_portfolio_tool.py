from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd
import time
import warnings

class PortfolioOptimizerInput(BaseModel):
    """Input schema for PortfolioOptimizer tool."""
    symbols: str = Field(..., description="List of asset symbols to be optimized (comma-separated), in yfinance format. For example: 'AAPL, BTC-USD, ^GSPC, COCO.L'")
    period: str = Field(
        default="1 year",
        description="Analysis period (3 months, 6 months, 1 year, 2 years, 5 years, 10 years, ytd, max)"
    )
    rebalance_period: str = Field(
        default="1M", 
        description="Portfolio rebalancing frequency, e.g.: '1M' for monthly, '1W' for weekly.")

    optimizer: str = Field(
        default="hierarchical_portfolio", 
        description="Optimizer to use. Options: efficient_frontier, efficient_cdar, efficient_cvar, efficient_semivariance, black_litterman, hierarchical_portfolio, cla")

    tz: str = Field(
        default="UTC", 
        description="Time zone for data retrieval and processing. Default is UTC.")

class PortfolioOptimizerTool(BaseTool):
    name: str = "Portfolio Optimization"
    description: str = (
        "Tool that retrieves historical data from Yahoo Finance using VectorBT Pro, "
        "calculates asset returns and optimizes portfolio allocation using one of the supported models via PyPortfolioOpt, "
        "with periodic rebalancing for better diversification and robustness."
    )
    args_schema: Type[BaseModel] = PortfolioOptimizerInput

    def _run(self, symbols: str, period: str = "1 year", rebalance_period: str = "1M", 
             optimizer: str = "hierarchical_portfolio", tz: str = "UTC") -> str:
        try:
            # Historical data retrieval
            symbols_list = [s.strip().upper() for s in symbols.split(",")]
            
            # Tenta buscar dados com retry
            max_retries = 3
            retry_delay = 2  # segundos
            
            for retry in range(max_retries):
                try:
                    # Limpa warnings anteriores
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
                        
                        # Verifica se temos dados válidos
                        close_prices = data.get("Close")
                        if close_prices is None or close_prices.empty:
                            raise ValueError("Não foi possível obter dados de preço")
                            
                        # Se chegou aqui, temos dados válidos
                        break
                        
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Aumenta o delay exponencialmente
                        continue
                    else:
                        raise ValueError(f"Falha ao buscar dados após {max_retries} tentativas: {str(e)}")
            
            # Processa os dados
            close_prices = close_prices.resample('1D').last().ffill().bfill()

            # Calculate daily returns
            returns = close_prices.pct_change().dropna()

            # Determine rebalancing points
            rebalance_dates = close_prices.vbt.wrapper.get_index_points(every=rebalance_period)
            
            # Portfolio optimization
            if optimizer == "efficient_frontier":
                pfo = vbt.PFO.from_pypfopt(
                    returns=returns,
                    optimizer=optimizer,
                    target="max_sharpe",  # Usando max_sharpe para Efficient Frontier
                    every=rebalance_period,
                    target_return=0.1,  # 10% retorno alvo anual
                    risk_free_rate=0.02,  # 2% taxa livre de risco
                    weight_bounds=(0.0, 1.0)  # 0-100% por ativo
                )
            elif optimizer == "efficient_cdar" or optimizer == "efficient_cvar":
                pfo = vbt.PFO.from_pypfopt(
                    returns=returns,
                    optimizer=optimizer,
                    target="min_cvar",  # Usando min_cvar para CVaR
                    every=rebalance_period,
                    beta=0.95  # Nível de confiança para CVaR
                )
            elif optimizer == "efficient_semivariance":
                pfo = vbt.PFO.from_pypfopt(
                    returns=returns,
                    optimizer=optimizer,
                    target="efficient_risk",  # Usando efficient_risk para Semivariance
                    every=rebalance_period,
                    target_volatility=0.1  # Volatilidade alvo de 10%
                )
            elif optimizer == "black_litterman" or optimizer == "bl":
                pfo = vbt.PFO.from_pypfopt(
                    returns=returns,
                    optimizer=optimizer,
                    target="bl_weights",  # Usando bl_weights para Black-Litterman
                    every=rebalance_period,
                    risk_aversion=3.0  # Aversão ao risco padrão
                )
            else:  # hierarchical_portfolio, cla e outros
                pfo = vbt.PFO.from_pypfopt(
                    returns=returns,
                    optimizer=optimizer,
                    target="optimize",  # Método padrão para HRP e CLA
                    every=rebalance_period
                )

            pf = pfo.simulate(data, freq="1D")
    
            # Exibir o valor total do portfólio ao longo do tempo
            total_value = pf.value

            # Exibir os retornos acumulados
            total_returns = pf.returns

            # Exibir capital inicial
            init_cash = pf.init_cash

            # Get the latest recommended allocation
            last_allocation = pfo.allocations.iloc[-1]
            
            # Output formatting
            output = "=== Portfolio Optimization ===\n"
            output += f"Analyzed symbols: {', '.join(symbols_list)}\n"
            output += f"Period: {period}\n"
            output += f"Optimization method: {optimizer}\n"
            output += f"Rebalancing frequency: {rebalance_period}\n\n"
            output += f"Initial Cash: {init_cash:.2f}\n"
            output += f"Final Cash: {total_value.iloc[-1]:.2f}\n"
            output += f"Total Returns: {total_returns.sum() * 100:.2f}%\n"
            output += "Recommended allocation:\n"
            for symbol, weight in last_allocation.items():
                emoji = "🟢" if weight > 0 else "🔴"
                output += f"- {symbol}: {weight * 100:.2f}% {emoji}\n"
            
            return output
        except Exception as e:
            return f"Error optimizing portfolio: {str(e)}"

if __name__ == "__main__":
    # Usage example
    optimizer = PortfolioOptimizerTool()
    symbols = "AAPL, MSFT, GOOGL, AMZN, PG, JNJ, KO, PEP, WMT, VZ, TSLA"  # Símbolos do mercado brasileiro
    result = optimizer._run(
        symbols, 
        period="1 year", 
        rebalance_period="1M", 
        optimizer="hierarchical_portfolio",
        tz="America/Sao_Paulo"
    )
    print(result)