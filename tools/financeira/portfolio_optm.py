from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import vectorbtpro as vbt
import pandas as pd

class PortfolioOptimizerInput(BaseModel):
    """Input schema for PortfolioOptimizer tool."""
    symbols: str = Field(..., description="Lista de símbolos dos ativos a serem otimizados (separados por vírgula), no formato que o yfinance aceita. Por exemplo: 'AAPL, BTC-USD, ^GSPC'")
    period: str = Field(
        default="1 year",
        description="Período de análise (1 day, 5 days, 1 month, 3 months, 6 months, 1 year, 2 years, 5 years, 10 years, ytd, max)"
    )
    rebalance_period: str = Field(
        default="1M", 
        description="Frequência de reequilíbrio do portfólio, ex: '1M' para mensal, '1W' para semanal.")

class PortfolioOptimizerTool(BaseTool):
    name: str = "Otimização de Portfólio"
    description: str = (
        "Ferramenta que obtém dados históricos do Yahoo Finance utilizando VectorBT Pro, "
        "calcula retornos dos ativos e otimiza a alocação do portfólio usando o modelo Hierarchical Risk Parity (HRP) via PyPortfolioOpt, "
        "com possibilidade de reequilíbrio periódico para melhor diversificação e robustez."
    )
    args_schema: Type[BaseModel] = PortfolioOptimizerInput

    def _run(self, symbols: str, period: str, rebalance_period: str) -> str:
        try:
            # Obtenção de dados históricos
            symbols_list = [s.strip().upper() for s in symbols.split(",")]
            data = vbt.YFData.pull(symbols_list, start=f"{period} ago", tz="UTC")
            close_prices = data.get("Close")
            close_prices = close_prices.resample('1D').last().ffill().bfill()

            # Cálculo de retornos diários
            returns = close_prices.pct_change().dropna()

            # Determinar os pontos de reequilíbrio
            rebalance_dates = close_prices.vbt.wrapper.get_index_points(every=rebalance_period)
            
            # Otimização de portfólio utilizando Hierarchical Risk Parity (HRP)
            pfo = vbt.PFO.from_pypfopt(
                returns=returns,
                optimizer="hrp",
                target="optimize",
                every=rebalance_period

            )
            
            # Obter a última alocação recomendada
            last_allocation = pfo.allocations.iloc[-1]
            
            # Formatação da saída
            output = "=== Otimização de Portfólio ===\n"
            output += f"Símbolos analisados: {', '.join(symbols_list)}\n"
            output += f"Período: {period}\n"
            output += f"Método de otimização: Hierarchical Risk Parity (HRP) via PyPortfolioOpt\n"
            output += f"Reequilíbrio a cada: {rebalance_period}\n\n"
            output += "Alocação recomendada:\n"
            for symbol, weight in last_allocation.items():
                output += f"- {symbol}: {weight * 100:.2f}%\n"
            
            return output
        except Exception as e:
            return f"Erro ao otimizar portfólio: {str(e)}"

if __name__ == "__main__":
    # Exemplo de uso
    optimizer = PortfolioOptimizerTool()
    symbols = "AAPL, BTC-USD, ^GSPC"
    result = optimizer._run(symbols, period="1 year", rebalance_period="1M")
    print(result)