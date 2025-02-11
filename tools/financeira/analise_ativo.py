from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import vectorbtpro as vbt
import numpy as np
import pandas as pd
from numba import njit

class AnaliseAtivoInput(BaseModel):
    """Input schema para análise de ativos."""
    symbols: str = Field(..., description="Lista de símbolos dos ativos a serem analisados (separados por vírgula). Exemplo: 'AAPL, BTC-USD, ^GSPC'")
    period: str = Field(
        default="1 year",
        description="Período de análise (1 day, 5 days, 1 month, 3 months, 6 months, 1 year, 2 years, 5 years, 10 years, ytd, max)"
    )

class AtivoDataTool(BaseTool):
    """Ferramenta de análise de ativos usando VectorBT Pro."""
    name: str = "Análise de Ativos Individuais com VectorBT Pro"
    description: str = (
        "Obtém dados de ativos individuais incluindo preços atuais, médias móveis, volatilidade, correlações e retornos. "
        "Utiliza VectorBT Pro para análise avançada de ativos como ações, títulos, moedas, criptomoedas, ETFs e commodities."
    )
    args_schema: Type[BaseModel] = AnaliseAtivoInput

    def _run(self, symbols: str, period: str) -> str:
        """Executa a análise dos ativos fornecidos."""
        try:
            symbols_list = [s.strip().upper() for s in symbols.split(",")]
            data = vbt.YFData.pull(symbols_list, start=f"{period} ago", tz="UTC")
            closes = data.get("Close")
            closes = closes.resample('1D').last().ffill().bfill()

            asset_stats = {}
            for symbol in symbols_list:
                asset_stats[symbol] = {
                    "precos": {
                        "atual": closes[symbol].iloc[-1],
                        "ma_20": vbt.talib("SMA").run(closes[symbol], timeperiod=20).sma.iloc[-1],
                        "ma_200": vbt.talib("SMA").run(closes[symbol], timeperiod=200).sma.iloc[-1]
                    },
                    "retornos": {
                        "1d": round(closes[symbol].pct_change(periods=2).iloc[-1] * 100, 2),
                        "1m": round(closes[symbol].pct_change(periods=21).iloc[-1] * 100, 2),
                        "3m": round(closes[symbol].pct_change(periods=63).iloc[-1] * 100, 2),
                        "6m": round(closes[symbol].pct_change(periods=126).iloc[-1] * 100, 2),
                        "1a": round(closes[symbol].pct_change(periods=252).iloc[-1] * 100, 2)
                    },
                    "volatilidade": round(closes[symbol].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100, 2)
                }

            if len(symbols_list) > 1:
                @njit
                def corr_meta_nb(from_i, to_i, col, a, b, *args):
                    a_window = a[from_i:to_i, col]
                    b_window = b[from_i:to_i, col]
                    if len(a_window) < 2 or len(b_window) < 2:
                        return np.nan
                    return np.corrcoef(a_window, b_window)[0, 1]

                symbols_pairs = [(s1, s2) for i, s1 in enumerate(symbols_list) for s2 in symbols_list[i+1:]]
                correlation_results = {}
                for s1, s2 in symbols_pairs:
                    correlation_results[f"{s1}_{s2}"] = vbt.pd_acc.rolling_apply(
                        14, 
                        corr_meta_nb, 
                        vbt.Rep("a"),
                        vbt.Rep("b"),
                        broadcast_named_args=dict(a=closes[s1], b=closes[s2])
                    )

                correlations_df = pd.DataFrame(correlation_results).ffill().bfill()

            output = f"DADOS DOS ATIVOS INDIVIDUAIS\n\n"
            output += f"Ativos analisados: {symbols}\n"
            output += f"Período: {period}\n"
            output += f"Data: {closes.index[-1].strftime('%Y-%m-%d')}\n\n"

            for symbol, stats in asset_stats.items():
                output += f"=== {symbol} ===\n"
                output += f"Preço Atual: {stats['precos']['atual']}\n"
                output += f"Média Móvel de 20 dias: {stats['precos']['ma_20']}\n"
                output += f"Média Móvel de 200 dias: {stats['precos']['ma_200']}\n"
                output += f"Volatilidade de 20 dias: {stats['volatilidade']}%\n"
                output += "Retornos:\n"
                for period, value in stats['retornos'].items():
                    output += f"- {period}: {value}%\n"
                output += "\n"

            if len(symbols_list) > 1:
                output += "=== MATRIZ DE CORRELAÇÃO ===\n"
                output += "Período: 20 dias\n"
                for col in correlations_df.columns:
                    output += f"Correlação entre {col.split('_')[0]} e {col.split('_')[1]}: {correlations_df[col].iloc[-1]:.2f}\n"

            return output

        except Exception as e:
            return f"Erro ao obter dados dos ativos: {str(e)}"

if __name__ == "__main__":
    # Criando uma instância da ferramenta
    tool = AtivoDataTool()

    # Criando um objeto de entrada com os parâmetros desejados
    input_data = AnaliseAtivoInput(symbols="AAPL, BTC-USD, ^GSPC", period="1 year")

    # Executando a ferramenta e imprimindo o resultado
    result = tool._run(input_data.symbols, input_data.period)
    print(result)
