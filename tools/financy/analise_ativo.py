from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import vectorbtpro as vbt
import numpy as np
import pandas as pd
from numba import njit
import yfinance as yf
from datetime import datetime, timedelta

class AssetAnalysisInput(BaseModel):
    """Input schema for asset analysis."""
    symbols: str = Field(..., description="List of asset symbols to be analyzed (comma separated). Example: 'AAPL, BTC-USD, ^GSPC'")
    period: str = Field(
        default="1 year",
        description="Analysis period (1 day, 5 days, 1 month, 3 months, 6 months, 1 year, 2 years, 5 years, 10 years, ytd, max)"
    )

class AssetDataTool(BaseTool):
    """Asset analysis tool using VectorBT Pro."""
    name: str = "Individual Asset Analysis with VectorBT Pro"
    description: str = (
        "Gets individual asset data including current prices, moving averages, volatility, correlations, and returns. "
        "Uses VectorBT Pro for advanced analysis of assets such as stocks, bonds, currencies, cryptocurrencies, ETFs, and commodities."
    )
    args_schema: Type[BaseModel] = AssetAnalysisInput

    def _run(self, symbols: str, period: str) -> str:
        """Runs the analysis for the provided assets."""
        try:
            symbols_list = [s.strip().upper() for s in symbols.split(",")]
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
            if yf_period:
                data = yf.download(symbols_list, period=yf_period)
            else:
                # fallback: tenta converter para data
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
            print(data)
            closes = data.get("Close")
            closes = closes.resample('1D').last().ffill().bfill()

            asset_stats = {}
            for symbol in symbols_list:
                asset_stats[symbol] = {
                    "prices": {
                        "current": closes[symbol].iloc[-1],
                        "ma_20": vbt.talib("SMA").run(closes[symbol], timeperiod=20).sma.iloc[-1],
                        "ma_200": vbt.talib("SMA").run(closes[symbol], timeperiod=200).sma.iloc[-1]
                    },
                    "returns": {
                        "1d": round(closes[symbol].pct_change(periods=2).iloc[-1] * 100, 2),
                        "1m": round(closes[symbol].pct_change(periods=21).iloc[-1] * 100, 2),
                        "3m": round(closes[symbol].pct_change(periods=63).iloc[-1] * 100, 2),
                        "6m": round(closes[symbol].pct_change(periods=126).iloc[-1] * 100, 2),
                        "1y": round(closes[symbol].pct_change(periods=252).iloc[-1] * 100, 2)
                    },
                    "volatility": round(closes[symbol].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100, 2)
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

                # Rank the top 3 highest, lowest, and neutral correlations
                correlations_sorted = correlations_df.iloc[-1].sort_values(ascending=False)

                neutral_threshold = 0.1
                top_3_positive = correlations_sorted[correlations_sorted > neutral_threshold].head(3)
                top_3_negative = correlations_sorted[correlations_sorted < -neutral_threshold].tail(3)
                neutral_correlations = correlations_sorted[abs(correlations_sorted) <= neutral_threshold].head(3)

            output = f"INDIVIDUAL ASSET DATA\n\n"
            output += f"Analyzed assets: {symbols}\n"
            output += f"Period: {period}\n"
            output += f"Date: {closes.index[-1].strftime('%Y-%m-%d')}\n\n"

            for symbol, stats in asset_stats.items():
                output += f"=== {symbol} ===\n"
                output += f"Current Price: {stats['prices']['current']}\n"
                output += f"20-day Moving Average: {stats['prices']['ma_20']}\n"
                output += f"200-day Moving Average: {stats['prices']['ma_200']}\n"
                output += f"20-day Volatility: {stats['volatility']}%\n"
                output += "Returns:\n"
                for period, value in stats['returns'].items():
                    output += f"- {period}: {value}%\n"
                output += "\n"

            if len(symbols_list) > 1:
                output += "🔗 **Correlation Rankings** 🔗\n"
                output += f"\n✅ **Top 3 Positive Correlations:**\n" + "\n".join([f"🔸 {k.replace('_', ' vs ')}: {v:.2f}" for k, v in top_3_positive.items()]) + "\n"
                output += f"\n❌ **Top 3 Negative Correlations:**\n" + "\n".join([f"🔻 {k.replace('_', ' vs ')}: {v:.2f}" for k, v in top_3_negative.items()]) + "\n"
                output += f"\n⚖️ **Top 3 Neutral Correlations:**\n" + "\n".join([f"⚪ {k.replace('_', ' vs ')}: {v:.2f}" for k, v in neutral_correlations.items()]) + "\n"

            return output

        except Exception as e:
            return f"Error getting asset data: {str(e)}"

if __name__ == "__main__":
    # Creating a tool instance
    tool = AssetDataTool()

    # Creating an input object with desired parameters
    input_data = AssetAnalysisInput(symbols="AAPL, BTC-USD, ^GSPC", period="1 year")

    # Running the tool and printing the result
    result = tool._run(input_data.symbols, input_data.period)
    print(result)
