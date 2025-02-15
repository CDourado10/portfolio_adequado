import vectorbtpro as vbt
import numpy as np
import pandas as pd
import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from numba import njit

# Function to load the sector map
def _load_sector_map(json_path="crews/consultor/setores_economicos/sectors_map.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading sector map: {e}")
        return {}

SECTOR_MAP = _load_sector_map()

# Input schema for the tool
class MacroDataInput(BaseModel):
    """Input schema for macroeconomic data."""
    sectors: str = Field(
        default="all",
        description="Sectors to analyze: TECH, FIN, HEALTH, ENERGY, CONS, INDUSTRIAL, MATERIALS, UTILITIES, REAL_ESTATE, INDEXES, CURRENCIES, CRYPTO, ETFS, COMMODITIES."
    )
    period: str = Field(
        default="1 year",
        description="Analysis period (1 day, 5 days, 1 month, 3 months, 6 months, 1 year, 2 years, 5 years, 10 years, ytd, max)"
    )

# CrewAI tool class
class MacroDataTool(BaseTool):
    """Macroeconomic analysis tool using VectorBT Pro."""
    name: str = "Macroeconomic Data with VectorBT Pro"
    description: str = (
        "Gets macroeconomic data for selected assets including current prices, moving averages, volatility, "
        "correlations and returns. Uses VectorBT Pro for advanced analysis of sectors such as technology, financial, healthcare, "
        "energy, consumer, industrial, materials, utilities, real estate, indices, currencies, cryptocurrencies, ETFs and commodities. "
        "Creates an index for each sector and calculates returns, volatility, current prices, moving averages and correlations between sectors. "
        "The calculation is performed using geometric mean."
    )
    args_schema: Type[BaseModel] = MacroDataInput

    def _run(self, sectors: str = "all", period: str = "1 year") -> str:
        """Runs macroeconomic analysis based on provided sectors."""
        try:
            # Determine sectors to analyze
            selected_sectors = SECTOR_MAP["sectors"].keys() if sectors == "all" else [s.strip().upper() for s in sectors.split(",")]
            selected_sectors_list = list(selected_sectors)

            result = {}
            sector_indices = {}

            for sector in selected_sectors:
                if sector in SECTOR_MAP["sectors"]:
                    symbols_list = SECTOR_MAP["sectors"][sector]["symbols"]
                    data = vbt.YFData.pull(symbols_list, start=f"{period} ago", tz="UTC")
                    sector_data = data.get("Close")
                    volume_data = data.get("Volume")
                    
                    # Resample e preenche dados faltantes
                    sector_data = sector_data.resample('1D').last().ffill().bfill()
                    if volume_data is not None:
                        volume_data = volume_data.resample('1D').last().ffill().bfill()
                    
                    def normalize_volume(volume_series, window=20, n_std=2):
                        """
                        Normaliza volume usando z-score com janela móvel e winsorização
                        """
                        if volume_series.isnull().all():
                            return pd.Series(1.0, index=volume_series.index)
                            
                        # Calcula média e desvio móveis
                        rolling_mean = volume_series.rolling(window=window, min_periods=1).mean()
                        rolling_std = volume_series.rolling(window=window, min_periods=1).std()
                        
                        # Calcula z-score
                        z_score = (volume_series - rolling_mean) / rolling_std
                        
                        # Aplica winsorização
                        z_score = z_score.clip(-n_std, n_std)
                        
                        # Normaliza para [0.1, 1] para evitar pesos zero
                        normalized = (z_score - z_score.min()) / (z_score.max() - z_score.min()) * 0.9 + 0.1
                        
                        return normalized
                    
                    # 1. Normaliza os preços para começar em 1
                    normalized_data = sector_data.div(sector_data.iloc[0])
                    
                    # 2. Tratamento especial para volumes
                    if volume_data is not None and not volume_data.empty:
                        # Identifica ativos com e sem volume
                        has_volume = ~volume_data.isnull().all()
                        
                        # Para ativos com volume, aplica normalização
                        normalized_volume = pd.DataFrame(index=volume_data.index)
                        for col in volume_data.columns:
                            if has_volume[col]:
                                normalized_volume[col] = normalize_volume(volume_data[col])
                            else:
                                # Para ativos sem volume, usa peso médio dos ativos com volume
                                normalized_volume[col] = 1.0  # Peso neutro inicial
                        
                        # Ajusta os pesos para manter a influência proporcional
                        n_with_volume = has_volume.sum()
                        n_without_volume = len(has_volume) - n_with_volume
                        
                        if n_with_volume > 0:
                            # Calcula peso base para ativos sem volume
                            avg_weight = 1.0 / len(has_volume)
                            
                            # Ajusta pesos dos ativos com volume
                            volume_weight_sum = n_with_volume / len(has_volume)
                            for col in normalized_volume.columns:
                                if has_volume[col]:
                                    normalized_volume[col] *= volume_weight_sum
                                else:
                                    normalized_volume[col] = avg_weight
                    else:
                        # Se não houver dados de volume, usa pesos iguais
                        normalized_volume = pd.DataFrame(1.0, index=sector_data.index, columns=sector_data.columns)
                        normalized_volume = normalized_volume.div(normalized_volume.sum(axis=1), axis=0)
                    
                    # 3. Calcula os pesos finais
                    volume_weights = normalized_volume.div(normalized_volume.sum(axis=1), axis=0)
                    
                    # 4. Calcula os retornos logarítmicos
                    log_returns = np.log(normalized_data)
                    
                    # 5. Aplica os pesos aos retornos logarítmicos e calcula a média geométrica
                    weighted_log_returns = (log_returns * volume_weights)
                    sector_index = pd.Series(
                        data=np.exp(weighted_log_returns.sum(axis=1)),
                        index=normalized_data.index,
                        name=sector
                    )
                    
                    # 6. Normaliza o índice final para começar em 1
                    sector_index = sector_index / sector_index.iloc[0]

                    result[sector] = {
                        "prices": {
                            "current": sector_index.iloc[-1],
                            "ma_20": vbt.talib("SMA").run(sector_index, timeperiod=20).sma.iloc[-1],
                            "ma_200": vbt.talib("SMA").run(sector_index, timeperiod=200).sma.iloc[-1]
                        },
                        "returns": {
                            "1d": round(sector_index.pct_change().iloc[-1] * 100, 2),
                            "1m": round(sector_index.pct_change(periods=21).iloc[-1] * 100, 2),
                            "3m": round(sector_index.pct_change(periods=63).iloc[-1] * 100, 2),
                            "6m": round(sector_index.pct_change(periods=126).iloc[-1] * 100, 2),
                            "1y": round(sector_index.pct_change(periods=252).iloc[-1] * 100, 2)
                        },
                        "volatility": round(sector_index.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100, 2)
                    }

                    sector_indices[sector] = sector_index

            # Optimized function to calculate correlation
            @njit
            def corr_meta_nb(from_i, to_i, col, a, b):
                a_window = a[from_i:to_i, col]
                b_window = b[from_i:to_i, col]
                return np.corrcoef(a_window, b_window)[1, 0]

            # Correlation Matrix between sectors
            correlations_df = None
            if len(selected_sectors_list) > 1:
                sector_pairs = [(s1, s2) for i, s1 in enumerate(sector_indices.keys()) for s2 in list(sector_indices.keys())[i+1:]]
                correlation_results = {}

                for s1, s2 in sector_pairs:
                    correlation_results[f"{s1}_{s2}"] = vbt.pd_acc.rolling_apply(
                        14, 
                        corr_meta_nb, 
                        vbt.Rep("a"),
                        vbt.Rep("b"),
                        broadcast_named_args=dict(a=sector_indices[s1], b=sector_indices[s2])
                    )

                correlations_df = pd.DataFrame(correlation_results).ffill().bfill()

                # Rank the top 3 highest, lowest, and neutral correlations
                correlations_sorted = correlations_df.iloc[-1].sort_values(ascending=False)

                neutral_threshold = 0.1
                top_3_positive = correlations_sorted[correlations_sorted > neutral_threshold].head(3)
                top_3_negative = correlations_sorted[correlations_sorted < -neutral_threshold].tail(3)
                neutral_correlations = correlations_sorted[abs(correlations_sorted) <= neutral_threshold].head(3)

            # Output construction
            output = f"📊 **MACROECONOMIC DATA** 📊\n\n"
            output += f"📅 **Period**: {period}\n"
            output += f"📆 **Analysis Date**: {sector_data.index[-1].strftime('%Y-%m-%d')}\n\n"

            for sector, data in result.items():
                output += f"🔹 **Sector: {sector}**\n"
                output += f"💰 Current Index: {data['prices']['current']}\n"
                output += f"📈 MA20: {data['prices']['ma_20']}\n"
                output += f"📉 MA200: {data['prices']['ma_200']}\n"
                output += f"📊 Volatility: {data['volatility']}%\n"
                output += "📊 **Returns:**\n"
                for per, val in data["returns"].items():
                    output += f"   🔸 {per}: {val}%\n"
                output += "\n"

            if correlations_df is not None:
                output += "🔗 **Correlation Rankings** 🔗\n"
                output += f"\n✅ **Top 3 Positive Correlations:**\n" + "\n".join([f"🔸 {k}: {v:.2f}" for k, v in top_3_positive.items()]) + "\n"
                output += f"\n❌ **Top 3 Negative Correlations:**\n" + "\n".join([f"🔻 {k}: {v:.2f}" for k, v in top_3_negative.items()]) + "\n"
                output += f"\n⚖️ **Top 3 Neutral Correlations:**\n" + "\n".join([f"⚪ {k}: {v:.2f}" for k, v in neutral_correlations.items()]) + "\n"

            return output

        except Exception as e:
            return f"❌ Error getting macroeconomic data: {str(e)}"

# Test block
if __name__ == "__main__":
    # Creating a tool instance
    tool = MacroDataTool()

    # Creating input for macroeconomic analysis
    input_data = MacroDataInput(sectors="all", period="1 year")

    # Running the tool
    result = tool._run("all", "1 year")

    # Displaying the result
    print(result)
