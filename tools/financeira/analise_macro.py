import vectorbtpro as vbt
import numpy as np
import pandas as pd
import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from numba import njit

# Função para carregar o mapa de setores
def _load_sector_map(json_path="consultor/setores_economicos/sectors_map.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Erro ao carregar o mapa de setores: {e}")
        return {}

SECTOR_MAP = _load_sector_map()

# Schema de entrada para a ferramenta
class MacroDataInput(BaseModel):
    """Input schema para dados macroeconômicos."""
    sectors: str = Field(
        default="all",
        description="Setores a serem analisados: TECH, FIN, HEALTH, ENERGY, CONS, INDUSTRIAL, MATERIALS, UTILITIES, REAL_ESTATE, INDEXES, CURRENCIES, CRYPTO, ETFS, COMMODITIES."
    )
    period: str = Field(
        default="1 year",
        description="Período de análise (1 day, 5 days, 1 month, 3 months, 6 months, 1 year, 2 years, 5 years, 10 years, ytd, max)"
    )



# Classe da ferramenta CrewAI
class MacroDataTool(BaseTool):
    """Ferramenta de análise macroeconômica utilizando VectorBT Pro."""
    name: str = "Dados Macroeconômicos com VectorBT Pro"
    description: str = (
        "Obtém dados macroeconômicos dos ativos selecionados incluindo preços atuais, médias móveis, volatilidade, "
        "correlações e retornos. Utiliza VectorBT Pro para análise avançada de setores como tecnologia, financeiro, saúde, "
        "energia, consumo, industrial, materiais, utilidades, imobiliário, índices, moedas, criptomoedas, ETFs e commodities."
        "Cria um índice para cada setor e calcula os retornos, volatilidade, preços atuais, médias móveis e correlações entre os setores. "
        "O cálculo realizado é com a média geometrica."
    )
    args_schema: Type[BaseModel] = MacroDataInput

    def _run(self, sectors: str = "all", period: str = "1 year") -> str:
        """Executa a análise macroeconômica baseada nos setores fornecidos."""
        try:
            # Determinar os setores a serem analisados
            selected_sectors = SECTOR_MAP["sectors"].keys() if sectors == "all" else [s.strip().upper() for s in sectors.split(",")]
            selected_sectors_list = list(selected_sectors)

            result = {}
            indices_setores = {}

            for sector in selected_sectors:
                if sector in SECTOR_MAP["sectors"]:
                    symbols_list = SECTOR_MAP["sectors"][sector]["symbols"]
                    data = vbt.YFData.pull(symbols_list, start=f"{period} ago", tz="UTC")
                    sector_data = data.get("Close")
                    sector_data = sector_data.resample('1D').last().ffill().bfill()
                    setor_indice = np.exp(np.log(sector_data).mean(axis=1))

                    result[sector] = {
                        "precos": {
                            "atual": setor_indice.iloc[-1],
                            "ma_20": vbt.talib("SMA").run(setor_indice, timeperiod=20).sma.iloc[-1],
                            "ma_200": vbt.talib("SMA").run(setor_indice, timeperiod=200).sma.iloc[-1]
                        },
                        "retornos": {
                            "1d": round(setor_indice.pct_change().iloc[-1] * 100, 2),
                            "1m": round(setor_indice.pct_change(periods=21).iloc[-1] * 100, 2),
                            "3m": round(setor_indice.pct_change(periods=63).iloc[-1] * 100, 2),
                            "6m": round(setor_indice.pct_change(periods=126).iloc[-1] * 100, 2),
                            "1a": round(setor_indice.pct_change(periods=252).iloc[-1] * 100, 2)
                        },
                        "volatilidade": round(setor_indice.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100, 2)
                    }

                    indices_setores[sector] = setor_indice

            # Função otimizada para calcular correlação
            @njit
            def corr_meta_nb(from_i, to_i, col, a, b):
                a_window = a[from_i:to_i, col]
                b_window = b[from_i:to_i, col]
                return np.corrcoef(a_window, b_window)[1, 0]

            # Matriz de Correlação entre setores
            correlations_df = None
            if len(selected_sectors_list) > 1:
                sector_pairs = [(s1, s2) for i, s1 in enumerate(indices_setores.keys()) for s2 in list(indices_setores.keys())[i+1:]]
                correlation_results = {}

                for s1, s2 in sector_pairs:
                    correlation_results[f"{s1}_{s2}"] = vbt.pd_acc.rolling_apply(
                        14, 
                        corr_meta_nb, 
                        vbt.Rep("a"),
                        vbt.Rep("b"),
                        broadcast_named_args=dict(a=indices_setores[s1], b=indices_setores[s2])
                    )

                correlations_df = pd.DataFrame(correlation_results).ffill().bfill()

                # Rankear as 3 maiores, menores e neutras correlações
                correlations_sorted = correlations_df.iloc[-1].sort_values(ascending=False)

                neutral_threshold = 0.1
                top_3_positive = correlations_sorted[correlations_sorted > neutral_threshold].head(3)
                top_3_negative = correlations_sorted[correlations_sorted < -neutral_threshold].tail(3)
                neutral_correlations = correlations_sorted[abs(correlations_sorted) <= neutral_threshold].head(3)

            # Construção da saída
            output = f"📊 **DADOS MACROECONÔMICOS** 📊\n\n"
            output += f"📅 **Período**: {period}\n"
            output += f"📆 **Data da Análise**: {sector_data.index[-1].strftime('%Y-%m-%d')}\n\n"

            for sector, data in result.items():
                output += f"🔹 **Setor: {sector}**\n"
                output += f"💰 Índice Atual: {data['precos']['atual']}\n"
                output += f"📈 MA20: {data['precos']['ma_20']}\n"
                output += f"📉 MA200: {data['precos']['ma_200']}\n"
                output += f"📊 Volatilidade: {data['volatilidade']}%\n"
                output += "📊 **Retornos:**\n"
                for per, val in data["retornos"].items():
                    output += f"   🔸 {per}: {val}%\n"
                output += "\n"

            if correlations_df is not None:
                output += "🔗 **Ranking de Correlações** 🔗\n"
                output += f"\n✅ **Top 3 Correlações Positivas:**\n" + "\n".join([f"🔸 {k}: {v:.2f}" for k, v in top_3_positive.items()]) + "\n"
                output += f"\n❌ **Top 3 Correlações Negativas:**\n" + "\n".join([f"🔻 {k}: {v:.2f}" for k, v in top_3_negative.items()]) + "\n"
                output += f"\n⚖️ **Top 3 Correlações Neutras:**\n" + "\n".join([f"⚪ {k}: {v:.2f}" for k, v in neutral_correlations.items()]) + "\n"

            return output

        except Exception as e:
            return f"❌ Erro ao obter dados macroeconômicos: {str(e)}"

# Bloco de teste
if __name__ == "__main__":
    # Criando uma instância da ferramenta
    tool = MacroDataTool()

    # Criando entrada para análise macroeconômica
    input_data = MacroDataInput(sectors="all", period="1 year")

    # Executando a ferramenta
    result = tool._run("all", "1 year")

    # Exibindo o resultado
    print(result)
