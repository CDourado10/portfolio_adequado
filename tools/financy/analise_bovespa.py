import vectorbtpro as vbt
import numpy as np
import pandas as pd
import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from numba import njit

def _load_sector_map(json_path="crews/consultor/setores_economicos/bovespa_sectors_map.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Erro ao carregar mapa de setores: {e}")
        return {}

SECTOR_MAP = _load_sector_map()

class BovespaDataInput(BaseModel):
    """Schema de entrada para dados do Bovespa."""
    setores: str = Field(
        default="all",
        description="Setores para análise: FINANCEIRO, CONSUMO, ENERGIA, MATERIAIS, UTILIDADES, IMOBILIARIO, SAUDE, TELECOM, INDUSTRIAL, TECH, INDICES"
    )
    periodo: str = Field(
        default="1 ano",
        description="Período de análise (3 meses, 6 meses, 1 ano, 2 anos, 5 anos, 10 anos, ytd, max)"
    )

class BovespaDataTool(BaseTool):
    """Ferramenta de análise do mercado brasileiro usando VectorBT Pro."""
    name: str = "Análise de Dados do Bovespa com VectorBT Pro"
    description: str = (
        "Obtém dados do mercado brasileiro para ativos selecionados incluindo preços atuais, médias móveis, "
        "volatilidade, correlações e retornos. Usa VectorBT Pro para análise avançada dos setores do Bovespa como "
        "financeiro, consumo, energia, materiais básicos, utilidades públicas, imobiliário, saúde, telecomunicações, "
        "industrial e tecnologia. Cria um índice para cada setor e calcula retornos, volatilidade, preços atuais, "
        "médias móveis e correlações entre setores. O cálculo é realizado usando média geométrica."
    )
    args_schema: Type[BaseModel] = BovespaDataInput

    def _run(self, setores: str = "all", periodo: str = "1 ano") -> str:
        """Executa análise do mercado brasileiro baseado nos setores fornecidos."""
        try:
            # Converte período para inglês para o Yahoo Finance
            periodo_map = {
                "3 meses": "3 months",
                "6 meses": "6 months",
                "1 ano": "1 year",
                "2 anos": "2 years",
                "5 anos": "5 years",
                "10 anos": "10 years",
                "ytd": "ytd",
                "max": "max"
            }
            period_en = periodo_map.get(periodo, periodo)

            # Determina setores para análise
            selected_sectors = SECTOR_MAP["sectors"].keys() if setores == "all" else [s.strip().upper() for s in setores.split(",")]
            selected_sectors_list = list(selected_sectors)

            result = {}
            sector_indices = {}
            
            # Análise por setor
            for sector in selected_sectors:
                if sector in SECTOR_MAP["sectors"]:
                    symbols_list = SECTOR_MAP["sectors"][sector]["symbols"]
                    try:
                        # Tenta obter dados para cada símbolo individualmente
                        valid_data = []
                        valid_volume = []
                        for symbol in symbols_list:
                            try:
                                # Configura o download com parâmetros específicos
                                # Mapeamento de períodos para o formato aceito pelo yfinance
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
                                user_period = period_en.strip().lower()
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
                                        raise ValueError(f"Período '{period_en}' inválido para análise.")
                                close_data = data.get('Close')
                                volume_data = data.get('Volume')  # Obtendo dados de volume
                                
                                # Verifica se os dados são válidos usando métodos específicos do pandas
                                if (close_data is not None and 
                                    not close_data.empty and 
                                    not close_data.isnull().all().all() and
                                    volume_data is not None and
                                    not volume_data.empty):
                                    valid_data.append(close_data)
                                    valid_volume.append(volume_data)  # Armazenando volume
                                else:
                                    print(f"⚠️ Dados vazios para {symbol}")
                            except Exception as symbol_error:
                                print(f"⚠️ Erro ao obter dados para {symbol}: {str(symbol_error)}")
                                continue

                        if not valid_data:
                            print(f"⚠️ Nenhum dado válido encontrado para o setor {sector}")
                            continue

                        # Combina os dados válidos
                        sector_data = pd.concat(valid_data, axis=1)
                        volume_data = pd.concat(valid_volume, axis=1)
                        
                        if sector_data.empty:
                            print(f"⚠️ Dados combinados vazios para o setor {sector}")
                            continue

                        # Preenche dados faltantes e normaliza
                        sector_data = sector_data.ffill().bfill()
                        volume_data = volume_data.ffill().bfill()
                        
                        # 1. Normaliza os preços para começar em 1
                        normalized_data = sector_data.div(sector_data.iloc[0])
                        
                        # 2. Normalização robusta do volume usando z-score com janela móvel
                        def normalize_volume(volume_series, window=20, n_std=2):
                            """
                            Normaliza volume usando z-score com janela móvel e winsorização
                            
                            Args:
                                volume_series: Série temporal de volume
                                window: Tamanho da janela móvel
                                n_std: Número de desvios padrão para winsorização
                            """
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
                        
                        # Aplica normalização do volume para cada ativo
                        normalized_volume = volume_data.apply(normalize_volume)
                        
                        # 3. Calcula os pesos baseados no volume normalizado
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

                        # Verifica se temos dados suficientes
                        if len(sector_index) < 20:
                            continue

                        # Cálculo de métricas usando métodos do pandas
                        metrics = {}
                        ma20 = sector_index.rolling(window=20).mean()
                        ma200 = sector_index.rolling(window=200).mean()
                        returns = sector_index.pct_change()
                        volatility = returns.rolling(window=20).std() * np.sqrt(252)
                        drawdown = (sector_index / sector_index.expanding().max() - 1).min()

                        metrics = {
                            "precos": {
                                "atual": float(sector_index.iloc[-1]),
                                "ma_20": float(ma20.iloc[-1]),
                                "ma_200": float(ma200.iloc[-1]) if len(sector_index) >= 200 else None
                            },
                            "retornos": {
                                "1d": float(returns.iloc[-1] * 100),
                                "1m": float(sector_index.pct_change(periods=21).iloc[-1] * 100),
                                "3m": float(sector_index.pct_change(periods=63).iloc[-1] * 100),
                                "6m": float(sector_index.pct_change(periods=126).iloc[-1] * 100),
                                "1a": float(sector_index.pct_change(periods=252).iloc[-1] * 100)
                            },
                            "volatilidade": float(volatility.iloc[-1] * 100),
                            "drawdown": float(abs(drawdown) * 100)
                        }

                        result[sector] = metrics
                        sector_indices[sector] = sector_index

                    except Exception as sector_error:
                        print(f"❌ Erro ao processar setor {sector}: {str(sector_error)}")
                        continue

            if not result:
                return "❌ Não foi possível obter dados para nenhum setor."

            # Função otimizada para cálculo de correlação
            @njit
            def corr_meta_nb(from_i, to_i, col, a, b):
                a_window = a[from_i:to_i, col]
                b_window = b[from_i:to_i, col]
                return np.corrcoef(a_window, b_window)[1, 0]

            # Matriz de correlação entre setores
            correlations_df = None
            if len(selected_sectors_list) > 1:
                sector_pairs = [(s1, s2) for i, s1 in enumerate(sector_indices.keys()) 
                              for s2 in list(sector_indices.keys())[i+1:]]
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

                # Ranking das correlações
                correlations_sorted = correlations_df.iloc[-1].sort_values(ascending=False)

                neutral_threshold = 0.1
                top_3_positive = correlations_sorted[correlations_sorted > neutral_threshold].head(3)
                top_3_negative = correlations_sorted[correlations_sorted < -neutral_threshold].tail(3)
                neutral_correlations = correlations_sorted[abs(correlations_sorted) <= neutral_threshold].head(3)

            # Construção da saída
            output = f"📊 **ANÁLISE DO MERCADO BRASILEIRO** 📊\n\n"
            output += f"📅 **Período**: {periodo}\n"
            output += f"📆 **Data da Análise**: {sector_data.index[-1].strftime('%d/%m/%Y')}\n\n"

            for sector, data in result.items():
                sector_name = SECTOR_MAP["sectors"][sector]["name"]
                output += f"🔹 **Setor: {sector_name}**\n"
                output += f"  🔹 **Índice**\n"
                output += f"    💰 Índice Atual: {data['precos']['atual']:.2f}\n"
                output += f"    📈 MM20: {data['precos']['ma_20']:.2f}\n"
                output += f"    📉 MM200: {data['precos']['ma_200']:.2f}\n"
                output += f"    📊 Volatilidade: {data['volatilidade']}%\n"
                output += f"    📉 Drawdown Máximo: {data['drawdown']}%\n"
                output += "    📊 **Retornos:**\n"
                for per, val in data["retornos"].items():
                    emoji = "🟢" if val > 0 else "🔴"
                    output += f"      {emoji} {per}: {val}%\n"
                output += "\n"

            if correlations_df is not None:
                output += "🔗 **Ranking de Correlações** 🔗\n"
                output += f"\n✅ **Top 3 Correlações Positivas:**\n" + "\n".join([f"🔸 {k}: {v:.2f}" for k, v in top_3_positive.items()]) + "\n"
                output += f"\n❌ **Top 3 Correlações Negativas:**\n" + "\n".join([f"🔻 {k}: {v:.2f}" for k, v in top_3_negative.items()]) + "\n"
                output += f"\n⚖️ **Top 3 Correlações Neutras:**\n" + "\n".join([f"⚪ {k}: {v:.2f}" for k, v in neutral_correlations.items()]) + "\n"

            return output

        except Exception as e:
            return f"❌ Erro ao obter dados do mercado brasileiro: {str(e)}"

# Bloco de teste
if __name__ == "__main__":
    # Criando uma instância da ferramenta
    tool = BovespaDataTool()

    # Criando input para análise do mercado brasileiro
    input_data = BovespaDataInput(setores="all", periodo="1 ano")

    # Executando a ferramenta
    result = tool._run("all", "1 ano")

    # Exibindo o resultado
    print(result)