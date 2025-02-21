import vectorbtpro as vbt
import warnings
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Test data - US Tech stocks
symbols_list = [
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "GOOGL",    # Alphabet
    "AMZN",     # Amazon
    "META"      # Meta
]
period = "6 months ago"

# Get data
print("Obtendo dados...")
data = vbt.YFData.pull(
    symbols_list,
    start=period
)

# Get close prices and calculate returns
print("\nProcessando retornos...")
close_prices = data.get("Close")
returns = close_prices.pct_change().dropna()

def test_hrp_config(linkage_method):
    try:
        config = {
            "returns": returns,
            "optimizer": "hierarchical_portfolio",
            "target": "optimize",
            "every": "1M",
            "optimizer_kwargs": {
                "linkage_method": linkage_method
            }
        }
        
        pfo = vbt.PFO.from_pypfopt(**config)
        pf = pfo.simulate(data, freq="1D")
        
        # Calculate risk metrics
        annual_return = pf.returns.mean() * 252 * 100  # Annualized return
        annual_volatility = pf.returns.std() * np.sqrt(252) * 100  # Annualized volatility
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        max_drawdown = pf.drawdown.max() * 100
        
        print(f"\nMétodo de Linkage: {linkage_method}")
        print("-" * 50)
        print("Métricas do Portfólio:")
        print(f"Retorno Anual: {annual_return:.2f}%")
        print(f"Volatilidade Anual: {annual_volatility:.2f}%")
        print(f"Índice Sharpe: {sharpe_ratio:.2f}")
        print(f"Drawdown Máximo: {max_drawdown:.2f}%")
        print("\nAlocações:")
        print("Primeira alocação:")
        for symbol, weight in pfo.allocations.iloc[0].items():
            print(f"{symbol:<8}: {weight*100:>6.2f}%")
        
        return True, {
            "return": annual_return,
            "volatility": annual_volatility,
            "sharpe": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "allocations": pfo.allocations.iloc[0].to_dict()
        }
    except Exception as e:
        return False, str(e)

# Test all available linkage methods
linkage_methods = [
    'single',
    'complete',
    'average',
    'ward'
]

print("\nTestando diferentes métodos de linkage do HRP:")
results = {}

for method in linkage_methods:
    success, result = test_hrp_config(method)
    if success:
        results[method] = result

# Compare methods
if results:
    print("\nComparação dos Métodos:")
    print("-" * 80)
    print(f"{'Método':<10} {'Retorno':<10} {'Volatilidade':<12} {'Sharpe':<10} {'MaxDD':<10}")
    print("-" * 80)
    
    for method, metrics in results.items():
        print(
            f"{method:<10} "
            f"{metrics['return']:>9.2f}% "
            f"{metrics['volatility']:>11.2f}% "
            f"{metrics['sharpe']:>9.2f} "
            f"{metrics['max_drawdown']:>9.2f}%"
        )
        
    # Analyze allocation differences
    print("\nVariação nas Alocações:")
    print("-" * 80)
    print(f"{'Ativo':<8} {'Mín':<8} {'Máx':<8} {'Variação':<8}")
    print("-" * 80)
    
    symbols = results['single']['allocations'].keys()
    for symbol in symbols:
        weights = [r['allocations'][symbol] for r in results.values()]
        min_weight = min(weights) * 100
        max_weight = max(weights) * 100
        weight_range = max_weight - min_weight
        print(f"{symbol:<8} {min_weight:>7.2f}% {max_weight:>7.2f}% {weight_range:>7.2f}%")
