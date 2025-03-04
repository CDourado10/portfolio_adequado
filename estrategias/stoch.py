import vectorbtpro as vbt

data = vbt.BinanceData.pull('BTCUSDT', start='3 weeks ago', timeframe='1m')

close_price = data.get('close')
print(close_price)
windows = list(range(8, 21))
wtypes = ["simple", "exp", "wilder"]
lower_ths = list(range(20, 31))
upper_ths = list(range(70, 81))

rsi = vbt.RSI.run(
    close_price, 
    window=windows,
    wtypes=wtypes,
    param_product=True
    )

lower_ths_prod, upper_ths_prod = zip(*product(lower_ths, upper_ths))

lower_th_index = vbt.Param(lower_ths_prod, name='lower_th')  
entries = rsi.rsi_crossed_below(lower_th_index)

upper_th_index = vbt.Param(upper_ths_prod, name='upper_th')
exits = rsi.rsi_crossed_above(upper_th_index)

clean_entries, clean_exits = entries.vbt.signals.clean(exits)  

def plot_rsi(rsi, entries, exits):
    fig = rsi.plot()  
    entries.vbt.signals.plot_as_entries(rsi.rsi, fig=fig)  
    exits.vbt.signals.plot_as_exits(rsi.rsi, fig=fig)  

    return fig

#plot_rsi(rsi, clean_entries, clean_exits).show()

pf = vbt.Portfolio.from_signals(
    close=close_price, 
    entries=clean_entries, 
    exits=clean_exits,
    size=100,
    size_type='value',
    init_cash='auto'
)

print(pf)

pf.plot(settings=dict(bm_returns=True)).show()

stats_df = pf.stats([
    'total_return', 
    'total_trades', 
    'win_rate', 
    'expectancy'
], agg_func=None)  

print(stats_df)

stats_df['Expectancy'].groupby('rsi_window').mean()

stats_df.sort_values(by='Expectancy', ascending=False).head()

pf[(22, 80, 20, "wilder")].plot_value().show()