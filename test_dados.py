import vectorbtpro as vbt

data = vbt.YFData.pull(["^GSPC", "^DJI", "^IXIC", "^BVSP", "^RUT", "^VIX"], start="1 year ago", tz="UTC")
print("Preços de fechamento")
fechamento = data.close.resample('1D').last().ffill().bfill()
#fechamento = fechamento.ffill().bfill()
print(fechamento)
close_price = data.get("Close")
close_price = close_price.ffill().bfill()
print(close_price)
#["COTN.L", "COFF.L", "COCO.L"]  ["GLD", "SLV", "COPX", "USO", "CORN", "CANE"]

sma = data.run("talib:SMA", timeperiod=20)
sma = sma.sma
print(sma)

sma = vbt.talib("SMA").run(fechamento, timeperiod=vbt.Default(20)).sma.iloc[-1]
print(sma)