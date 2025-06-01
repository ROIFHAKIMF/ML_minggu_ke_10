import yfinance as yf

start = "2018-01-01"
end = "2025-05-31"

symbols = {
    "btc.csv": "BTC-USD",
    "gold.csv": "GC=F",
    "sp500.csv": "^GSPC",
    "dxy.csv": "DX-Y.NYB",
    "tnx.csv": "^TNX",
    "bito.csv": "BITO",
    "vix.csv": "^VIX"
}

for fname, ticker in symbols.items():
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(fname)
