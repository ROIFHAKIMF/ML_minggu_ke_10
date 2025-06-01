import pandas as pd
import matplotlib.pyplot as plt

# Load data BTC
df = pd.read_csv('gold.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# Filter data 2018 - 2024
df_filtered = df[(df.index >= '2018-01-01') & (df.index <= '2025-05-31')]
df_filtered['EMA100'] = df_filtered['Close'].ewm(span=100, adjust=False).mean()
df_filtered['EMA200'] = df_filtered['Close'].ewm(span=200, adjust=False).mean()

# Plot
plt.figure(figsize=(14,7))
plt.plot(df_filtered.index, df_filtered['Close'], label='Harga GOLD', color='orange', linewidth=2)
plt.plot(df_filtered.index, df_filtered['EMA100'], label='EMA 100', color='red', linestyle='--')
plt.plot(df_filtered.index, df_filtered['EMA200'], label='EMA 200', color='blue', linestyle='--')
plt.title('Harga Gold + 100, 200 (2018 - 2024)')
plt.xlabel('Tahun')
plt.ylabel('Harga (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gold_chart.png")
plt.show()
