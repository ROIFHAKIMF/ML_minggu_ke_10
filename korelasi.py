import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load semua data CSV
btc = pd.read_csv('btc.csv', parse_dates=['Date']).set_index('Date')
gold = pd.read_csv('gold.csv', parse_dates=['Date']).set_index('Date')
sp500 = pd.read_csv('sp500.csv', parse_dates=['Date']).set_index('Date')
dxy = pd.read_csv('dxy.csv', parse_dates=['Date']).set_index('Date')
tnx = pd.read_csv('tnx.csv', parse_dates=['Date']).set_index('Date')
bito = pd.read_csv('bito.csv', parse_dates=['Date']).set_index('Date')
vix = pd.read_csv('vix.csv', parse_dates=['Date']).set_index('Date')

# Gabungkan semua ke dalam satu DataFrame berdasarkan tanggal
df = pd.concat([
    btc['Close'].rename('BTC'),
    gold['Close'].rename('Gold'),
    sp500['Close'].rename('SP500'),
    dxy['Close'].rename('DXY'),
    tnx['Close'].rename('TNX'),
    bito['Close'].rename('BITO'),
    vix['Close'].rename('VIX')
], axis=1)

# Drop NaN agar perhitungan korelasi akurat
df.dropna(inplace=True)

# Hitung korelasi
correlation = df.corr()

# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Korelasi antar fitur dengan Harga BTC')
plt.tight_layout()
plt.savefig("korelasi.png")
plt.show()
