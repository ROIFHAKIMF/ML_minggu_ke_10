import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fungsi load dan siapin data
def load_and_prepare(filename):
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

# Load semua data
btc = load_and_prepare("btc.csv")
gold = load_and_prepare("gold.csv")
sp500 = load_and_prepare("sp500.csv")
dxy = load_and_prepare("dxy.csv")
tnx = load_and_prepare("tnx.csv")
bito = load_and_prepare("bito.csv")
vix = load_and_prepare("vix.csv")

# Gabungkan jadi satu DataFrame
df = pd.DataFrame({
    'BTC': btc['Close'],
    'Gold': gold['Close'],
    'SP500': sp500['Close'],
    'DXY': dxy['Close'],
    'TNX': tnx['Close'],
    'BITO': bito['Close'],
    'VIX': vix['Close'],
})

# Bersihkan data
df.dropna(inplace=True)

# Visualisasi korelasi
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Korelasi antar fitur dengan Harga BTC")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Split fitur dan target
X = df.drop('BTC', axis=1)
y = df['BTC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Training model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Pakai np.sqrt utk versi aman
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Plot hasil prediksi
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label="Actual BTC")
plt.plot(y_test.index, y_pred, label="Predicted BTC")
plt.title("Prediksi Harga BTC vs Aktual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("result.png")
plt.show()
