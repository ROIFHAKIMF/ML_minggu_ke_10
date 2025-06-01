import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import ta

# =======================
# Load Data BTC
# =======================
# Asumsikan file CSV kamu punya kolom: Date, Close, Volume (opsional)
df = pd.read_csv('btc_price.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()
df['Close'] = df['Close'].astype(float)

# =======================
# Tambah Indikator Teknikal
# =======================
df['SMA_14'] = ta.trend.sma_indicator(df['Close'], window=14)
df['EMA_14'] = ta.trend.ema_indicator(df['Close'], window=14)
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
df['MACD'] = ta.trend.macd_diff(df['Close'])

bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['BB_High'] = bb.bollinger_hband()
df['BB_Low'] = bb.bollinger_lband()
df['BB_Width'] = bb.bollinger_wband()

# Optional: Tambah perubahan volume
if 'Volume' in df.columns:
    df['Volume_Change'] = df['Volume'].pct_change()

# Drop NaN hasil indikator
df.dropna(inplace=True)

# =======================
# Siapkan Fitur dan Target
# =======================
feature_cols = ['SMA_14', 'EMA_14', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'BB_Width']
if 'Volume_Change' in df.columns:
    feature_cols.append('Volume_Change')

X = df[feature_cols]
y = df['Close'].shift(-1)  # Prediksi harga esok hari
X = X[:-1]
y = y[:-1]

# =======================
# Split Train/Test
# =======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# =======================
# Train XGBoost
# =======================
model = XGBRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)

# =======================
# Evaluasi & Prediksi
# =======================
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

# =======================
# Plot Prediksi vs Aktual
# =======================
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='Actual BTC')
plt.plot(y_test.index, y_pred, label='Predicted BTC')
plt.title('Prediksi Harga BTC vs Aktual (XGBoost + Indikator Teknikal)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
