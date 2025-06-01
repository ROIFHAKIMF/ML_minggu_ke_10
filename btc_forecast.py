import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# === 1. Load data BTC ===
df = pd.read_csv("btc.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
btc = df["Close"].copy().dropna()

# === 2. Buat lag features (supervised format) ===
def create_lagged_features(series, n_lags):
    df = pd.DataFrame()
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = series.shift(i)
    df['target'] = series.values
    df.dropna(inplace=True)
    return df

n_lags = 30  # 30 hari ke belakang
data = create_lagged_features(btc, n_lags)

# === 3. Split train/test (misal 80:20) ===
split = int(len(data) * 0.8)
train, test = data[:split], data[split:]

X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]

# === 4. Train XGBoost ===
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# === 5. Evaluasi ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

# === 6. Forecast ke 2026 (multi-step forward) ===
forecast_days = 365  # 1 tahun
last_input = data.iloc[-1].drop("target").values

preds = []
for _ in range(forecast_days):
    pred = model.predict([last_input])[0]
    preds.append(pred)
    # Geser input: buang lag paling lama, tambah prediksi terbaru
    last_input = np.roll(last_input, -1)
    last_input[-1] = pred

# === 7. Buat tanggal prediksi ===
last_date = btc.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

# === 8. Plot hasil ===
plt.figure(figsize=(12,6))
plt.plot(btc, label="BTC Historical")
plt.plot(future_dates, preds, label="BTC Forecast 2026", color="orange")
plt.title("Prediksi Harga BTC 2026 (Univariate XGBoost)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
