import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load data
df = pd.read_csv('btc.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df[['Close']].dropna()

# 2. Normalisasi
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 3. Buat sequence (window)
def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, window=60)

# 4. Split data
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 5. Build model LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 6. Prediksi
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test)

# 7. Plot hasil
# Ambil tanggal dari data asli untuk test set
dates = df.index[-len(y_test):]
plt.figure(figsize=(12,6))
plt.plot(dates, real_prices, label='Actual BTC Price')
plt.plot(dates, predicted_prices, label='Predicted BTC Price')
plt.title('BTC Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("btc_lstm_prediction.png")
plt.show()

# Asumsikan kode sampai model.fit() sudah sama seperti sebelumnya.

# ===== 8. PREDIKSI 30 HARI KE DEPAN =====

# a) Ambil 60 hari data terakhir (scaled) dari scaled_data
last_60 = scaled_data[-60:]  # shape = (60, 1)

# b) Siapkan list untuk menampung prediksi (dalam skala 0-1)
future_preds_scaled = []

# c) Lakukan iterasi 30 kali
current_window = last_60.copy()  # (60, 1)
for day in range(180):
    # reshape current_window ke bentuk (1, 60, 1) supaya bisa diprediksi LSTM
    input_seq = current_window.reshape((1, current_window.shape[0], 1))
    pred_scaled = model.predict(input_seq)          # shape (1,1)
    future_preds_scaled.append(pred_scaled[0,0])    # ambil nilai skala tunggal

    # geser window: buang elemen pertama, tambahkan prediksi
    current_window = np.vstack((current_window[1:], [[pred_scaled[0,0]]]))

# d) Ubah kembali skala prediksi ke harga asli (USD)
future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)  # shape (30,1)
future_preds = scaler.inverse_transform(future_preds_scaled)        # shape (30,1)

# e) Siapkan index tanggal 30 hari ke depan
#    Cari tanggal terakhir di df.index, lalu tambahkan 1 hari secara berurutan
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=180, freq='D')

# f) Plot hasil historical + 30 hari prediksi
plt.figure(figsize=(12, 6))
# Plot data historis (harga asli)
plt.plot(df.index, df['Close'], label='Harga Asli (s/d hari terakhir)', color='blue')
# Plot prediksi 30 hari ke depan
plt.plot(future_dates, future_preds, label='Prediksi 180 Hari ke Depan', color='red', linestyle='--')

plt.title('Prediksi Harga BTC 30 Hari ke Depan dengan LSTM')
plt.xlabel('Tanggal')
plt.ylabel('Harga (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("btc_180day_forecast.png")
plt.show()


