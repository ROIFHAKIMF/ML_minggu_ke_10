import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import statsmodels.api as sm

# Load data BTC
df = pd.read_csv('btc.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# Filter data 2018 - 2024
df_filtered = df[(df.index >= '2018-01-01') & (df.index <= '2025-05-31')].copy()

# Tambahkan EMA
df_filtered['EMA100'] = df_filtered['Close'].ewm(span=100, adjust=False).mean()
df_filtered['EMA200'] = df_filtered['Close'].ewm(span=200, adjust=False).mean()

# Ekstrak harga penutupan
close_prices = df_filtered['Close'].values
dates = df_filtered.index

# Deteksi peaks (puncak) dan valleys (lembah)
peaks, _ = find_peaks(close_prices, distance=20, prominence=500)
valleys, _ = find_peaks(-close_prices, distance=20, prominence=500)

# ===== Double Top Detection =====
def detect_double_top(prices, peaks, threshold=0.02):
    if len(peaks) < 2:
        return False
    last_two_peaks = prices[peaks[-2:]]
    diff = abs(last_two_peaks[0] - last_two_peaks[1])
    avg = np.mean(last_two_peaks)
    return diff < threshold * avg

# ===== Ascending Triangle Detection =====
def detect_ascending_triangle(prices, peaks, valleys, top_std_thresh=0.02):
    if len(valleys) < 5 or len(peaks) < 3:
        return False
    valley_prices = prices[valleys[-5:]]
    valley_indices = np.arange(len(valley_prices))

    # Linear regression ke lembah
    model = sm.OLS(valley_prices, sm.add_constant(valley_indices)).fit()
    slope = model.params[1]

    top_prices = prices[peaks[-3:]]
    top_std = np.std(top_prices) / np.mean(top_prices)

    return slope > 0 and top_std < top_std_thresh

# Cek dan print hasil deteksi
if detect_double_top(close_prices, peaks):
    print("🚨 POLA DOUBLE TOP TERDETEKSI")

if detect_ascending_triangle(close_prices, peaks, valleys):
    print("📈 POLA ASCENDING TRIANGLE TERDETEKSI")

# ===== Visualisasi =====
# DEBUG: cek jumlah peaks dan valleys
print(f"✅ Peaks: {len(peaks)}, Valleys: {len(valleys)}")

plt.figure(figsize=(14, 7))
plt.plot(dates, close_prices, label='Harga BTC', color='orange', linewidth=2)
plt.plot(dates[peaks], close_prices[peaks], 'ro', label='Peaks')
plt.plot(dates[valleys], close_prices[valleys], 'go', label='Valleys')
plt.plot(dates, df_filtered['EMA100'], label='EMA 100', color='red', linestyle='--')
plt.plot(dates, df_filtered['EMA200'], label='EMA 200', color='blue', linestyle='--')

# ===== DOUBLE TOP FIXED =====
if detect_double_top(close_prices, peaks):
    if len(peaks) >= 2:
        idx1, idx2 = peaks[-2], peaks[-1]
        x_vals = [df_filtered.index[idx1], df_filtered.index[idx2]]
        y_vals = [close_prices[idx1], close_prices[idx2]]
        plt.plot(x_vals, y_vals, color='purple', linewidth=2, linestyle='-', label='Double Top Line')

# ===== ASCENDING TRIANGLE FIXED =====
if detect_ascending_triangle(close_prices, peaks, valleys):
    # Garis resistance (horizontal di puncak)
    top_y = np.mean(close_prices[peaks[-3:]])
    plt.hlines(y=top_y, xmin=df_filtered.index[peaks[-3]], xmax=df_filtered.index[peaks[-1]],
               color='green', linestyle='--', linewidth=2, label='Resistance')

    # Garis support (regresi lembah terakhir)
    valley_idx = valleys[-5:]
    valley_x = np.arange(len(valley_idx))
    valley_y = close_prices[valley_idx]

    model = sm.OLS(valley_y, sm.add_constant(valley_x)).fit()
    reg_line = model.predict(sm.add_constant(valley_x))
    support_dates = df_filtered.index[valley_idx]

    plt.plot(support_dates, reg_line, color='brown', linestyle='--', linewidth=2, label='Support Line')

plt.title('Deteksi Pola BTC: Double Top & Ascending Triangle (2018 - 2024)')
plt.xlabel('Tahun')
plt.ylabel('Harga (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("btc_chart_with_patterns_fixed.png")
plt.show()
