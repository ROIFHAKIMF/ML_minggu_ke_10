from fredapi import Fred
import matplotlib.pyplot as plt

fred = Fred(api_key='62a928422ecc71d22b67157593bb0263')  # Daftar gratis di fred.stlouisfed.org

m2 = fred.get_series('M2SL')
walcl = fred.get_series('WALCL')  # Fed balance sheet

m2.plot(label='M2 Money Supply', figsize=(12,6))
walcl.plot(label='FED Balance Sheet')
plt.legend()
plt.title('US Liquidity Metrics')
plt.grid(True)
plt.savefig("m2.png")
plt.show()
