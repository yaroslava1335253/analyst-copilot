import yfinance as yf
from data_adapter import DataAdapter

# Test MSFT
print('=== MSFT ===')
adapter = DataAdapter('MSFT')
snapshot = adapter.fetch()
print(f'Beta: {snapshot.beta.value}')
print(f'Suggested WACC: {snapshot.suggested_wacc.value*100:.1f}%')
print(f'Suggested FCF Growth: {snapshot.suggested_fcf_growth.value*100:.1f}%')
print()

# Test AMZN
print('=== AMZN ===')
adapter2 = DataAdapter('AMZN')
snapshot2 = adapter2.fetch()
print(f'Beta: {snapshot2.beta.value}')
print(f'Suggested WACC: {snapshot2.suggested_wacc.value*100:.1f}%')
print(f'Suggested FCF Growth: {snapshot2.suggested_fcf_growth.value*100:.1f}%')
