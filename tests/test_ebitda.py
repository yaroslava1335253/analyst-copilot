import yfinance as yf

ticker = 'MSFT'
stock = yf.Ticker(ticker)

# Check all available data sources for EBITDA
print("=== stock.info keys with 'ebitda' ===")
for k, v in stock.info.items():
    if 'ebitda' in k.lower():
        print(f"  {k}: {v:,.0f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

print("\n=== Quarterly Income Statement - EBITDA row ===")
qis = stock.quarterly_income_stmt
if 'EBITDA' in qis.index:
    print(qis.loc['EBITDA'])
elif 'Normalized EBITDA' in qis.index:
    print("Normalized EBITDA:")
    print(qis.loc['Normalized EBITDA'])

print("\n=== Sum of last 4 quarters ===")
if 'EBITDA' in qis.index:
    ttm_sum = qis.loc['EBITDA'].iloc[:4].sum()
    print(f"TTM EBITDA (sum of 4 quarters): ${ttm_sum:,.0f}")

print("\n=== Annual Income Statement ===")
ais = stock.income_stmt
if 'EBITDA' in ais.index:
    print(ais.loc['EBITDA'])

print("\n=== What Yahoo website shows: $188.3B ===")
print("=== What we need to find is the TTM value ===")
