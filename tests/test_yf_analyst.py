import yfinance as yf

stock = yf.Ticker('MSFT')
info = stock.info

print('=== Price Targets ===')
print('Target Mean:', info.get('targetMeanPrice'))
print('Target High:', info.get('targetHighPrice'))
print('Target Low:', info.get('targetLowPrice'))
print('Num Analysts:', info.get('numberOfAnalystOpinions'))
print('Recommendation:', info.get('recommendationKey'))

print('\n=== Earnings Estimates ===')
try:
    est = stock.earnings_estimate
    if est is not None:
        print(est)
except Exception as e:
    print(f"Error: {e}")

print('\n=== Revenue Estimates ===')
try:
    rev = stock.revenue_estimate
    if rev is not None:
        print(rev)
except Exception as e:
    print(f"Error: {e}")

print('\n=== Recommendations Summary ===')
try:
    rec_sum = stock.recommendations_summary
    if rec_sum is not None:
        print(rec_sum)
except Exception as e:
    print(f"Error: {e}")
