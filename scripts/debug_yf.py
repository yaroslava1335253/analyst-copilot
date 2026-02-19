
import yfinance as yf
import pandas as pd

try:
    ticker = yf.Ticker("MSFT")
    inc = ticker.income_stmt
    bal = ticker.balance_sheet
    
    print("INCOME STATEMENT INDICES:")
    print(inc.index.tolist())
    
    print("\nBALANCE SHEET INDICES:")
    print(bal.index.tolist())
    
except Exception as e:
    print(e)
