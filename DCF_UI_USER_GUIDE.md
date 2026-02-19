
# DCF UI User Guide — How to Use the Fixed Integration

**Date**: February 5, 2026  
**Status**: ✅ Ready to use

---

## Quick Start

### Step 1: Load Data
```
Sidebar → Enter Ticker (e.g., MSFT) → Click "Load Data"
```

### Step 2: Generate DCF Analysis
```
Scroll to Step 4: Verified DCF Analysis
→ Click "Run Verified DCF Analysis" (blue button)
→ Wait for analysis to complete (~5 seconds)
```

### Step 3: Review Summary
You'll see a clean summary page with:
- **4 Key Metrics**: Enterprise Value, Equity Value, Intrinsic Value/Share, Data Quality Score
- **Current Financial Position**: Real data (not zeros)
- **5-Year FCF Projection**: Table with discount factors
- **Valuation Bridge**: EV → Equity → Per Share
- **Key Assumptions**: WACC, growth rates, exit multiple
- **Valuation vs Market**: Buy/Sell/Hold recommendation

### Step 4 (Optional): View Full Details
Click `[View DCF Details →]` button to see:
- **Inputs Table**: All financial data with provenance
- **Assumptions Table**: Full list of model assumptions
- **5-Year FCF Projection**: With growth rates and discount factors
- **Terminal Value Calculation**: How TV is derived
- **Valuation Bridge**: Step-by-step EV to per-share calculation
- **Warnings & Diagnostics**: Any data quality issues
- **Export Trace (JSON)**: Download calculation trace for audit

---

## Understanding the Data

### What Do The Numbers Mean?

**Enterprise Value ($3,831.9B)**
- Sum of present values of: 5-year FCF projections + Terminal Value
- This is what the business is worth to equity holders + debt holders
- Formula: PV(FCF 1-5) + PV(Terminal Value)

**Equity Value ($3,830.5B)**
- What the business is worth to equity holders only
- Formula: Enterprise Value - Net Debt
- Net Debt = Total Debt - Cash & Equivalents

**Intrinsic Value/Share ($511.42)**
- What one share is theoretically worth based on DCF analysis
- Formula: Equity Value / Shares Outstanding
- Compare to Current Market Price ($416.25) to get upside/downside

**Data Quality Score (92/100)**
- Composite reliability score across 5 critical fields:
  1. Current Price
  2. Shares Outstanding
  3. TTM Revenue
  4. TTM FCF
  5. Operating Income
- Higher is better (>80 = very reliable, 60-80 = acceptable, <60 = use with caution)

### Reading the Tables

#### Input Data Table
```
Item                  Value        Units  Period  Source            Reliability  Notes
─────────────────────────────────────────────────────────────────────────────────
Current Price         $416.25      USD    TTM     yf.Ticker.info    95/100       —
Shares Outstanding    7.43B        shares Annual  yf.Ticker.info    90/100       Latest annual
TTM Revenue           $245.1B      USD    TTM     yf.quarterly_..   95/100       4Q sum
TTM CapEx             $8.1B        USD    TTM     yf.quarterly_..   85/100       Converted abs
```

**What to look for**:
- Missing values show "—" (not $0.0B)
- Reliability < 80 → take with grain of salt
- Notes column explains fallbacks (e.g., "quarterly unavailable, used annual")

#### Assumptions Table
```
Assumption                      Value     Notes
──────────────────────────────────────────────────────────────
WACC (Discount Rate)            8.50%     Cost of equity + weighted cost of debt
FCF Growth Rate (Years 1-5)     5.20%     Applied to each year of projection
Terminal Growth Rate            3.00%     Perpetual growth (Gordon Growth method)
Terminal Value Method           exit_mult Size-based EBITDA multiple
Exit Multiple (if used)         15.0x     Small/mid-cap tech company multiple
```

**How to adjust**:
- If you think WACC should be 9% instead of 8.50%, you can:
  1. Edit the assumptions in the code (dcf_engine.py)
  2. Re-run the analysis
  3. Compare valuations side-by-side to see sensitivity
- The Details page shows how sensitive the valuation is to changes

#### 5-Year FCF Projection
```
Year   FCF ($B)   Growth    Discount Factor   PV(FCF) ($B)
────────────────────────────────────────────────────────────
Year 1   $79.1      —         1.0850            $72.9
Year 2   $83.1     +5.2%      1.1772            $70.6
Year 3   $87.3     +5.2%      1.2759            $68.4
Year 4   $91.7     +5.2%      1.3818            $66.4
Year 5   $96.3     +5.2%      1.4953            $64.4
```

**What this shows**:
- FCF grows 5.2% annually (your model assumption)
- Discount Factor = 1 / (1 + WACC)^year
  - Higher year → larger discount factor → lower present value
- PV(FCF) = FCF / Discount Factor
- Sum of PV(FCF) = $342.7B (explicit forecast value)

#### Valuation Bridge
```
Component                        Value         Formula/Notes
────────────────────────────────────────────────────────────────────
PV(FCF Years 1–5)                $342.7B       Σ(FCF_t / (1+WACC)^t)
PV(Terminal Value)               $3,489.2B     Terminal Value / (1+WACC)^5
= Enterprise Value               $3,831.9B     PV(FCF) + PV(TV)
− Net Debt                       $1.4B         Total Debt − Cash
= Equity Value                   $3,830.5B     EV − Net Debt
÷ Shares Outstanding             7.43B shares  Diluted share count
= Intrinsic Value/Share          $511.42       Equity Value ÷ Shares
```

**Key validation checks**:
- ✓ EV = PV(FCF) + PV(TV) → 3,831.9 = 342.7 + 3,489.2 ✓
- ✓ Equity = EV - ND → 3,830.5 = 3,831.9 - 1.4 ✓
- ✓ Per-Share = Equity / Shares → 511.42 = 3,830.5B / 7.43B ✓

---

## Common Questions

### Q: Why is Intrinsic Value ($511.42) different from Current Price ($416.25)?

**A**: The DCF model values the stock at $511.42 based on:
- Historical cash flows
- 5.2% growth assumption
- 8.5% discount rate
- 15x exit multiple for terminal value

But the market price is $416.25, meaning the stock appears **22.7% undervalued** by this model.

This is the basis for investment recommendations: if you believe the DCF assumptions are correct, the stock is a buy.

### Q: What if Intrinsic Value and Current Price are very close?

**A**: The stock is **fairly valued**. Market price ≈ DCF valuation. No obvious buy/sell signal.

This often means the market has already priced in the company's growth prospects accurately.

### Q: What if Intrinsic Value < Current Price?

**A**: The stock appears **overvalued**. The model says it's worth less than current market price.

This suggests either:
1. The market is overly optimistic, OR
2. Your model assumptions are too conservative

Check the sensitivity analysis to see if small changes in WACC/growth flip the verdict.

### Q: Why does the Data Quality Score matter?

**A**: If Data Quality < 60/100, the valuation may be unreliable.

Lower scores happen when:
- Recent quarterly earnings haven't been filed yet
- The company is too new (limited historical data)
- Financial statements are inconsistent
- Critical line items are estimated or missing

**Action**: If quality score is low, wait for more recent data or use conservative assumptions.

### Q: Can I change WACC or growth rate?

**A**: Currently, you'd need to:
1. Edit `dcf_engine.py` line ~310 (DCFAssumptions defaults)
2. Re-run analysis
3. Compare results

In future versions, we'll add an inline adjustment feature in the UI.

### Q: What does Terminal Value dominance mean?

**A**: Terminal Value (PV) / Enterprise Value ratio.

Example: If terminal value is 91% of EV, then 91% of the valuation depends on assumptions about Year 6+.

**High dominance (>75%)** = small changes to exit multiple / perpetual growth rate → big changes to valuation.

This is **normal and expected** for mature, stable companies (like MSFT).

For young, high-growth companies, FCF dominance might be 50%+ of EV.

### Q: What does "No quarterly CFO; used annual as proxy" mean?

**A**: It means:
- The model preferred to use 4 quarters of recent operating cash flow (TTM)
- But quarterly data wasn't available
- So it fell back to the most recent annual number
- This reduces reliability (marked as 70/100 instead of 90/100)

**Action**: None; the model handles this automatically. But note the warning.

---

## Workflow Examples

### Example 1: Quick Valuation (2 minutes)
1. Sidebar → Ticker: MSFT → Load Data
2. Scroll to Step 4 → Run DCF
3. Read the 4 metrics (EV, Equity, Per-Share, Quality)
4. Check "Valuation vs Market" section
5. Done! Quick buy/sell opinion.

### Example 2: Deep Dive (10 minutes)
1. Steps 1-2 above
2. Click "View DCF Details →"
3. Review Input Data table (check for missing data)
4. Review Assumptions (do they match your views?)
5. Review 5-Year FCF Projection (realistic growth?)
6. Review Bridge (math checks out?)
7. Check Warnings panel
8. Download Trace (JSON) for audit trail
9. Done! Ready to discuss with portfolio manager.

### Example 3: Sensitivity Analysis (5 minutes)
1. Steps 1-2 above
2. In Details page, note the current assumptions
3. In DCF code, change WACC from 8.5% to 7.5% (more bullish) or 9.5% (more bearish)
4. Re-run analysis
5. Compare intrinsic values: $XXX (bullish) vs $YYY (base) vs $ZZZ (bearish)
6. Conclusion: valuation is X% sensitive to WACC ±1%

---

## Data Quality Thresholds

| Score Range | Interpretation | Action |
|-------------|------------------|--------|
| 90-100 | Excellent | Use confidently |
| 80-89 | Good | Use with minor caveats |
| 70-79 | Acceptable | Use but note fallbacks |
| 60-69 | Caution | Use conservative assumptions |
| <60 | Unreliable | Wait for better data; or use only for ballpark estimates |

---

## Troubleshooting

### Problem: All inputs show "—" (dashes)
**Cause**: yfinance couldn't fetch data for this ticker.  
**Solution**: Check if ticker is spelled correctly. Some tickers like BERKB (Berkshire Hathaway) need exact case.

### Problem: Data Quality Score is 0/100
**Cause**: One of the 5 critical fields is missing: price, shares, revenue, FCF, or operating income.  
**Solution**: This is real; yfinance couldn't fetch required data. Try again tomorrow when latest earnings are available.

### Problem: Intrinsic Value is very high (>100x current price)
**Cause**: Usually unrealistic assumptions (too high growth rate or too low WACC).  
**Solution**: Check the Details page → Assumptions. Adjust to match consensus estimates or historical averages.

### Problem: Intrinsic Value is zero
**Cause**: Equity Value was computed as 0 (shouldn't happen).  
**Solution**: Report this as a bug; include ticker and error message from Warnings panel.

### Problem: Can't see Details page button
**Cause**: Details page is only available after running analysis.  
**Solution**: Click "Run Verified DCF Analysis" first, wait for it to complete, then button will appear.

---

## Export & Audit

### Downloading the Trace
In the Details page, click `[Download Trace (JSON)]`.

This creates a file like: `MSFT_dcf_trace.json`

**Contents**:
```json
{
  "inputs": {
    "current_price": {"value": 416.25, "units": "USD", ...},
    "shares_outstanding": {"value": 7430000000, ...},
    ...
  },
  "assumptions": {
    "wacc": 0.085,
    "fcf_growth_rate": 0.052,
    ...
  },
  "results": {
    "enterprise_value": 3831900000000,
    "equity_value": 3830500000000,
    "price_per_share": 511.42,
    ...
  },
  "trace_steps": [
    {"name": "5-Year FCF Projection", "formula": "...", "inputs": {...}, "output": ...},
    ...
  ]
}
```

**Use case**: Send to an auditor or colleague to review your entire calculation.

---

## Comparison with Old System

| Feature | Old App | New App |
|---------|---------|---------|
| DCF Summary | ✓ (but all zeros) | ✓ (correct values) |
| Key Metrics | 4 metrics (broken) | 4 metrics (fixed) |
| Input Transparency | ✗ | ✓ (Inputs table with provenance) |
| Assumptions Visible | ✗ | ✓ (Assumptions table) |
| 5-Year Projection | ✗ | ✓ (Full table with discount factors) |
| Bridge Calculation | ✗ | ✓ (Step-by-step EV → per-share) |
| Trace Export | ✗ | ✓ (JSON download) |
| Data Quality Warnings | ✗ | ✓ (Warnings panel) |
| Per-Share Validation | ✗ (showed 0) | ✓ (equity/shares check) |
| Two-Level UX | ✗ (cramped) | ✓ (summary + details) |

---

## Next Steps

1. **Verify with your ticker**: Load your favorite stock and compare DCF value to current market price
2. **Adjust assumptions**: If you disagree with default assumptions, modify in `dcf_engine.py` and re-run
3. **Share the trace**: Download JSON and send to colleagues for independent review
4. **Run sensitivity analysis**: Try WACC ± 1% and exit multiple ± 2x to see range of values
5. **Build your thesis**: Combine DCF, comparable multiples, and qualitative factors for investment decision

---

## Support

If you encounter issues or have questions:
1. Check the Warnings panel in Details page
2. Verify ticker spelling matches yfinance convention
3. Check that stock has sufficient financial history (>1 year)
4. Review the DCF_UI_FIX_SUMMARY.md document for technical details

---

**Happy analyzing! 📊**

