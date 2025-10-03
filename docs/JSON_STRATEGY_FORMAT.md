# 📋 JSON Strategy Format Specification

Complete reference for WZRD strategy JSON format.

## 🎯 Basic Structure

```json
{
  "strategy_name": "Strategy Name",
  "description": "Strategy description",
  "timeframe": "5min",
  "symbol": "SPY",
  "signals": [...],
  "entry_conditions": [...],
  "exit_conditions": [...]
}
```

## 📊 Required Fields

### Core Metadata
- **`strategy_name`** (string): Unique strategy identifier
- **`description`** (string): Strategy explanation
- **`timeframe`** (string): Chart timeframe (`5min`, `15min`, `1H`, `1D`)
- **`symbol`** (string): Trading symbol (`SPY`, `QQQ`, `AAPL`, etc.)

### Signal Array
- **`signals`** (array): List of entry/exit signals

## 🎪 Signal Object Format

```json
{
  "type": "entry_signal",
  "timestamp": "2024-10-01 09:30:00",
  "price": 575.25,
  "reason": "EMA 9 crosses above EMA 20",
  "direction": "long",
  "shares": 100,
  "pnl": 0
}
```

### Signal Fields
- **`type`** (string): `"entry_signal"` or `"exit_signal"`
- **`timestamp`** (string): Format: `"YYYY-MM-DD HH:MM:SS"`
- **`price`** (number): Entry/exit price
- **`reason`** (string): Signal justification
- **`direction`** (string): `"long"` or `"short"`
- **`shares`** (number): Position size
- **`pnl`** (number): Profit/loss (for exit signals)

## ⏰ Time Rules

### Entry Signals
- **Only 8:00 AM - 1:00 PM EST**
- Must be market hours (9:30 AM - 4:00 PM EST)
- Avoid weekends

### Exit Signals
- Any market hours
- Must have matching entry signal

## 💰 Price Guidelines

### Common Symbol Ranges
| Symbol | Typical Range |
|--------|---------------|
| SPY | $400 - $600 |
| QQQ | $300 - $500 |
| AAPL | $150 - $200 |
| MSFT | $300 - $450 |
| TSLA | $150 - $300 |

## 📝 Complete Example

```json
{
  "strategy_name": "EMA_Crossover_SPY_5min",
  "description": "Buy when 9 EMA crosses above 20 EMA, sell when crosses below",
  "timeframe": "5min",
  "symbol": "SPY",
  "signals": [
    {
      "type": "entry_signal",
      "timestamp": "2024-10-01 09:35:00",
      "price": 575.25,
      "reason": "EMA 9 crosses above EMA 20 with volume confirmation",
      "direction": "long",
      "shares": 100,
      "pnl": 0
    },
    {
      "type": "exit_signal",
      "timestamp": "2024-10-01 11:45:00",
      "price": 578.80,
      "reason": "EMA 9 crosses below EMA 20",
      "direction": "long",
      "shares": 100,
      "pnl": 355.0
    }
  ],
  "entry_conditions": [
    "EMA 9 > EMA 20",
    "Volume > 20-period average",
    "Time between 8:00 AM - 1:00 PM EST"
  ],
  "exit_conditions": [
    "EMA 9 < EMA 20",
    "Stop loss at 1% below entry",
    "Take profit at 2% above entry"
  ]
}
```

## ✅ Validation Checklist

### Required Validation
- [ ] All required fields present
- [ ] Timestamps in correct format
- [ ] Entry signals only 8 AM - 1 PM EST
- [ ] Prices realistic for symbol
- [ ] Signal types match: entry/exit pairs
- [ ] No weekend dates

### Best Practices
- [ ] Descriptive strategy name
- [ ] Clear entry/exit reasons
- [ ] Consistent timeframe
- [ ] Realistic position sizes
- [ ] Complete entry/exit pairs

## 🚨 Common Errors

### Invalid Timestamps
```json
❌ "timestamp": "10/01/2024 9:30 AM"
✅ "timestamp": "2024-10-01 09:30:00"
```

### Wrong Time Windows
```json
❌ "timestamp": "2024-10-01 15:30:00"  // 3:30 PM - too late
✅ "timestamp": "2024-10-01 10:30:00"  // 10:30 AM - valid
```

### Unrealistic Prices
```json
❌ "price": 50.00     // SPY too low
✅ "price": 575.25    // SPY realistic
```

## 🔧 Testing Your JSON

```bash
# Validate JSON syntax
python -m json.tool your_strategy.json

# Test in Strategy Viewer
# 1. Go to http://localhost:8510
# 2. Paste JSON in "Paste JSON" tab
# 3. Click "Parse Strategy"
# 4. Check for errors
```

---

**Follow this format for perfect WZRD strategy compatibility!** 🎯