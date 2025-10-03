# 🚀 WZRD-Algo-System: Spec-Driven Trading Strategy Engine

[![Strategy Engine](https://img.shields.io/badge/Engine-Deterministic-blue)](engine/runner.py)
[![Schema Validation](https://img.shields.io/badge/Validation-Schema--Based-green)](schemas/)
[![Acceptance Testing](https://img.shields.io/badge/Testing-Acceptance--Driven-purple)](utils/validation.py)

> **Professional spec-driven trading system**: Rules → Deterministic Engine → Standardized Outputs

## 🎯 What This System Does

**WZRD-Algo-Mini** converts natural language trading strategy descriptions into structured JSON artifacts that generate interactive charts with backtested signals using professional WZRD chart templates.

### Core Workflow
```
Natural Language → Signal Codifier → JSON Strategy → Strategy Viewer → Interactive Charts
```

## 📁 Project Structure

```
wzrd-algo-mini/
├── 🖥️  apps/                    # Main Applications
│   ├── signal_codifier.py       # Strategy → JSON converter
│   ├── strategy_viewer_enhanced.py # Advanced charting
│   └── scan_builder.py          # Parameter validation
├── 📚 docs/                     # Documentation
├── 📊 strategies/               # Strategy JSON Files
├── 🧪 tests/                    # Test Files & Examples
├── 🔧 utils/                    # Core Engine
└── 🔄 workflows/               # Automation Scripts
```

## ⚡ Quick Start (2 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Services
```bash
# All services at once
./start_services.sh

# Or individually:
# Signal Codifier (Port 8502)
streamlit run apps/signal_codifier.py --server.port 8502

# Strategy Viewer (Port 8510)
streamlit run apps/strategy_viewer_enhanced.py --server.port 8510

# Scan Builder (Port 8503)
streamlit run apps/scan_builder.py --server.port 8503
```

### 3. Test System
1. **Strategy Viewer**: http://localhost:8510
2. Load: `strategies/corrected_ema_strategy.json`
3. See interactive chart with WZRD templates! 🎉

## 🎨 Key Features

### 📊 Advanced Charting
- **WZRD Templates**: Professional deviation bands (920, 7289)
- **EMA Clouds**: Multi-timeframe trend visualization
- **Extended Hours**: 4AM-8PM trading data
- **Signal Alignment**: Perfect price/time positioning

### 🕐 Smart Time Filtering
- **Entry Signals**: Only 8:00am - 1:00pm EST
- **Direction Confirmation**: 1hr EMA trend required
- **Market Hours**: Automatic session detection

### 🔧 Complete Workflow
- **Signal Codifier**: Natural language → JSON conversion
- **Strategy Viewer**: Interactive charts with zoom controls
- **Scan Builder**: Parameter validation and quick testing

## 🔧 Configuration

### Required Environment Variables
```bash
# .env file (optional)
POLYGON_API_KEY=your_api_key_here
```

### Service Ports
| Service | Port | Purpose |
|---------|------|---------|
| Signal Codifier | 8502 | Strategy → JSON conversion |
| Strategy Viewer | 8510 | Advanced charts & signals |
| Scan Builder | 8503 | Parameter validation |

## 📋 Strategy JSON Format

```json
{
  "strategy_name": "EMA Crossover Strategy",
  "description": "9 EMA crosses above 20 EMA on 5-minute charts",
  "timeframe": "5min",
  "symbol": "SPY",
  "signals": [
    {
      "type": "entry_signal",
      "timestamp": "2024-10-01 09:30:00",
      "price": 575.25,
      "reason": "EMA 9 crosses above EMA 20",
      "direction": "long"
    }
  ]
}
```

## 🎯 Working Examples

### Test Strategies (Ready to Use)
- `strategies/corrected_ema_strategy.json` - EMA 9/20 crossover ✅
- `strategies/spy_working_signals_strategy.json` - SPY signals with proper timing ✅

### Example Prompts
```
"Create a SPY strategy that buys when the 9 EMA crosses above the 20 EMA on 5-minute charts. Only enter trades between 8am and 1pm EST. Exit when the EMAs cross back down."

"I want a momentum strategy for QQQ that buys when price breaks above the 20-period high with volume confirmation. Only morning entries."
```

## 🚀 Advanced Features

### Chart Controls
- **Professional WZRD Templates**: Real deviation bands and EMA clouds
- **Custom Date Ranges**: View any historical period
- **Extended Hours Data**: Pre-market and after-hours
- **Signal-Aware Generation**: Perfect price alignment

### Signal Analysis
- **Entry Validation**: 8am-1pm EST compliance
- **Perfect Positioning**: Signals align with actual chart prices
- **Multiple Formats**: Supports various signal types

## 🛠️ Development

### Project Dependencies
```bash
streamlit pandas plotly numpy python-dotenv pytz
```

### Service Management
```bash
# Check running services
lsof -i :8502 -i :8510 -i :8503

# Kill all services
pkill -f streamlit

# Restart fresh
./start_services.sh
```

## 🐛 Troubleshooting

### Common Issues
- **Blank Charts**: Check JSON format and signal timestamps
- **Service Errors**: Verify ports are available
- **Missing Signals**: Ensure proper date format and time ranges

### Quick Fixes
```bash
# Reset services
pkill -f streamlit
./start_services.sh

# Validate JSON
python -m json.tool strategies/your_strategy.json
```

## 🎉 Success Indicators

Your system works when:
- ✅ Services start on ports 8502, 8510, 8503
- ✅ Example strategies load and display charts
- ✅ Signals appear at correct prices and times
- ✅ WZRD chart templates show deviation bands

---

**Ready to create professional trading strategies!** 🚀

Start with `./start_services.sh` and load an example strategy.