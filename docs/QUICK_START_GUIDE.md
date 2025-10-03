# 🚀 Quick Start Guide

Get WZRD-Algo-Mini running in 5 minutes!

## 📋 Prerequisites

- Python 3.8+
- Git

## ⚡ Installation

### 1. Clone Repository
```bash
git clone https://github.com/mdthewzrd/wzrd-algo-system.git
cd wzrd-algo-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start All Services
```bash
./start_services.sh
```

## 🎯 Access Applications

| Application | URL | Purpose |
|-------------|-----|---------|
| **Signal Codifier** | http://localhost:8502 | Convert natural language → JSON |
| **Strategy Viewer** | http://localhost:8510 | View charts with signals |
| **Scan Builder** | http://localhost:8503 | Validate parameters |

## 🧪 Test System

### Option 1: Load Example Strategy
1. Go to **Strategy Viewer**: http://localhost:8510
2. Click "Load Existing File"
3. Select `corrected_ema_strategy.json`
4. Click "Load Strategy"
5. See interactive chart with signals! 🎉

### Option 2: Create New Strategy
1. Go to **Signal Codifier**: http://localhost:8502
2. Enter: `"Create a SPY strategy that buys when 9 EMA crosses above 20 EMA"`
3. Click "Generate Strategy"
4. Copy the JSON output
5. Go to **Strategy Viewer** and paste JSON
6. View your new strategy chart!

## ✅ Success Indicators

You know it's working when:
- ✅ All services start without errors
- ✅ Example strategies load and show charts
- ✅ WZRD deviation bands are visible
- ✅ Signals appear at correct prices

## 🐛 Troubleshooting

### Services Won't Start
```bash
# Check if ports are busy
lsof -i :8502 -i :8510 -i :8503

# Kill existing processes
pkill -f streamlit

# Restart
./start_services.sh
```

### Charts Don't Load
- Check JSON format is valid
- Verify signal timestamps are in correct format: "YYYY-MM-DD HH:MM:SS"
- Ensure prices are realistic for the symbol

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## 🎓 Next Steps

1. **Read Strategy Format**: See `docs/JSON_STRATEGY_FORMAT.md`
2. **Learn WZRD Templates**: Explore deviation bands and EMA clouds
3. **Create Custom Strategies**: Use Signal Codifier for your ideas
4. **Validate Parameters**: Use Scan Builder for testing

---

**You're ready to create professional trading strategies!** 🚀