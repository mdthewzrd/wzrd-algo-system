#!/usr/bin/env python3
"""
WZRD-Algo-Mini Demo Workflow

Demonstrates complete workflow from strategy creation to visualization.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def create_demo_strategy():
    """Create a demo strategy for testing"""
    print("📝 Creating demo strategy...")

    # Generate realistic demo signals
    base_date = datetime(2024, 10, 1, 9, 30)  # Start at 9:30 AM
    signals = []

    # Entry signal
    entry_time = base_date + timedelta(minutes=30)  # 10:00 AM
    signals.append({
        "type": "entry_signal",
        "timestamp": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        "price": 575.25,
        "reason": "Demo: EMA 9 crosses above EMA 20",
        "direction": "long",
        "shares": 100,
        "pnl": 0
    })

    # Exit signal
    exit_time = entry_time + timedelta(hours=1)  # 11:00 AM
    exit_price = 578.50
    pnl = (exit_price - 575.25) * 100  # $325 profit
    signals.append({
        "type": "exit_signal",
        "timestamp": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
        "price": exit_price,
        "reason": "Demo: Take profit target reached",
        "direction": "long",
        "shares": 100,
        "pnl": round(pnl, 2)
    })

    strategy = {
        "strategy_name": "Demo_EMA_Crossover",
        "description": "Demo strategy showing EMA crossover with WZRD charts",
        "timeframe": "5min",
        "symbol": "SPY",
        "signals": signals,
        "entry_conditions": [
            "EMA 9 > EMA 20",
            "Volume > average",
            "Time: 8 AM - 1 PM EST"
        ],
        "exit_conditions": [
            "Take profit: +$3.25",
            "Stop loss: -$2.00",
            "Time limit: 4 hours"
        ]
    }

    return strategy

def save_demo_strategy(strategy):
    """Save demo strategy to file"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies')
    os.makedirs(output_dir, exist_ok=True)

    filename = f"demo_strategy_{int(time.time())}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(strategy, f, indent=2)

    print(f"💾 Demo strategy saved: {filename}")
    return filepath

def validate_strategy(strategy):
    """Validate strategy format"""
    print("✅ Validating strategy format...")

    required_fields = ['strategy_name', 'description', 'timeframe', 'symbol', 'signals']
    for field in required_fields:
        if field not in strategy:
            raise ValueError(f"Missing required field: {field}")

    # Validate signals
    for i, signal in enumerate(strategy['signals']):
        required_signal_fields = ['type', 'timestamp', 'price', 'reason']
        for field in required_signal_fields:
            if field not in signal:
                raise ValueError(f"Missing signal field {field} in signal {i}")

        # Validate timestamp format
        try:
            datetime.strptime(signal['timestamp'], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError(f"Invalid timestamp format in signal {i}: {signal['timestamp']}")

        # Validate entry time (8 AM - 1 PM EST)
        if signal['type'] == 'entry_signal':
            dt = datetime.strptime(signal['timestamp'], "%Y-%m-%d %H:%M:%S")
            if not (8 <= dt.hour <= 13):
                raise ValueError(f"Entry signal at {signal['timestamp']} outside 8 AM - 1 PM window")

    print("✅ Strategy validation passed!")

def run_demo_workflow():
    """Run complete demo workflow"""
    print("🚀 WZRD-Algo-Mini Demo Workflow")
    print("=" * 50)

    try:
        # Step 1: Create strategy
        strategy = create_demo_strategy()

        # Step 2: Validate strategy
        validate_strategy(strategy)

        # Step 3: Save strategy
        filepath = save_demo_strategy(strategy)

        # Step 4: Show next steps
        print("\n🎯 Demo Strategy Created Successfully!")
        print("\n📋 Next Steps:")
        print("1. Start services: ./start_services.sh")
        print("2. Go to Strategy Viewer: http://localhost:8510")
        print(f"3. Load strategy file: {os.path.basename(filepath)}")
        print("4. View interactive chart with WZRD templates!")

        print("\n📊 Strategy Summary:")
        print(f"   • Name: {strategy['strategy_name']}")
        print(f"   • Symbol: {strategy['symbol']}")
        print(f"   • Timeframe: {strategy['timeframe']}")
        print(f"   • Signals: {len(strategy['signals'])}")

        if len(strategy['signals']) >= 2:
            entry = strategy['signals'][0]
            exit = strategy['signals'][1]
            print(f"   • Entry: ${entry['price']} at {entry['timestamp']}")
            print(f"   • Exit: ${exit['price']} at {exit['timestamp']}")
            print(f"   • P&L: ${exit['pnl']}")

    except Exception as e:
        print(f"❌ Demo workflow error: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 Demo workflow completed successfully!")
    return True

if __name__ == "__main__":
    run_demo_workflow()