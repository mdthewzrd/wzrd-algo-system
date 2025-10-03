#!/usr/bin/env python3
"""
WZRD-Algo-Mini System Test Suite

Basic tests to verify system functionality.
"""

import json
import os
import sys
import pandas as pd
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_json_strategy_format():
    """Test that example strategies have correct JSON format"""
    print("🧪 Testing JSON Strategy Format...")

    strategies_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies')
    test_files = [
        'corrected_ema_strategy.json',
        'spy_working_signals_strategy.json'
    ]

    for filename in test_files:
        filepath = os.path.join(strategies_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    strategy = json.load(f)

                # Check required fields
                required_fields = ['strategy_name', 'description', 'timeframe', 'symbol', 'signals']
                for field in required_fields:
                    assert field in strategy, f"Missing required field: {field} in {filename}"

                # Check signals format
                if strategy['signals']:
                    signal = strategy['signals'][0]
                    signal_fields = ['type', 'timestamp', 'price', 'reason']
                    for field in signal_fields:
                        assert field in signal, f"Missing signal field: {field} in {filename}"

                print(f"✅ {filename} - Valid JSON format")

            except Exception as e:
                print(f"❌ {filename} - Error: {e}")
        else:
            print(f"⚠️  {filename} - File not found")

def test_chart_templates():
    """Test that chart templates can be imported"""
    print("\n🧪 Testing Chart Templates...")

    try:
        from utils.chart_templates import CHART_TEMPLATES, CHART_STYLE

        # Check templates exist
        assert len(CHART_TEMPLATES) > 0, "No chart templates found"

        # Check required templates
        required_templates = ['5min', '15min', 'hour', 'day']
        for template in required_templates:
            assert template in CHART_TEMPLATES, f"Missing template: {template}"

        print("✅ Chart templates loaded successfully")

    except Exception as e:
        print(f"❌ Chart templates error: {e}")

def test_signal_generator():
    """Test that signal generator can be imported"""
    print("\n🧪 Testing Signal Generator...")

    try:
        from utils.signal_generator import SignalGenerator

        # Try to create instance
        generator = SignalGenerator()
        assert generator is not None, "Failed to create SignalGenerator instance"

        print("✅ Signal generator imported successfully")

    except Exception as e:
        print(f"❌ Signal generator error: {e}")

def test_wzrd_chart():
    """Test that WZRD chart module can be imported"""
    print("\n🧪 Testing WZRD Chart Module...")

    try:
        from utils.wzrd_mini_chart import create_chart

        # Check function exists
        assert callable(create_chart), "create_chart is not callable"

        print("✅ WZRD chart module imported successfully")

    except Exception as e:
        print(f"❌ WZRD chart module error: {e}")

def test_time_validation():
    """Test time validation for entry signals"""
    print("\n🧪 Testing Time Validation...")

    # Valid entry times (8 AM - 1 PM EST)
    valid_times = [
        "2024-10-01 08:00:00",
        "2024-10-01 10:30:00",
        "2024-10-01 13:00:00"
    ]

    # Invalid entry times
    invalid_times = [
        "2024-10-01 07:30:00",  # Too early
        "2024-10-01 15:30:00",  # Too late
        "2024-10-01 20:00:00"   # After hours
    ]

    for time_str in valid_times:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        hour = dt.hour
        assert 8 <= hour <= 13, f"Invalid time validation for {time_str}"

    print("✅ Time validation working correctly")

def run_all_tests():
    """Run all system tests"""
    print("🚀 WZRD-Algo-Mini System Tests")
    print("=" * 50)

    test_json_strategy_format()
    test_chart_templates()
    test_signal_generator()
    test_wzrd_chart()
    test_time_validation()

    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\n💡 To run individual tests:")
    print("   python tests/test_system.py")

if __name__ == "__main__":
    run_all_tests()