#!/usr/bin/env python3
"""
WZRD Strategy Engine Runner

Deterministic engine that consumes StrategySpec + TestPlan and produces standardized outputs.
"""

import json
import yaml
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import warnings

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.validation import StrategyValidator, AcceptanceTestRunner, load_strategy_spec, load_test_plan
from utils.features import FeatureEngine, ExpressionEvaluator
from utils.execution import ExecutionEngine, OrderSide

class StrategyRunner:
    """Main strategy execution engine"""

    def __init__(self, strategy_spec: Dict[str, Any], test_plan: Dict[str, Any],
                 output_dir: str, verbose: bool = False):
        """
        Initialize strategy runner

        Args:
            strategy_spec: Strategy specification dictionary
            test_plan: Test plan dictionary
            output_dir: Directory for output files
            verbose: Enable verbose logging
        """
        self.strategy_spec = strategy_spec
        self.test_plan = test_plan
        self.output_dir = output_dir
        self.verbose = verbose

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.validator = StrategyValidator()
        self.acceptance_runner = AcceptanceTestRunner(self.validator)

        # Initialize state
        self.data = None
        self.features_data = None
        self.signals_data = None
        self.execution_engine = None
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """
        Execute complete strategy run

        Returns:
            Results dictionary with run statistics
        """
        self._log("🚀 Starting WZRD Strategy Run")

        try:
            # Step 1: Validate inputs
            self._validate_inputs()

            # Step 2: Load data
            self._load_market_data()

            # Step 3: Compute features
            self._compute_features()

            # Step 4: Generate signals
            self._generate_signals()

            # Step 5: Execute trades
            self._execute_strategy()

            # Step 6: Generate reports
            self._generate_reports()

            # Step 7: Run acceptance tests
            self._run_acceptance_tests()

            self._log("✅ Strategy run completed successfully")

        except Exception as e:
            self._log(f"❌ Strategy run failed: {e}")
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            raise

        return self.results

    def _validate_inputs(self):
        """Validate strategy spec and test plan"""
        self._log("📋 Validating inputs...")

        # Validate strategy spec
        spec_errors = self.validator.validate_strategy_spec(self.strategy_spec)
        if spec_errors:
            raise ValueError(f"Strategy spec validation failed: {spec_errors}")

        # Validate test plan
        plan_errors = self.validator.validate_test_plan(self.test_plan)
        if plan_errors:
            raise ValueError(f"Test plan validation failed: {plan_errors}")

        self._log("✅ Input validation passed")

    def _load_market_data(self):
        """Load market data based on test plan"""
        self._log("📊 Loading market data...")

        data_config = self.test_plan['data']
        source = data_config['source']
        start_date = data_config['start']
        end_date = data_config['end']
        bars = data_config['bars']

        if source == "mock":
            # Generate mock data for testing
            self.data = self._generate_mock_data(start_date, end_date, bars)
        elif source == "polygon":
            # In production, integrate with Polygon API
            self._log("⚠️  Polygon integration not implemented, using mock data")
            self.data = self._generate_mock_data(start_date, end_date, bars)
        else:
            raise ValueError(f"Unsupported data source: {source}")

        self._log(f"✅ Loaded {len(self.data)} bars of data")

    def _generate_mock_data(self, start_date: str, end_date: str, bars: str) -> pd.DataFrame:
        """Generate realistic mock market data"""
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Generate timestamps
        if bars == "1min":
            freq = "1min"
        elif bars == "5min":
            freq = "5min"
        elif bars == "15min":
            freq = "15min"
        elif bars == "1H":
            freq = "1H"
        else:
            freq = "1D"

        # Create business hour range
        timestamps = pd.date_range(
            start=start_dt,
            end=end_dt,
            freq=freq
        )

        # Filter for market hours (9:30 AM - 4:00 PM ET)
        if bars != "1D":
            timestamps = timestamps[
                (timestamps.hour >= 9) &
                ((timestamps.hour < 16) | ((timestamps.hour == 9) & (timestamps.minute >= 30)))
            ]

        # Generate realistic price data
        np.random.seed(42)  # For reproducibility
        base_price = 575.0  # SPY-like price

        # Generate returns with some autocorrelation
        n_bars = len(timestamps)
        returns = np.random.normal(0, 0.001, n_bars)

        # Add some trend and autocorrelation
        for i in range(1, n_bars):
            returns[i] += 0.05 * returns[i-1]  # Autocorrelation

        # Calculate prices
        prices = [base_price]
        for i in range(1, n_bars):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)

        # Generate OHLC from prices
        data = []
        for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC
            if i == 0:
                open_price = close_price
            else:
                open_price = prices[i-1]

            range_pct = np.random.uniform(0.001, 0.005)  # 0.1% to 0.5% range
            high = max(open_price, close_price) * (1 + range_pct/2)
            low = min(open_price, close_price) * (1 - range_pct/2)

            volume = np.random.randint(100000, 1000000)

            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        return df

    def _compute_features(self):
        """Compute features specified in strategy"""
        self._log("🔧 Computing features...")

        feature_specs = self.strategy_spec.get('features', {})
        if not feature_specs:
            self._log("⚠️  No features specified")
            self.features_data = self.data.copy()
            return

        # Compute features
        feature_engine = FeatureEngine(self.data)
        self.features_data = feature_engine.compute_features(feature_specs)

        self._log(f"✅ Computed {len(feature_specs)} features")

    def _generate_signals(self):
        """Generate entry and exit signals based on rules"""
        self._log("🎯 Generating signals...")

        # Initialize signal columns
        self.signals_data = self.features_data.copy()

        # Generate entry signals
        entry_rules = self.strategy_spec.get('entry_rules', [])
        for rule in entry_rules:
            rule_id = rule['id']
            condition = rule['when']
            cooldown_bars = rule.get('cooldown_bars', 0)

            # Evaluate condition
            evaluator = ExpressionEvaluator(self.features_data)
            signal_series = evaluator.evaluate(condition)

            # Apply cooldown
            if cooldown_bars > 0:
                signal_series = self._apply_cooldown(signal_series, cooldown_bars)

            # Store signal
            self.signals_data[f'entry_{rule_id}'] = signal_series

        # Generate exit signals
        exit_rules = self.strategy_spec.get('exit_rules', [])
        for rule in exit_rules:
            rule_id = rule['id']
            rule_type = rule['type']

            if rule_type == 'condition':
                condition = rule['when']
                evaluator = ExpressionEvaluator(self.features_data)
                signal_series = evaluator.evaluate(condition)
                self.signals_data[f'exit_{rule_id}'] = signal_series
            elif rule_type == 'target_touch':
                # Implement target touch logic
                target_expr = rule['target']
                evaluator = ExpressionEvaluator(self.features_data)
                target_series = evaluator.evaluate(f"close >= {target_expr}")
                self.signals_data[f'exit_{rule_id}'] = target_series
            # Add other exit rule types as needed

        total_entry_signals = sum(self.signals_data[col].sum()
                                for col in self.signals_data.columns
                                if col.startswith('entry_'))
        total_exit_signals = sum(self.signals_data[col].sum()
                               for col in self.signals_data.columns
                               if col.startswith('exit_'))

        self._log(f"✅ Generated {total_entry_signals} entry signals, {total_exit_signals} exit signals")

    def _apply_cooldown(self, signals: pd.Series, cooldown_bars: int) -> pd.Series:
        """Apply cooldown period between signals"""
        result = signals.copy()
        last_signal_idx = -cooldown_bars - 1

        for i, signal in enumerate(signals):
            if signal and (i - last_signal_idx) <= cooldown_bars:
                result.iloc[i] = False  # Suppress signal due to cooldown
            elif signal:
                last_signal_idx = i

        return result

    def _execute_strategy(self):
        """Execute strategy using signals"""
        self._log("⚡ Executing strategy...")

        # Initialize execution engine
        assumptions = self.strategy_spec.get('assumptions', {})
        self.execution_engine = ExecutionEngine(self.features_data, assumptions)

        execution_config = self.strategy_spec.get('execution', {})
        execution_mode = execution_config.get('order_effective', 'next_bar_open')

        # Track open positions
        open_positions = {}  # rule_id -> order_id

        # Process each bar
        for timestamp, bar in self.signals_data.iterrows():
            # Check for entry signals
            for col in bar.index:
                if col.startswith('entry_') and bar[col]:
                    rule_id = col.replace('entry_', '')

                    # Only enter if no existing position for this rule
                    if rule_id not in open_positions:
                        # Determine side from entry rule
                        entry_rule = next((r for r in self.strategy_spec['entry_rules']
                                         if r['id'] == rule_id), None)
                        if entry_rule:
                            side = OrderSide.LONG if entry_rule['side'] == 'long' else OrderSide.SHORT

                            # Place entry order
                            order_id = self.execution_engine.place_order(
                                symbol=self.strategy_spec['universe'][0],  # Simplified
                                side=side,
                                quantity=100,  # Simplified position sizing
                                timestamp=timestamp
                            )
                            open_positions[rule_id] = order_id

            # Check for exit signals
            for col in bar.index:
                if col.startswith('exit_') and bar[col]:
                    rule_id = col.replace('exit_', '')

                    # Find matching open position
                    for entry_rule_id, entry_order_id in list(open_positions.items()):
                        # Simple matching - could be more sophisticated
                        if entry_rule_id in open_positions:
                            # Place exit order
                            entry_order = self.execution_engine._find_order(entry_order_id)
                            if entry_order:
                                exit_side = OrderSide.SHORT if entry_order.side == OrderSide.LONG else OrderSide.LONG

                                exit_order_id = self.execution_engine.place_order(
                                    symbol=entry_order.symbol,
                                    side=exit_side,
                                    quantity=entry_order.quantity,
                                    timestamp=timestamp
                                )

                                # Create trade
                                trade_id = self.execution_engine.create_trade(
                                    entry_order_id, exit_order_id, rule_id
                                )

                                # Remove from open positions
                                del open_positions[entry_rule_id]
                                break

        # Execute all orders
        filled_orders = self.execution_engine.execute_orders(execution_mode)

        self._log(f"✅ Executed {len(filled_orders)} orders, created {len(self.execution_engine.trades)} trades")

    def _generate_reports(self):
        """Generate output reports"""
        self._log("📊 Generating reports...")

        reports = self.test_plan['run'].get('reports', [])

        # Always generate basic outputs
        self._save_signals()
        self._save_trades()
        self._save_equity_curve()
        self._save_summary_report()

        # Generate requested reports
        if 'chart' in reports:
            self._generate_chart()

        if 'trade_analysis' in reports:
            self._generate_trade_analysis()

        self._log("✅ Reports generated")

    def _save_signals(self):
        """Save signals to parquet file"""
        signals_file = os.path.join(self.output_dir, 'signals.parquet')

        # Extract just the signal columns
        signal_cols = [col for col in self.signals_data.columns
                      if col.startswith('entry_') or col.startswith('exit_')]
        signals_df = self.signals_data[signal_cols].copy()

        signals_df.to_parquet(signals_file)
        self._log(f"💾 Saved signals to {signals_file}")

    def _save_trades(self):
        """Save trades to CSV file"""
        trades_file = os.path.join(self.output_dir, 'trades.csv')

        if self.execution_engine.trades:
            trades_df = self.execution_engine.get_trades_dataframe()
            trades_df.to_csv(trades_file, index=False)
            self._log(f"💾 Saved {len(trades_df)} trades to {trades_file}")
        else:
            # Create empty file
            pd.DataFrame().to_csv(trades_file, index=False)
            self._log(f"💾 Saved empty trades file to {trades_file}")

    def _save_equity_curve(self):
        """Save equity curve to CSV file"""
        equity_file = os.path.join(self.output_dir, 'equity_curve.csv')

        equity_df = self.execution_engine.get_equity_curve()
        equity_df.to_csv(equity_file, index=False)
        self._log(f"💾 Saved equity curve to {equity_file}")

    def _save_summary_report(self):
        """Save summary report to JSON file"""
        report_file = os.path.join(self.output_dir, 'report.json')

        # Calculate metrics
        trade_summary = self.execution_engine.get_trade_summary()
        equity_df = self.execution_engine.get_equity_curve()

        # Calculate additional metrics
        if len(equity_df) > 1:
            returns = equity_df['equity'].pct_change().dropna()

            metrics = {
                'total_return_pct': ((equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1) * 100,
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown_pct': self._calculate_max_drawdown(equity_df['equity']),
                'volatility_pct': returns.std() * np.sqrt(252) * 100,
            }
        else:
            metrics = {}

        report = {
            'strategy_name': self.strategy_spec.get('name', 'Unknown'),
            'run_timestamp': datetime.now().isoformat(),
            'data_period': {
                'start': self.test_plan['data']['start'],
                'end': self.test_plan['data']['end'],
                'bars': len(self.data)
            },
            'trade_summary': trade_summary,
            'metrics': metrics,
            'status': 'completed'
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.results = report
        self._log(f"💾 Saved summary report to {report_file}")

    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        return abs(drawdown.min()) * 100

    def _generate_chart(self):
        """Generate strategy chart (placeholder)"""
        chart_file = os.path.join(self.output_dir, 'chart.png')

        # Placeholder - would integrate with WZRD chart templates
        self._log(f"📈 Chart generation placeholder - would save to {chart_file}")

    def _generate_trade_analysis(self):
        """Generate detailed trade analysis"""
        analysis_file = os.path.join(self.output_dir, 'trade_analysis.json')

        # Placeholder for detailed trade analysis
        analysis = {
            'analysis_type': 'trade_analysis',
            'placeholder': True
        }

        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

    def _run_acceptance_tests(self):
        """Run acceptance tests on the strategy"""
        self._log("🧪 Running acceptance tests...")

        acceptance_results = self.acceptance_runner.run_acceptance_tests(
            self.strategy_spec, self.features_data
        )

        # Save acceptance results
        acceptance_file = os.path.join(self.output_dir, 'acceptance_results.json')
        with open(acceptance_file, 'w') as f:
            json.dump(acceptance_results, f, indent=2)

        if acceptance_results['passed']:
            self._log("✅ Acceptance tests passed")
        else:
            self._log(f"❌ Acceptance tests failed: {acceptance_results['errors']}")

    def _log(self, message: str):
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(message)

def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(description='WZRD Strategy Runner')
    parser.add_argument('--strategy', required=True, help='Path to strategy spec JSON file')
    parser.add_argument('--plan', required=True, help='Path to test plan YAML file')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    try:
        # Load inputs
        strategy_spec = load_strategy_spec(args.strategy)
        test_plan = load_test_plan(args.plan)

        # Run strategy
        runner = StrategyRunner(strategy_spec, test_plan, args.output, args.verbose)
        results = runner.run()

        print(f"✅ Strategy run completed successfully")
        print(f"📊 Results saved to: {args.output}")
        print(f"📈 Total trades: {results.get('trade_summary', {}).get('total_trades', 0)}")
        print(f"💰 Total P&L: ${results.get('trade_summary', {}).get('total_pnl', 0):.2f}")

    except Exception as e:
        print(f"❌ Strategy run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()