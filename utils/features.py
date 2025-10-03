#!/usr/bin/env python3
"""
WZRD Feature Engineering System

Provides technical indicators and features for strategy specifications.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import warnings

class FeatureEngine:
    """Computes technical features from OHLCV data"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engine with OHLCV data

        Args:
            data: DataFrame with columns [open, high, low, close, volume]
        """
        self.data = data.copy()
        self.features = {}

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def compute_features(self, feature_specs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compute all features specified in feature_specs

        Args:
            feature_specs: Dictionary of feature definitions

        Returns:
            DataFrame with original data plus computed features
        """
        result = self.data.copy()

        for feature_name, spec in feature_specs.items():
            try:
                feature_values = self._compute_single_feature(spec)
                result[feature_name] = feature_values
                self.features[feature_name] = feature_values
            except Exception as e:
                warnings.warn(f"Failed to compute feature '{feature_name}': {e}")
                result[feature_name] = np.nan

        return result

    def _compute_single_feature(self, spec: Dict[str, Any]) -> pd.Series:
        """Compute a single feature based on specification"""
        feature_type = spec['type']

        if feature_type == 'session_vwap':
            return self._session_vwap()
        elif feature_type == 'zscore':
            return self._zscore(spec)
        elif feature_type == 'sma':
            return self._sma(spec)
        elif feature_type == 'ema':
            return self._ema(spec)
        elif feature_type == 'rolling_std':
            return self._rolling_std(spec)
        elif feature_type == 'rolling_min':
            return self._rolling_min(spec)
        elif feature_type == 'rolling_max':
            return self._rolling_max(spec)
        elif feature_type == 'rsi':
            return self._rsi(spec)
        elif feature_type == 'bbands':
            return self._bollinger_bands(spec)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def _session_vwap(self) -> pd.Series:
        """Compute session VWAP (Volume Weighted Average Price)"""
        # Reset VWAP at start of each trading day
        dates = self.data.index.date if hasattr(self.data.index, 'date') else self.data.index

        vwap_values = []
        current_date = None
        cumulative_pv = 0
        cumulative_volume = 0

        for i, (timestamp, row) in enumerate(self.data.iterrows()):
            # Extract date (handle both datetime index and regular index)
            if hasattr(timestamp, 'date'):
                row_date = timestamp.date()
            else:
                row_date = dates[i] if hasattr(dates, '__getitem__') else timestamp

            # Reset on new trading day
            if current_date != row_date:
                current_date = row_date
                cumulative_pv = 0
                cumulative_volume = 0

            # Update cumulative values
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            pv = typical_price * row['volume']
            cumulative_pv += pv
            cumulative_volume += row['volume']

            # Calculate VWAP
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
            else:
                vwap = typical_price

            vwap_values.append(vwap)

        return pd.Series(vwap_values, index=self.data.index, name='vwap')

    def _zscore(self, spec: Dict[str, Any]) -> pd.Series:
        """Compute z-score of an input expression"""
        input_expr = spec['input']
        window = spec['window']

        # Evaluate input expression
        # This is a simplified implementation - a full version would use a proper expression parser
        if input_expr == 'close - vwap':
            if 'vwap' not in self.features:
                # Compute VWAP if not already computed
                self.features['vwap'] = self._session_vwap()
            values = self.data['close'] - self.features['vwap']
        elif input_expr == 'close':
            values = self.data['close']
        else:
            # Fallback: try to evaluate as pandas expression
            try:
                # Replace feature names with actual values
                eval_expr = input_expr
                for feature_name, feature_values in self.features.items():
                    eval_expr = eval_expr.replace(feature_name, f"self.features['{feature_name}']")

                # Basic eval (security risk in production - use proper expression parser)
                values = eval(eval_expr)
            except:
                raise ValueError(f"Cannot evaluate input expression: {input_expr}")

        # Compute rolling z-score
        rolling_mean = values.rolling(window=window).mean()
        rolling_std = values.rolling(window=window).std()
        zscore = (values - rolling_mean) / rolling_std

        return zscore

    def _sma(self, spec: Dict[str, Any]) -> pd.Series:
        """Compute Simple Moving Average"""
        period = spec['period']
        input_col = spec.get('input', 'close')

        if input_col in self.data.columns:
            return self.data[input_col].rolling(window=period).mean()
        else:
            raise ValueError(f"Input column '{input_col}' not found")

    def _ema(self, spec: Dict[str, Any]) -> pd.Series:
        """Compute Exponential Moving Average"""
        period = spec['period']
        input_col = spec.get('input', 'close')

        if input_col in self.data.columns:
            return self.data[input_col].ewm(span=period).mean()
        else:
            raise ValueError(f"Input column '{input_col}' not found")

    def _rolling_std(self, spec: Dict[str, Any]) -> pd.Series:
        """Compute Rolling Standard Deviation"""
        window = spec['window']
        input_col = spec.get('input', 'close')

        if input_col in self.data.columns:
            return self.data[input_col].rolling(window=window).std()
        else:
            raise ValueError(f"Input column '{input_col}' not found")

    def _rolling_min(self, spec: Dict[str, Any]) -> pd.Series:
        """Compute Rolling Minimum"""
        window = spec['window']
        input_col = spec.get('input', 'close')

        if input_col in self.data.columns:
            return self.data[input_col].rolling(window=window).min()
        else:
            raise ValueError(f"Input column '{input_col}' not found")

    def _rolling_max(self, spec: Dict[str, Any]) -> pd.Series:
        """Compute Rolling Maximum"""
        window = spec['window']
        input_col = spec.get('input', 'close')

        if input_col in self.data.columns:
            return self.data[input_col].rolling(window=window).max()
        else:
            raise ValueError(f"Input column '{input_col}' not found")

    def _rsi(self, spec: Dict[str, Any]) -> pd.Series:
        """Compute Relative Strength Index"""
        period = spec.get('period', 14)
        input_col = spec.get('input', 'close')

        if input_col not in self.data.columns:
            raise ValueError(f"Input column '{input_col}' not found")

        prices = self.data[input_col]
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _bollinger_bands(self, spec: Dict[str, Any]) -> pd.Series:
        """Compute Bollinger Bands (returns middle band)"""
        period = spec.get('period', 20)
        std_dev = spec.get('std_dev', 2)
        input_col = spec.get('input', 'close')

        if input_col not in self.data.columns:
            raise ValueError(f"Input column '{input_col}' not found")

        prices = self.data[input_col]
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        # Return middle band (SMA) - could be extended to return upper/lower bands
        return sma

class ExpressionEvaluator:
    """Evaluates boolean expressions for entry/exit rules"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize expression evaluator

        Args:
            data: DataFrame with OHLCV data and computed features
        """
        self.data = data

    def evaluate(self, expression: str) -> pd.Series:
        """
        Evaluate boolean expression

        Args:
            expression: Boolean expression string (e.g., "z <= -2.0")

        Returns:
            Boolean Series indicating where condition is true
        """
        try:
            # Create namespace with column names
            namespace = {}
            for col in self.data.columns:
                namespace[col] = self.data[col]

            # Evaluate expression
            # Note: In production, use a proper expression parser for security
            result = eval(expression, {"__builtins__": {}}, namespace)

            # Ensure result is a boolean Series
            if isinstance(result, pd.Series):
                return result.astype(bool)
            else:
                # Scalar result - broadcast to Series
                return pd.Series([bool(result)] * len(self.data), index=self.data.index)

        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}")

def compute_strategy_features(data: pd.DataFrame,
                            feature_specs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convenience function to compute features for a strategy

    Args:
        data: OHLCV DataFrame
        feature_specs: Feature specifications from strategy spec

    Returns:
        DataFrame with original data plus computed features
    """
    engine = FeatureEngine(data)
    return engine.compute_features(feature_specs)

if __name__ == "__main__":
    # Example usage
    print("🔧 WZRD Feature Engineering System")
    print("Available features:")
    print("- session_vwap: Session Volume Weighted Average Price")
    print("- zscore: Z-score with rolling window")
    print("- sma: Simple Moving Average")
    print("- ema: Exponential Moving Average")
    print("- rolling_std: Rolling Standard Deviation")
    print("- rolling_min/max: Rolling Min/Max")
    print("- rsi: Relative Strength Index")
    print("- bbands: Bollinger Bands")