#!/usr/bin/env python3
"""
WZRD Execution Engine

Simulates order execution with realistic slippage and fees.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

class OrderSide(Enum):
    """Order side enumeration"""
    LONG = "long"
    SHORT = "short"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class Order:
    """Represents a trading order"""

    def __init__(self, order_id: str, symbol: str, side: OrderSide,
                 quantity: int, order_type: OrderType = OrderType.MARKET,
                 price: Optional[float] = None, timestamp: Optional[pd.Timestamp] = None):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.timestamp = timestamp or pd.Timestamp.now()
        self.status = OrderStatus.PENDING
        self.fill_price = None
        self.fill_timestamp = None
        self.fees = 0.0
        self.slippage = 0.0

class Trade:
    """Represents a completed trade"""

    def __init__(self, trade_id: str, entry_order: Order, exit_order: Order,
                 rule_id: str = None):
        self.trade_id = trade_id
        self.entry_order = entry_order
        self.exit_order = exit_order
        self.rule_id = rule_id
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.duration_bars = 0
        self.duration_minutes = 0

    def calculate_pnl(self):
        """Calculate trade P&L"""
        if (self.entry_order.status == OrderStatus.FILLED and
                self.exit_order.status == OrderStatus.FILLED):

            entry_price = self.entry_order.fill_price
            exit_price = self.exit_order.fill_price
            quantity = self.entry_order.quantity

            if self.entry_order.side == OrderSide.LONG:
                self.pnl = (exit_price - entry_price) * quantity
            else:  # SHORT
                self.pnl = (entry_price - exit_price) * quantity

            # Subtract fees
            self.pnl -= (self.entry_order.fees + self.exit_order.fees)

            # Calculate percentage
            if entry_price > 0:
                self.pnl_pct = (self.pnl / (entry_price * quantity)) * 100

            # Calculate duration
            if self.entry_order.fill_timestamp and self.exit_order.fill_timestamp:
                duration = self.exit_order.fill_timestamp - self.entry_order.fill_timestamp
                self.duration_minutes = duration.total_seconds() / 60

class ExecutionEngine:
    """Simulates realistic order execution"""

    def __init__(self, data: pd.DataFrame, assumptions: Dict[str, Any]):
        """
        Initialize execution engine

        Args:
            data: OHLCV DataFrame with timestamp index
            assumptions: Execution assumptions (slippage, fees, etc.)
        """
        self.data = data
        self.assumptions = assumptions
        self.orders = []
        self.trades = []
        self.positions = {}  # symbol -> current position
        self.equity_curve = []
        self.initial_capital = 100000  # Default initial capital
        self.current_capital = self.initial_capital

        # Extract assumptions
        self.slippage_bps = assumptions.get('slippage_bps', 1)
        self.fee_per_share = assumptions.get('fee_per_share', 0.0035)

    def place_order(self, symbol: str, side: OrderSide, quantity: int,
                   order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None,
                   timestamp: Optional[pd.Timestamp] = None) -> str:
        """
        Place an order

        Args:
            symbol: Trading symbol
            side: Order side (long/short)
            quantity: Number of shares
            order_type: Order type
            price: Limit/stop price (if applicable)
            timestamp: Order timestamp

        Returns:
            Order ID
        """
        order_id = f"ORDER_{len(self.orders) + 1:06d}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            timestamp=timestamp
        )

        self.orders.append(order)
        return order_id

    def execute_orders(self, execution_mode: str = "next_bar_open") -> List[Order]:
        """
        Execute pending orders

        Args:
            execution_mode: When orders execute ("same_bar_close", "next_bar_open", "next_bar_close")

        Returns:
            List of filled orders
        """
        filled_orders = []

        for order in self.orders:
            if order.status == OrderStatus.PENDING:
                filled = self._execute_single_order(order, execution_mode)
                if filled:
                    filled_orders.append(order)

        return filled_orders

    def _execute_single_order(self, order: Order, execution_mode: str) -> bool:
        """Execute a single order"""
        try:
            # Find the bar for execution
            if execution_mode == "same_bar_close":
                execution_bar = self._find_bar_by_timestamp(order.timestamp)
                execution_price = execution_bar['close']
            elif execution_mode == "next_bar_open":
                next_bar = self._find_next_bar(order.timestamp)
                if next_bar is None:
                    return False
                execution_price = next_bar['open']
            elif execution_mode == "next_bar_close":
                next_bar = self._find_next_bar(order.timestamp)
                if next_bar is None:
                    return False
                execution_price = next_bar['close']
            else:
                raise ValueError(f"Unknown execution mode: {execution_mode}")

            # Apply slippage
            slippage_amount = execution_price * (self.slippage_bps / 10000)
            if order.side == OrderSide.LONG:
                execution_price += slippage_amount  # Pay higher for buys
            else:
                execution_price -= slippage_amount  # Receive lower for sells

            # Calculate fees
            fees = order.quantity * self.fee_per_share

            # Fill the order
            order.fill_price = execution_price
            order.fill_timestamp = order.timestamp  # Simplified
            order.fees = fees
            order.slippage = slippage_amount
            order.status = OrderStatus.FILLED

            # Update position
            current_position = self.positions.get(order.symbol, 0)
            if order.side == OrderSide.LONG:
                self.positions[order.symbol] = current_position + order.quantity
            else:
                self.positions[order.symbol] = current_position - order.quantity

            # Update capital
            cost = execution_price * order.quantity + fees
            if order.side == OrderSide.LONG:
                self.current_capital -= cost
            else:
                self.current_capital += cost

            return True

        except Exception as e:
            order.status = OrderStatus.REJECTED
            return False

    def _find_bar_by_timestamp(self, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        """Find bar by timestamp"""
        if timestamp in self.data.index:
            return self.data.loc[timestamp]
        else:
            # Find nearest bar
            nearest_idx = self.data.index.searchsorted(timestamp)
            if nearest_idx < len(self.data):
                return self.data.iloc[nearest_idx]
        return None

    def _find_next_bar(self, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        """Find next bar after timestamp"""
        next_idx = self.data.index.searchsorted(timestamp, side='right')
        if next_idx < len(self.data):
            return self.data.iloc[next_idx]
        return None

    def create_trade(self, entry_order_id: str, exit_order_id: str,
                    rule_id: str = None) -> str:
        """
        Create a trade from entry and exit orders

        Args:
            entry_order_id: Entry order ID
            exit_order_id: Exit order ID
            rule_id: Rule that generated the trade

        Returns:
            Trade ID
        """
        entry_order = self._find_order(entry_order_id)
        exit_order = self._find_order(exit_order_id)

        if not entry_order or not exit_order:
            raise ValueError("Invalid order IDs")

        trade_id = f"TRADE_{len(self.trades) + 1:06d}"

        trade = Trade(
            trade_id=trade_id,
            entry_order=entry_order,
            exit_order=exit_order,
            rule_id=rule_id
        )

        trade.calculate_pnl()
        self.trades.append(trade)

        return trade_id

    def _find_order(self, order_id: str) -> Optional[Order]:
        """Find order by ID"""
        for order in self.orders:
            if order.order_id == order_id:
                return order
        return None

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve over time"""
        equity_data = []

        for timestamp, bar in self.data.iterrows():
            # Calculate current equity (cash + positions)
            equity = self.current_capital

            for symbol, position in self.positions.items():
                if position != 0:
                    market_value = position * bar['close']  # Simplified
                    equity += market_value

            equity_data.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': self.current_capital,
                'positions_value': equity - self.current_capital
            })

        return pd.DataFrame(equity_data)

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all trades"""
        if not self.trades:
            return {"total_trades": 0}

        pnls = [trade.pnl for trade in self.trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]

        summary = {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) if self.trades else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "avg_winning_trade": np.mean(winning_trades) if winning_trades else 0,
            "avg_losing_trade": np.mean(losing_trades) if losing_trades else 0,
            "profit_factor": (sum(winning_trades) / abs(sum(losing_trades))
                            if losing_trades else float('inf'))
        }

        return summary

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        trades_data = []

        for trade in self.trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'symbol': trade.entry_order.symbol,
                'side': trade.entry_order.side.value,
                'quantity': trade.entry_order.quantity,
                'entry_time': trade.entry_order.fill_timestamp,
                'entry_price': trade.entry_order.fill_price,
                'exit_time': trade.exit_order.fill_timestamp,
                'exit_price': trade.exit_order.fill_price,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'duration_minutes': trade.duration_minutes,
                'rule_id': trade.rule_id,
                'entry_fees': trade.entry_order.fees,
                'exit_fees': trade.exit_order.fees,
                'total_fees': trade.entry_order.fees + trade.exit_order.fees
            })

        return pd.DataFrame(trades_data)

if __name__ == "__main__":
    print("⚡ WZRD Execution Engine")
    print("Features:")
    print("- Realistic slippage simulation")
    print("- Commission fees")
    print("- Multiple execution modes")
    print("- Position tracking")
    print("- Trade P&L calculation")
    print("- Equity curve generation")