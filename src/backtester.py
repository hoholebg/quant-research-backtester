"""
Systematic Long/Short Vectorized Backtester for FX & Fixed Income Spreads
"""

import numpy as np
import pandas as pd

class SystematicBacktester:
    """Backtests cross-asset quantitative trading strategies with transaction costs."""
    
    def __init__(self, transaction_cost_bps: float = 5.0):
        self.tc = transaction_cost_bps / 10000.0

    def run_strategy(self, price_series: pd.Series, signals: pd.Series) -> pd.DataFrame:
        asset_returns = price_series.pct_change().fillna(0)
        positions = signals.shift(1).fillna(0)
        
        # Position turnover costs
        turnover = np.abs(positions - positions.shift(1).fillna(0))
        costs = turnover * self.tc
        
        strategy_returns = positions * asset_returns - costs
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Performance metrics
        sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-8)) * np.sqrt(252)
        max_dd = (cumulative_returns / cumulative_returns.cummax() - 1.0).min()
        
        return pd.DataFrame({
            "asset_price": price_series,
            "signal": signals,
            "strategy_return": strategy_returns,
            "equity_curve": cumulative_returns
        }), {"sharpe_ratio": float(sharpe), "max_drawdown": float(max_dd)}
