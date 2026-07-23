import time
import numpy as np
import pandas as pd
from src.numba_monte_carlo import simulate_geometric_brownian_motion, compute_portfolio_var
from src.backtester import SystematicBacktester

def main():
    print("=== Quant Research Suite & Numba Monte Carlo Engine ===")
    
    # 1. Benchmark Numba Acceleration
    print("Simulating 100,000 asset paths (252 steps) via Numba JIT...")
    t0 = time.time()
    paths = simulate_geometric_brownian_motion(S0=100.0, mu=0.05, sigma=0.20, T=1.0, n_sims=100000, n_steps=252)
    elapsed = time.time() - t0
    print(f"Paths generated in {elapsed*1000:.2f} ms!")
    
    final_returns = (paths[:, -1] / paths[:, 0]) - 1.0
    var95 = compute_portfolio_var(final_returns, alpha=0.95)
    print(f"Calculated 95% 1-Year Value-at-Risk (VaR): {var95*100:.2f}%")

    # 2. Long/Short Systematic FX Strategy Demo
    print("\n=== Backtesting Systematic Mean-Reversion Strategy ===")
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    fx_prices = pd.Series(1.0 + np.cumsum(np.random.normal(0, 0.005, 500)), index=dates)
    
    # Simple Z-score mean reversion signal
    z_score = (fx_prices - fx_prices.rolling(20).mean()) / fx_prices.rolling(20).std()
    signals = pd.Series(0, index=dates)
    signals[z_score < -1.0] = 1   # Long
    signals[z_score > 1.0] = -1   # Short

    backtester = SystematicBacktester(transaction_cost_bps=2.0)
    df, metrics = backtester.run_strategy(fx_prices, signals)
    
    print(f"Annualized Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown:        {metrics['max_drawdown']*100:.2f}%")

if __name__ == "__main__":
    main()
