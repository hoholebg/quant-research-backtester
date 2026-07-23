"""
High-Performance Numba JIT Accelerated Monte Carlo Path Generator
"""

import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def simulate_geometric_brownian_motion(S0: float, mu: float, sigma: float, T: float, n_sims: int, n_steps: int) -> np.ndarray:
    """Numba-accelerated GBM path simulation."""
    dt = T / n_steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * np.sqrt(dt)
    
    paths = np.empty((n_sims, n_steps + 1))
    for i in range(n_sims):
        paths[i, 0] = S0
        for t in range(1, n_steps + 1):
            z = np.random.normal(0.0, 1.0)
            paths[i, t] = paths[i, t - 1] * np.exp(drift + vol_dt * z)
            
    return paths

@jit(nopython=True, fastmath=True)
def compute_portfolio_var(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Numba-accelerated Value-at-Risk (VaR)."""
    sorted_ret = np.sort(returns)
    idx = int((1.0 - alpha) * len(returns))
    return -sorted_ret[idx]
