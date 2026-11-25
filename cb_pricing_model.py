import numpy as np

def price_ccb_four_region_lsm(
    S0: float,
    C0: float,
    FV: float = 100.0,
    maturity_years: float = 2.0,
    r: float = 0.025,
    q: float = 0.0,
    sigma: float = 0.28,
    call_ratio: float = 1.30,
    put_ratio: float = 0.70,
    p_reset: float = 0.80,
    B: float = 108.0,
    call_settle: float = 103.0,
    put_pay: float = 101.0,
    n_steps: int = 24,
    M: int = 2000,
    seed: int = 42
) -> float:
    """
    Chinese Convertible Bond Valuation via Four-Region Segmented Least-Squares Monte Carlo.
    Based on Yu Liu (2026), arXiv:2409.06496v7.
    """
    if S0 <= 0 or C0 <= 0:
        return B
    
    np.random.seed(seed)
    dt = max(0.01, maturity_years / n_steps)
    discount = np.exp(-r * dt)
    
    # 1. GBM path simulation
    Z = np.random.standard_normal((M, n_steps))
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * Z
    
    S = np.zeros((M, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp(log_returns[:, t])
        
    # Window step estimates
    nc = max(2, int(30 / 252 * (n_steps / max(0.5, maturity_years))))
    np_win = max(2, int(30 / 252 * (n_steps / max(0.5, maturity_years))))
    
    # Track conversion price forward under exogenous reset policy
    C = np.full((M, n_steps + 1), C0)
    for t in range(1, n_steps + 1):
        win_start_p = max(0, t - np_win)
        win_S_p = S[:, win_start_p:t]
        put_thresh = put_ratio * C[:, t - 1][:, None]
        Y_t = np.mean(win_S_p < put_thresh, axis=1)
        
        # Reset trigger condition
        reset_triggered = (Y_t >= 0.5)
        do_reset = reset_triggered & (np.random.rand(M) < p_reset)
        
        # New conversion price floor: max(recent average, current price)
        avg_recent = np.mean(S[:, max(0, t-3):t+1], axis=1)
        floor = np.maximum(avg_recent, S[:, t])
        new_C = np.minimum(C[:, t - 1], floor)
        C[:, t] = np.where(do_reset, new_C, C[:, t - 1])
        
    # Terminal payoff V_T = max(B, m_T * S_T)
    m_T = FV / np.maximum(0.01, C[:, n_steps])
    V = np.maximum(B, m_T * S[:, n_steps])
    
    # Backward dynamic programming
    for t in range(n_steps - 1, 0, -1):
        win_c_start = max(0, t - nc)
        win_p_start = max(0, t - np_win)
        
        Ct = C[:, t]
        k_t = call_ratio * Ct
        ptr_t = put_ratio * Ct
        
        F_t = np.mean(S[:, win_c_start:t] > k_t[:, None], axis=1)
        Y_t = np.mean(S[:, win_p_start:t] < ptr_t[:, None], axis=1)
        
        y = discount * V
        St = S[:, t]
        
        # 9-features + intercept
        X = np.column_stack([
            np.ones(M),
            St, St**2,
            F_t, F_t**2,
            Y_t, Y_t**2,
            St * F_t, St * Y_t, F_t * Y_t
        ])
        
        # 4 regions
        r1 = St > k_t
        r2 = (St > Ct) & (St <= k_t)
        r3 = (St > ptr_t) & (St <= Ct)
        r4 = St <= ptr_t
        
        continuation = np.zeros(M)
        for r_mask in [r1, r2, r3, r4]:
            count = np.sum(r_mask)
            if count >= 12:
                X_sub = X[r_mask]
                y_sub = y[r_mask]
                theta, _, _, _ = np.linalg.lstsq(X_sub, y_sub, rcond=None)
                continuation[r_mask] = X_sub @ theta
            elif count > 0:
                continuation[r_mask] = np.mean(y[r_mask])
                
        # Decision operator D_t
        forced_call = F_t >= (15.0 / 30.0)
        m_t = FV / np.maximum(0.01, Ct)
        conv_val = m_t * St
        put_eligible = Y_t >= (30.0 / 30.0)
        
        G_t = np.where(put_eligible, np.maximum(conv_val, put_pay), conv_val)
        
        V = np.where(
            forced_call,
            np.maximum(call_settle, conv_val),
            np.maximum(G_t, continuation)
        )
        
    V0 = float(discount * np.mean(V))
    return round(V0, 2)


def price_ccb_calibrated_defensive(
    S0: float,
    C0: float,
    bond_price: float,
    maturity_years: float = 2.0,
    sigma: float = 0.28,
    FV: float = 100.0,
    r: float = 0.025,
    q: float = 0.0,
    p_reset: float = 0.40,
    sigma_cap: float = 0.35,
    n_steps: int = 24,
    M: int = 1600,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Calibrated Defensive Four-Region LSM Valuation.

    Addresses short-horizon downward-reset execution lag and missing pure-bond cushion
    in high-price / high-premium bonds by applying:
    1) Conservative reset probability (p_reset=0.40) and capped implied vol (<=35%)
    2) Maturity-aware conservative pure-bond floor B_cal = min(106, 99 + 1.8*T)
    3) Downside cushion & premium-lag haircut above ¥115 price and 25% premium.

    Returns (calibrated_price, raw_conservative_lsm, bond_floor_cal, haircut_pct).
    """
    B_cal = round(min(106.0, 99.0 + 1.8 * max(0.35, min(maturity_years, 4.0))), 2)
    sigma_cal = min(float(sigma), sigma_cap)
    raw_cal = price_ccb_four_region_lsm(
        S0=S0, C0=C0, FV=FV, maturity_years=maturity_years,
        r=r, q=q, sigma=sigma_cal,
        call_ratio=1.30, put_ratio=0.70, p_reset=p_reset,
        B=B_cal, n_steps=n_steps, M=M, seed=seed,
    )
    conv_val = (FV / max(0.01, C0)) * S0
    prem_pct = (bond_price - conv_val) / max(0.01, conv_val) * 100.0
    haircut = min(
        0.22,
        0.25 * max(0.0, (bond_price - 115.0) / 100.0)
        + 0.12 * max(0.0, (prem_pct - 25.0) / 100.0),
    )
    cal_price = round(raw_cal * (1.0 - haircut), 2)
    return cal_price, round(raw_cal, 2), B_cal, round(haircut * 100.0, 2)
