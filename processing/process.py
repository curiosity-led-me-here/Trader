import numpy as np
import pandas as pd
import math
from converter import convert, split, attach, save_output

# ---------------- Black-Scholes helpers ----------------
def bs_call_price(S, K, r, sigma, tau):
    if tau <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma*sigma) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    def N(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return S * N(d1) - K * math.exp(-r*tau) * N(d2)

def bs_vega(S, K, r, sigma, tau):
    if tau <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma*sigma) * tau) / (sigma * math.sqrt(tau))
    n = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2*math.pi)
    return S * n(d1) * math.sqrt(tau)

def implied_vol_bisect(mkt_price, S, K, r, tau, tol=1e-6, max_iter=60, low=1e-6, high=5.0):
    intrinsic = max(S - K, 0.0)
    if mkt_price <= intrinsic + 1e-12:
        return 0.0
    price_high = bs_call_price(S, K, r, high, tau)
    if price_high < mkt_price - 1e-9:
        return high
    lo, hi = low, high
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        p = bs_call_price(S, K, r, mid, tau)
        if abs(p - mkt_price) < tol:
            return mid
        if p > mkt_price:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

# ---------------- Main feature computation ----------------
def compute_features(df, r=0.0,
                     compute_analytic_greeks=False, iv_inversion_on=False):
    """
    Input df: ['Datetime','underlying','strike','call','put']
    - expiry_datetime: if given, tau is auto-calculated (minutes to expiry).
    - compute_analytic_greeks: if True and iv_inversion_on True, compute BS greeks from IV.
    - iv_inversion_on: if True, compute implied vol per row.
    """
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values(['Datetime','underlying','strike']).reset_index(drop=True)

    # -------- Tau (time to expiry) --------
    expiry = df['Datetime'].iloc[-1]
    df['tau_minutes'] = (expiry - df['Datetime']).dt.total_seconds() / 60.0
    df.loc[df['tau_minutes'] < 0, 'tau_minutes'] = 0

    # -------- Cross-sectional Greeks (across strikes) --------
    def cross_sectional(group):
        group = group.sort_values('strike')
        K = group['strike'].to_numpy(float)
        C = group['call'].to_numpy(float)
        P = group['put'].to_numpy(float)

        # log transforms for relative stability
        logC = np.log(C + 1e-9)
        logP = np.log(P + 1e-9)

        dlogC_dK  = np.gradient(logC, K, edge_order=2)
        d2logC_dK2 = np.gradient(dlogC_dK, K, edge_order=2)
        dlogP_dK  = np.gradient(logP, K, edge_order=2)
        d2logP_dK2 = np.gradient(dlogP_dK, K, edge_order=2)

        return pd.DataFrame({
            'dC_dK_pct': dlogC_dK,
            'd2C_dK2_pct': d2logC_dK2,
            'dP_dK_pct': dlogP_dK,
            'd2P_dK2_pct': d2logP_dK2
        }, index=group.index)

    cross_df = df.groupby(['Datetime'], group_keys=False).apply(cross_sectional)
    df = df.join(cross_df)
    df['call_delta_proxy'] = -df['dC_dK_pct']
    df['call_gamma_proxy'] = df['d2C_dK2_pct']
    df['put_delta_proxy']  = df['dP_dK_pct']
    df['put_gamma_proxy']  = df['d2P_dK2_pct']

    df['call_vega_proxy']  = df['call_gamma_proxy'] * (df['underlying']**2)
    df['put_vega_proxy']   = df['put_gamma_proxy']  * (df['underlying']**2)

    # -------- Temporal Greeks (same strike across time) --------
    df = df.sort_values(['underlying','strike','Datetime']).reset_index(drop=True)

    def time_diffs(group):
        group = group.sort_values('Datetime')
        S = group['underlying'].to_numpy(float)
        C = group['call'].to_numpy(float)
        P = group['put'].to_numpy(float)
        t = group['Datetime'].astype('int64') / 1e9  # seconds

        # log transforms
        logC = np.log(C + 1e-9)
        logP = np.log(P + 1e-9)

        # elasticities wrt S
        dlogC_dS  = np.gradient(logC, S, edge_order=2)
        d2logC_dS2 = np.gradient(dlogC_dS, S, edge_order=2)

        # time decay (per second, but in % space)
        theta_call = np.gradient(logC, t)
        theta_put  = np.gradient(logP, t)

        return pd.DataFrame({
            'dC_dS_time_pct': dlogC_dS,
            'd2C_dS2_time_pct': d2logC_dS2,
            'theta_call_pct_per_sec': theta_call,
            'theta_put_pct_per_sec': theta_put
        }, index=group.index)

    time_df = df.groupby(['strike'], group_keys=False).apply(time_diffs)
    df = df.join(time_df)
    df['theta_call_per_min'] = df['theta_call_pct_per_sec'] * 60.0
    df['theta_put_per_min']  = df['theta_put_pct_per_sec']  * 60.0

    if iv_inversion_on:
        ivs, deltas, gammas, vegas, thetas = [], [], [], [], []
        for _, row in df.iterrows():
            S, K, tau_min, mkt_price = float(row['underlying']), float(row['strike']), row['tau_minutes'], float(row['call'])
            tau = max(1e-6, tau_min) / (252.0*6.5*60.0) if not np.isnan(tau_min) else None
            iv = implied_vol_bisect(mkt_price, S, K, r, tau) if tau else np.nan
            ivs.append(iv)
            if compute_analytic_greeks and not np.isnan(iv):
                epsS = max(1e-3, 1e-4*S)
                p_plus, p_minus = bs_call_price(S+epsS, K, r, iv, tau), bs_call_price(S-epsS, K, r, iv, tau)
                deltas.append((p_plus - p_minus) / (2*epsS))
                gammas.append((p_plus - 2*mkt_price + p_minus) / (epsS**2))
                vegas.append(bs_vega(S, K, r, iv, tau))
                eps_t = 1.0/(252.0*6.5*60.0)
                price_t_plus = bs_call_price(S, K, r, iv, max(tau-eps_t, 0))
                thetas.append((price_t_plus - mkt_price) / eps_t)
            else:
                deltas.append(np.nan); gammas.append(np.nan); vegas.append(np.nan); thetas.append(np.nan)
        df['implied_vol'], df['bs_delta'], df['bs_gamma'], df['bs_vega'], df['bs_theta'] = ivs, deltas, gammas, vegas, thetas

    return df.sort_values(['Datetime','underlying','strike']).reset_index(drop=True)


path = r"/Users/ashu/Documents/Trader/NIFTY 50/2024-11-28_16-38-19.xlsx"
data = convert(path)
data = compute_features(data)
data.index.name = None
save_output(data, save_path=r'/Users/ashu/Downloads/output_2024-11-28_16-38-19.xlsx')