import numpy as np
import pandas as pd
import math
from functools import partial

# ----------------------
# Black-Scholes helpers (for optional IV inversion + analytic greeks)
# ----------------------
def bs_call_price(S, K, r, sigma, tau):
    # tau in years
    if tau <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma*sigma) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    from math import erf
    def N(x): return 0.5*(1.0 + math.erf(x / math.sqrt(2.0)))
    return S * N(d1) - K * math.exp(-r*tau) * N(d2)

def bs_vega(S, K, r, sigma, tau):
    # returns vega (derivative of price wrt sigma), per 1 vol point (not %)
    if tau <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma*sigma) * tau) / (sigma * math.sqrt(tau))
    from math import exp, pi
    n = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2*math.pi)
    return S * n(d1) * math.sqrt(tau)

def implied_vol_bisect(mkt_price, S, K, r, tau, q=0.0,
                       tol=1e-6, max_iter=60, low=1e-6, high=5.0):
    # bisection for call (if mkt_price < intrinsic returns 0.0)
    intrinsic = max(S - K, 0.0)
    if mkt_price <= intrinsic + 1e-12:
        return 0.0
    # check high bound
    price_high = bs_call_price(S, K, r, high, tau)
    if price_high < mkt_price - 1e-9:
        return np.nan
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

# ----------------------
# Main routine: compute features
# ----------------------
def compute_features(df,
                     r=0.0,
                     default_tau_minutes=None,
                     compute_analytic_greeks=False,
                     iv_inversion_on=False):
    """
    Input df: columns ['Datetime','underlying','strike','call','put']
    - default_tau_minutes: if dataframe has no tau column, use this as default expiry (minutes) for analytic greeks / IV.
    - compute_analytic_greeks: if True and iv_inversion_on True, compute BS analytic greeks from implied vol.
    - iv_inversion_on: if True, attempt to compute implied vol per (Datetime, strike). Needs tau (column or default).
    Returns: df with added columns (moneyness, cross-sectional dC/dK, d2C/dK2, temporal dC/dS, d2C/dS2, theta, optional iv & bs greeks)
    """
    df = df.copy()
    # ensure sorting
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values(['Datetime','underlying','strike']).reset_index(drop=True)

    # row-wise moneyness
    df['moneyness'] = np.log(df['underlying'] / df['strike'])

    # allow per-row tau (in minutes) or use default
    if 'tau_minutes' in df.columns:
        df['tau_minutes'] = df['tau_minutes'].astype(float)
    else:
        if default_tau_minutes is None:
            # if no tau provided and not needed, set to NaN
            df['tau_minutes'] = np.nan
        else:
            df['tau_minutes'] = float(default_tau_minutes)

    # -------- Cross-sectional derivatives across strikes at same Datetime (dC/dK, d2C/dK2) --------
    def cross_sectional(group):
        # group is a snapshot: all strikes for one Datetime & underlying
        # ensure ordering by strike ascending
        group = group.sort_values('strike')
        K = group['strike'].to_numpy(dtype=float)
        C = group['call'].to_numpy(dtype=float)
        P = group['put'].to_numpy(dtype=float)

        # gradients wrt strike K
        # central differences via np.gradient handle nonuniform spacing
        dC_dK = np.gradient(C, K)
        dP_dK = np.gradient(P, K)
        d2C_dK2 = np.gradient(dC_dK, K)
        d2P_dK2 = np.gradient(dP_dK, K)

        # we will return -dC/dK (Breeden) as call_delta_proxy and second derivative as density proxy
        out = pd.DataFrame({
            'dC_dK': dC_dK,
            'd2C_dK2': d2C_dK2,
            'dP_dK': dP_dK,
            'd2P_dK2': d2P_dK2
        }, index=group.index)
        return out

    cross_df = df.groupby(['Datetime','underlying'], group_keys=False).apply(cross_sectional)
    df = df.join(cross_df)

    # optional: proxies derived
    # call_delta_proxy ≈ -dC/dK (Breeden) ; call_gamma_proxy ≈ d2C/dK2
    df['call_delta_proxy'] = -df['dC_dK']
    df['call_gamma_proxy'] = df['d2C_dK2']
    df['put_delta_proxy'] = df['dP_dK']
    df['put_gamma_proxy'] = df['d2P_dK2']
    # vega proxy via gamma*S^2 (heuristic)
    df['call_vega_proxy'] = df['call_gamma_proxy'] * (df['underlying']**2)
    df['put_vega_proxy']  = df['put_gamma_proxy'] * (df['underlying']**2)

    # -------- Temporal derivatives across time for the SAME strike (Delta wrt S, Gamma wrt S, Theta) --------
    # We'll compute finite differences for each (underlying, strike) chain across time
    df = df.sort_values(['underlying','strike','Datetime']).reset_index(drop=True)

    # helper to compute time-based finite diffs for a series
    def time_diffs(group):
        # group sorted by Datetime for a given underlying+strike
        S = group['underlying'].to_numpy(dtype=float)
        C = group['call'].to_numpy(dtype=float)
        P = group['put'].to_numpy(dtype=float)
        t = group['Datetime'].to_numpy(dtype='datetime64[ns]').astype('datetime64[s]').astype(np.int64).astype(float) # seconds

        # compute dC/dS using central differences where possible
        # dC/dS approx gradient(C) / gradient(S)
        # handle cases where S doesn't change (avoid div by zero): use price deltas instead
        dC_dt = np.gradient(C, t)  # price change per second
        dS_dt = np.gradient(S, t)  # underlying change per second
        with np.errstate(divide='ignore', invalid='ignore'):
            dC_dS = np.where(np.abs(dS_dt) > 1e-12, dC_dt / dS_dt, np.nan)

        # second derivative wrt S (Gamma) approximated by second time-differencing trick:
        # compute dC_dS as above then gradient wrt S via mapping S->dC_dS
        # But safer: compute central finite diff of C wrt S across nearby time points:
        # Use numpy.gradient(C, S) directly which handles nonuniform S spacing; where S constant, result NaN
        try:
            dC_dS_alt = np.gradient(C, S)
        except Exception:
            dC_dS_alt = np.full_like(C, np.nan)
        # prefer alt if it yields finite values
        mask_valid = np.isfinite(dC_dS_alt)
        dC_dS[mask_valid] = dC_dS_alt[mask_valid]

        # Gamma: second derivative d2C/dS2
        d2C_dS2 = np.full_like(C, np.nan)
        try:
            # if S has repeated values, gradient will produce nan; np.gradient handles non-linear S
            dC_dS_for_gamma = np.gradient(C, S)
            d2C_dS2 = np.gradient(dC_dS_for_gamma, S)
        except Exception:
            d2C_dS2 = np.full_like(C, np.nan)

        # Theta: dC/dt (price decay wrt time) per second -> convert to per minute by *60 if desired
        theta_per_second_call = np.gradient(C, t)
        theta_per_second_put  = np.gradient(P, t)

        return pd.DataFrame({
            'dC_dS_time': dC_dS,
            'd2C_dS2_time': d2C_dS2,
            'theta_call_per_second': theta_per_second_call,
            'theta_put_per_second': theta_per_second_put
        }, index=group.index)

    time_df = df.groupby(['underlying','strike'], group_keys=False).apply(time_diffs)
    df = df.join(time_df)

    # convert some time-based to per-minute if desired
    df['theta_call_per_min'] = df['theta_call_per_second'] * 60.0
    df['theta_put_per_min']  = df['theta_put_per_second'] * 60.0

    # -------- Optional: implied vol inversion + analytic BS greeks (if requested) --------
    if iv_inversion_on:
        # prepare vectorized inversion (slower in Python but workable)
        ivs = []
        bs_delta_list = []
        bs_gamma_list = []
        bs_vega_list = []
        bs_theta_list = []
        for idx, row in df.iterrows():
            S = float(row['underlying'])
            K = float(row['strike'])
            tau_min = row['tau_minutes']
            if np.isnan(tau_min):
                tau = None
            else:
                tau = max(1e-6, tau_min) / (252.0 * 6.5 * 60.0)  # minutes->years approx
            mkt_price = float(row['call'])
            if tau is None:
                iv = np.nan
            else:
                iv = implied_vol_bisect(mkt_price, S, K, r, tau)
            ivs.append(iv)
            if compute_analytic_greeks and (not np.isnan(iv)):
                # compute analytic delta/gamma/vega/theta via small numeric derivatives around sigma (or closed form)
                # Here we'll compute vega via bs_vega and delta/gamma via central difference on BS price w.r.t S
                # delta ≈ dBS/dS
                epsS = max(1e-3, 1e-4 * S)
                p_plus = bs_call_price(S+epsS, K, r, iv, tau)
                p_minus = bs_call_price(S-epsS, K, r, iv, tau)
                delta_bs = (p_plus - p_minus) / (2*epsS)
                # gamma via second derivative
                gamma_bs = (p_plus - 2* mkt_price + p_minus) / (epsS**2)
                vega_bs  = bs_vega(S, K, r, iv, tau)
                # theta approximate via price diff over a small dt (1 minute)
                eps_t = 1.0/(252.0*6.5*60.0) # 1 minute in years
                price_t_plus = bs_call_price(S, K, r, iv, max(tau-eps_t, 0.0))
                theta_bs = (price_t_plus - mkt_price) / eps_t  # approx dV/dt
                bs_delta_list.append(delta_bs)
                bs_gamma_list.append(gamma_bs)
                bs_vega_list.append(vega_bs)
                bs_theta_list.append(theta_bs)
            else:
                bs_delta_list.append(np.nan)
                bs_gamma_list.append(np.nan)
                bs_vega_list.append(np.nan)
                bs_theta_list.append(np.nan)

        df['implied_vol'] = ivs
        if compute_analytic_greeks:
            df['bs_delta'] = bs_delta_list
            df['bs_gamma'] = bs_gamma_list
            df['bs_vega']  = bs_vega_list
            df['bs_theta'] = bs_theta_list

    # final sort back to Datetime-major order
    df = df.sort_values(['Datetime','underlying','strike']).reset_index(drop=True)
    return df

# ----------------------
# Usage example:
# df must have columns: 'Datetime','underlying','strike','call','put'
# call the function:
# out = compute_features(df, r=0.0, default_tau_minutes=60, compute_analytic_greeks=True, iv_inversion_on=False)
# ----------------------