# Option Pricing Dashboard using Streamlit
# This script provides an interactive interface to price options
# using several methods implemented in the Option_Pricing notebook.

import time
import warnings
import numpy as np
from scipy.stats import norm

try:
    import streamlit as st
except ImportError:  # streamlit might not be installed
    st = None


def black_scholes_price(S, K, r, sigma, T, option_type="call"):
    """Black-Scholes-Merton formula for European options."""
    if T <= 0:
        if option_type.lower() == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def european_monte_carlo(S, K, r, sigma, T, n_paths=10000, option_type="call", seed=42):
    """Monte Carlo pricing for European options."""
    np.random.seed(seed)
    Z = np.random.randn(n_paths)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)


def binomial_option_price(S, K, r, sigma, T, steps=100, option_type="call", american=False):
    """Binomial tree option pricing supporting European and American options."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    prices = np.zeros(steps + 1)
    for i in range(steps + 1):
        ST = S * (u ** (steps - i)) * (d ** i)
        if option_type == "call":
            prices[i] = max(ST - K, 0)
        else:
            prices[i] = max(K - ST, 0)

    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            prices[i] = discount * (p * prices[i] + (1 - p) * prices[i + 1])
            if american:
                ST = S * (u ** (step - i)) * (d ** i)
                exercise = ST - K if option_type == "call" else K - ST
                prices[i] = max(prices[i], exercise)
    return prices[0]


def american_put_fdm_cn_psor(S0, K, r, sigma, T, Ns, Nt, Smax_ratio=3, omega=1.4, tol=1e-6, max_iter=2000):
    """Finite Difference solver (Crank-Nicolson with PSOR) for American puts."""
    if S0 <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        warnings.warn("Inputs S0, K, sigma, T should be positive.")
    if Ns <= 2 or Nt <= 1:
        raise ValueError("Ns must be > 2 and Nt must be > 1.")
    if not (1 < omega < 2):
        warnings.warn(f"omega = {omega} is outside the typical range (1, 2).")

    Smax = K * Smax_ratio
    dS = Smax / Ns
    dt = T / Nt
    S_vec = np.linspace(0, Smax, Ns + 1)
    S_int_vec = S_vec[1:-1]
    V = np.maximum(K - S_vec, 0)

    j = np.arange(1, Ns)
    sigma2 = sigma ** 2
    alpha = -0.25 * dt * (sigma2 * j ** 2 - r * j)
    beta = 1 + 0.5 * dt * (sigma2 * j ** 2 + r)
    gamma = -0.25 * dt * (sigma2 * j ** 2 + r * j)

    for n in range(Nt - 1, -1, -1):
        t = n * dt
        alpha_exp = 0.25 * dt * (sigma2 * j ** 2 - r * j)
        beta_exp = 1 - 0.5 * dt * (sigma2 * j ** 2 + r)
        gamma_exp = 0.25 * dt * (sigma2 * j ** 2 + r * j)

        rhs = alpha_exp * V[0:-2] + beta_exp * V[1:-1] + gamma_exp * V[2:]
        boundary_S0 = K * np.exp(-r * (T - t))
        rhs[0] += alpha[0] * boundary_S0
        rhs[0] += alpha_exp[0] * boundary_S0

        V_old_iter = V[1:-1].copy()
        payoff_int = np.maximum(K - S_int_vec, 0)

        for _ in range(max_iter):
            V_new_iter = V_old_iter.copy()
            max_diff = 0.0

            val = (rhs[0] - (gamma[0] * V_old_iter[1])) / beta[0]
            V_new_iter[0] = max(payoff_int[0], V_old_iter[0] + omega * (val - V_old_iter[0]))
            max_diff = max(max_diff, abs(V_new_iter[0] - V_old_iter[0]))

            for j_idx in range(1, Ns - 2):
                val = (rhs[j_idx] - (alpha[j_idx] * V_new_iter[j_idx - 1]) - (gamma[j_idx] * V_old_iter[j_idx + 1])) / beta[j_idx]
                V_new_iter[j_idx] = max(payoff_int[j_idx], V_old_iter[j_idx] + omega * (val - V_old_iter[j_idx]))
                max_diff = max(max_diff, abs(V_new_iter[j_idx] - V_old_iter[j_idx]))

            val = (rhs[Ns - 2] - (alpha[Ns - 2] * V_new_iter[Ns - 3])) / beta[Ns - 2]
            V_new_iter[Ns - 2] = max(payoff_int[Ns - 2], V_old_iter[Ns - 2] + omega * (val - V_old_iter[Ns - 2]))
            max_diff = max(max_diff, abs(V_new_iter[Ns - 2] - V_old_iter[Ns - 2]))

            V_old_iter = V_new_iter
            if max_diff < tol:
                break

        V[1:-1] = V_old_iter
        V[0] = boundary_S0
        V[Ns] = 0

    option_price = np.interp(S0, S_vec, V)
    return option_price


def american_lsmc_price(S, K, r, sigma, T, steps=50, n_paths=10000, option_type="put", seed=42):
    """Least-Squares Monte Carlo (Longstaff–Schwartz) for American options."""
    np.random.seed(seed)
    dt = T / steps
    discount = np.exp(-r * dt)

    S_paths = np.zeros((steps + 1, n_paths))
    S_paths[0, :] = S
    for t in range(1, steps + 1):
        Z = np.random.randn(n_paths)
        S_paths[t, :] = S_paths[t - 1, :] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    if option_type.lower() == "put":
        payoff = lambda x: np.maximum(K - x, 0)
    else:
        payoff = lambda x: np.maximum(x - K, 0)

    CF = np.zeros((steps + 1, n_paths))
    CF[steps, :] = payoff(S_paths[steps, :])

    for t in range(steps - 1, -1, -1):
        itm = payoff(S_paths[t, :]) > 0
        X = S_paths[t, itm]
        discounted_future = np.zeros(X.shape[0])
        for i, path_idx in enumerate(np.where(itm)[0]):
            future_ex_idx = np.where(CF[t + 1 :, path_idx] > 0)[0]
            if len(future_ex_idx) > 0:
                first_ex = future_ex_idx[0]
                discounted_future[i] = CF[t + 1 + first_ex, path_idx] * discount ** (first_ex + 1)
            else:
                discounted_future[i] = 0
        if len(X) > 0:
            A = np.vstack([np.ones(X.shape[0]), X, X ** 2]).T
            coeff, _, _, _ = np.linalg.lstsq(A, discounted_future, rcond=None)
            continuation_value = coeff[0] + coeff[1] * X + coeff[2] * X ** 2
            exercise_value = payoff(X)
            exercise_idx = exercise_value > continuation_value
            exercise_paths = np.where(itm)[0][exercise_idx]
            CF[t, exercise_paths] = exercise_value[exercise_idx]
            for ep in exercise_paths:
                CF[t + 1 :, ep] = 0

    prices = np.zeros(n_paths)
    for path_idx in range(n_paths):
        ex_times = np.where(CF[:, path_idx] > 0)[0]
        if len(ex_times) > 0:
            first_ex = ex_times[0]
            prices[path_idx] = CF[first_ex, path_idx] * np.exp(-r * (first_ex * dt))
        else:
            prices[path_idx] = 0
    return np.mean(prices)


def timed(func, *args, **kwargs):
    start = time.perf_counter()
    value = func(*args, **kwargs)
    runtime = time.perf_counter() - start
    return value, runtime


def main():
    if st is None:
        print("streamlit is required to run the dashboard")
        return

    st.title("Option Pricing Dashboard")

    st.sidebar.header("Option Parameters")
    S = st.sidebar.number_input("Spot Price", value=100.0)
    K = st.sidebar.number_input("Strike Price", value=100.0)
    T = st.sidebar.number_input("Time to Maturity (years)", value=1.0, step=0.1)
    r = st.sidebar.number_input("Risk-free Rate", value=0.05)
    sigma = st.sidebar.number_input("Volatility", value=0.2)
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    style = st.sidebar.selectbox("Style", ["European", "American"])

    steps = st.sidebar.number_input("Steps (binomial/LSMC)", value=100, step=1)
    n_paths = st.sidebar.number_input("Paths (MC/LSMC)", value=10000, step=1000)

    if st.button("Calculate"):
        results = []
        prev = st.session_state.get("prev_results")

        price, t_bs = timed(black_scholes_price, S, K, r, sigma, T, option_type)
        results.append(("Black-Scholes", price, t_bs))

        mc_price, t_mc = timed(european_monte_carlo, S, K, r, sigma, T, n_paths, option_type)
        results.append(("Monte Carlo", mc_price, t_mc))

        american = style == "American"
        bin_price, t_bin = timed(binomial_option_price, S, K, r, sigma, T, int(steps), option_type, american)
        results.append(("Binomial", bin_price, t_bin))

        if style == "American" and option_type == "put":
            fdm_price, t_fdm = timed(american_put_fdm_cn_psor, S, K, r, sigma, T, int(steps), int(steps))
            results.append(("FDM CN-PSOR", fdm_price, t_fdm))
            lsmc_price, t_lsmc = timed(american_lsmc_price, S, K, r, sigma, T, int(steps), int(n_paths), option_type)
            results.append(("LSMC", lsmc_price, t_lsmc))

        table = []
        for name, val, runtime in results:
            diff = None
            if prev and name in prev:
                diff = val - prev[name]
            table.append({"Method": name, "Price": val, "Difference": diff, "Runtime (s)": runtime})

        st.table(table)
        st.session_state.prev_results = {name: val for name, val, _ in results}


if __name__ == "__main__":
    main()
