import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Options Pricing Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bloomberg-style CSS
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-title {
        color: #60a5fa;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    
    .metric-delta {
        font-size: 12px;
        font-weight: 500;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 2px solid #3b82f6;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    }
    
    .terminal-title {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 5px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .terminal-subtitle {
        color: #d1d5db;
        font-size: 16px;
        margin-bottom: 10px;
    }
    
    /* Greeks container */
    .greeks-container {
        background: #111827;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        margin: 10px 0;
    }
    
    .greek-item {
        background: #1f2937;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #4b5563;
    }
    
    .greek-name {
        color: #9ca3af;
        font-size: 12px;
        margin-bottom: 5px;
    }
    
    .greek-value {
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        background-color: #1f2937;
        color: #ffffff;
        border: 1px solid #4b5563;
    }
    
    .stSelectbox > div > div > select {
        background-color: #1f2937;
        color: #ffffff;
        border: 1px solid #4b5563;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Performance table */
    .performance-table {
        background: #111827;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #374151;
    }
    
    /* Live clock */
    .live-clock {
        color: #10b981;
        font-family: 'Courier New', monospace;
        font-size: 18px;
        font-weight: 700;
    }
    
    /* Status indicators */
    .status-connected {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-calculating {
        color: #f59e0b;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def american_put_fdm_cn_psor(S0, K, r, sigma, T, Ns, Nt, Smax_ratio=3, omega=1.4, tol=1e-6, max_iter=2000):
    """
    Prices an American put option using Finite Differences (Crank-Nicolson)
    with Projected Successive Over-Relaxation (PSOR).
    """
    if S0 <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        warnings.warn("Inputs S0, K, sigma, T should be positive.")
    if Ns <= 2 or Nt <= 1:
        raise ValueError("Ns must be > 2 and Nt must be > 1.")
    if not (1 < omega < 2):
        warnings.warn(f"omega = {omega} is outside the typical range (1, 2).")

    # Grid Setup
    Smax = K * Smax_ratio
    dS = Smax / Ns
    dt = T / Nt

    # Stock price grid (Ns+1 points from 0 to Smax)
    S_vec = np.linspace(0, Smax, Ns + 1)
    # Interior stock price grid points (Ns-1 points)
    S_int_vec = S_vec[1:-1]

    # Initialize option value grid (vector)
    V = np.maximum(K - S_vec, 0) # Payoff at maturity T

    # Set up Crank-Nicolson coefficients (constant for this PDE)
    j = np.arange(1, Ns) # Interior indices
    sigma2 = sigma**2
    aj = 0.5 * dt * (r * j - 0.5 * sigma2 * j**2)
    bj = 1 + dt * (r + sigma2 * j**2)
    cj = -0.5 * dt * (r * j + 0.5 * sigma2 * j**2)

    # Coefficients for the iterative solver matrix (implicit part)
    alpha = -0.25 * dt * (sigma2 * j**2 - r * j)
    beta = 1 + 0.5 * dt * (sigma2 * j**2 + r)
    gamma = -0.25 * dt * (sigma2 * j**2 + r * j)

    # Backward Time Iteration
    for n in range(Nt - 1, -1, -1): # Time steps from T-dt down to 0
        t = n * dt

        # Calculate the known RHS vector from V at step n+1
        alpha_exp = 0.25 * dt * (sigma2 * j**2 - r * j)
        beta_exp = 1 - 0.5 * dt * (sigma2 * j**2 + r)
        gamma_exp = 0.25 * dt * (sigma2 * j**2 + r * j)

        rhs = alpha_exp * V[0:-2] + beta_exp * V[1:-1] + gamma_exp * V[2:]

        # Incorporate boundary conditions into RHS
        boundary_S0 = K * np.exp(-r * (T - t))
        rhs[0] += alpha[0] * boundary_S0
        rhs[0] += alpha_exp[0] * boundary_S0

        # PSOR Iteration
        V_old_iter = V[1:-1].copy()
        payoff_int = np.maximum(K - S_int_vec, 0)

        for k in range(max_iter):
            V_new_iter = V_old_iter.copy()
            max_diff = 0.0

            # Update V[1] (j=1) using boundary V[0]
            val = (rhs[0] - (gamma[0] * V_old_iter[1])) / beta[0]
            V_new_iter[0] = max(payoff_int[0], V_old_iter[0] + omega * (val - V_old_iter[0]))
            max_diff = max(max_diff, abs(V_new_iter[0] - V_old_iter[0]))

            # Update interior points V[2] to V[Ns-2] (j=2 to Ns-2)
            for j_idx in range(1, Ns - 2):
                 val = (rhs[j_idx] - (alpha[j_idx] * V_new_iter[j_idx-1]) - (gamma[j_idx] * V_old_iter[j_idx+1])) / beta[j_idx]
                 V_new_iter[j_idx] = max(payoff_int[j_idx], V_old_iter[j_idx] + omega * (val - V_old_iter[j_idx]))
                 max_diff = max(max_diff, abs(V_new_iter[j_idx] - V_old_iter[j_idx]))

            # Update V[Ns-1] (j=Ns-1) using boundary V[Ns] = 0
            val = (rhs[Ns-2] - (alpha[Ns-2] * V_new_iter[Ns-3])) / beta[Ns-2]
            V_new_iter[Ns-2] = max(payoff_int[Ns-2], V_old_iter[Ns-2] + omega * (val - V_old_iter[Ns-2]))
            max_diff = max(max_diff, abs(V_new_iter[Ns-2] - V_old_iter[Ns-2]))

            V_old_iter = V_new_iter

            # Check convergence
            if max_diff < tol:
                break
        else:
             warnings.warn(f"PSOR did not converge at time step n={n} (t={t:.3f}) after {max_iter} iterations.")

        # Update V for the current time step n
        V[1:-1] = V_old_iter
        V[0] = boundary_S0
        V[Ns] = 0

    # Interpolate the value at S0 from the final grid V (at t=0)
    option_price = np.interp(S0, S_vec, V)
    return option_price

def american_lsmc_price(S, K, r, sigma, T, steps=50, n_paths=10000, option_type='put', seed=42):
    """
    Least-Squares Monte Carlo (Longstaff–Schwartz) for American put or call.
    """
    np.random.seed(seed)
    dt = T / steps
    discount = np.exp(-r*dt)
    
    # Simulate underlying paths
    S_paths = np.zeros((steps+1, n_paths))
    S_paths[0,:] = S
    for t in range(1, steps+1):
        Z = np.random.randn(n_paths)
        S_paths[t,:] = S_paths[t-1,:] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    
    # Payoff function
    if option_type.lower() == 'put':
        payoff = lambda x: np.maximum(K - x, 0)
    else:
        payoff = lambda x: np.maximum(x - K, 0)
    
    # Cashflows array to store the realized payoff (if exercised)
    CF = np.zeros((steps+1, n_paths))
    # At maturity, exercise if in the money
    CF[steps,:] = payoff(S_paths[steps,:])
    
    # Work backward
    for t in range(steps-1, -1, -1):
        # In-the-money paths
        itm = payoff(S_paths[t,:]) > 0
        if np.sum(itm) == 0:
            continue
            
        X = S_paths[t, itm]  # underlying in the money
        
        # Calculate continuation value
        discounted_future = np.zeros(X.shape[0])
        for i, path_idx in enumerate(np.where(itm)[0]):
            future_ex_idx = np.where(CF[t+1:, path_idx] > 0)[0]
            if len(future_ex_idx) > 0:
                first_ex = future_ex_idx[0]
                discounted_future[i] = CF[t+1+first_ex, path_idx] * discount**(first_ex+1)
            else:
                discounted_future[i] = 0
        
        # Regress discounted_future on polynomial basis of X (S[t])
        if len(X) > 3:  # Need enough points for regression
            A = np.vstack([np.ones(X.shape[0]), X, X**2]).T
            try:
                coeff, _, _, _ = np.linalg.lstsq(A, discounted_future, rcond=None)
                continuation_value = coeff[0] + coeff[1]*X + coeff[2]*X**2
            except:
                continuation_value = discounted_future  # Fallback
        else:
            continuation_value = discounted_future
        
        # Exercise decision
        exercise_value = payoff(X)
        exercise_idx = (exercise_value > continuation_value)
        
        # For those who exercise, set CF at time t
        exercise_paths = np.where(itm)[0][exercise_idx]
        CF[t, exercise_paths] = exercise_value[exercise_idx]
        # For times t+1..end, set CF=0 if exercised
        for ep in exercise_paths:
            CF[t+1:, ep] = 0
    
    # The price is the average discounted payoff from t=0
    prices = np.zeros(n_paths)
    for path_idx in range(n_paths):
        ex_times = np.where(CF[:, path_idx]>0)[0]
        if len(ex_times) > 0:
            first_ex = ex_times[0]
            prices[path_idx] = CF[first_ex, path_idx]*np.exp(-r*(first_ex*dt))
        else:
            prices[path_idx] = 0
    
    return np.mean(prices)

class OptionsPricingEngine:
    """Professional options pricing engine with multiple models"""
    
    @staticmethod
    def black_scholes_price(S, K, T, r, sigma, option_type='call'):
        """Black-Scholes-Merton pricing model"""
        if T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def monte_carlo_price(S, K, T, r, sigma, n_paths=100000, option_type='call'):
        """Monte Carlo simulation pricing"""
        dt = T
        z = np.random.standard_normal(n_paths)
        ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        return price
    
    @staticmethod
    def binomial_price(S, K, T, r, sigma, steps=1000, option_type='call', american=False):
        """Binomial tree pricing model"""
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        
        # Initialize asset prices at maturity
        asset_prices = np.zeros(steps + 1)
        for i in range(steps + 1):
            asset_prices[i] = S * (u ** (steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.zeros(steps + 1)
        for i in range(steps + 1):
            if option_type == 'call':
                option_values[i] = max(asset_prices[i] - K, 0)
            else:
                option_values[i] = max(K - asset_prices[i], 0)
        
        # Work backwards through the tree
        for step in range(steps - 1, -1, -1):
            for i in range(step + 1):
                # Calculate continuation value
                continuation_value = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                if american:
                    # Calculate exercise value
                    asset_price = S * (u ** (step - i)) * (d ** i)
                    if option_type == 'call':
                        exercise_value = max(asset_price - K, 0)
                    else:
                        exercise_value = max(K - asset_price, 0)
                    
                    option_values[i] = max(continuation_value, exercise_value)
                else:
                    option_values[i] = continuation_value
        
        return option_values[0]
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call', is_american=False):
        """Calculate option Greeks"""
        h = 0.01
        
        # Get base price - use appropriate method based on option style
        if is_american:
            if option_type == 'put':
                price = american_put_fdm_cn_psor(S, K, r, sigma, T, 100, 100)
            else:
                price = OptionsPricingEngine.binomial_price(S, K, T, r, sigma, 1000, option_type, True)
        else:
            price = OptionsPricingEngine.black_scholes_price(S, K, T, r, sigma, option_type)
        
        # Delta
        if is_american:
            if option_type == 'put':
                price_up = american_put_fdm_cn_psor(S + h, K, r, sigma, T, 100, 100)
                price_down = american_put_fdm_cn_psor(S - h, K, r, sigma, T, 100, 100)
            else:
                price_up = OptionsPricingEngine.binomial_price(S + h, K, T, r, sigma, 1000, option_type, True)
                price_down = OptionsPricingEngine.binomial_price(S - h, K, T, r, sigma, 1000, option_type, True)
        else:
            price_up = OptionsPricingEngine.black_scholes_price(S + h, K, T, r, sigma, option_type)
            price_down = OptionsPricingEngine.black_scholes_price(S - h, K, T, r, sigma, option_type)
        delta = (price_up - price_down) / (2 * h)
        
        # Gamma
        gamma = (price_up - 2 * price + price_down) / (h**2)
        
        # Theta (per day)
        if T > h/365:
            if is_american:
                if option_type == 'put':
                    price_theta = american_put_fdm_cn_psor(S, K, r, sigma, T - h/365, 100, 100)
                else:
                    price_theta = OptionsPricingEngine.binomial_price(S, K, T - h/365, r, sigma, 1000, option_type, True)
            else:
                price_theta = OptionsPricingEngine.black_scholes_price(S, K, T - h/365, r, sigma, option_type)
            theta = (price_theta - price) / (h/365) / 365
        else:
            theta = 0
        
        # Vega (per 1% change in volatility)
        if is_american:
            if option_type == 'put':
                price_vega = american_put_fdm_cn_psor(S, K, r, sigma + h, T, 100, 100)
            else:
                price_vega = OptionsPricingEngine.binomial_price(S, K, T, r, sigma + h, 1000, option_type, True)
        else:
            price_vega = OptionsPricingEngine.black_scholes_price(S, K, T, r, sigma + h, option_type)
        vega = (price_vega - price) / h * 0.01
        
        # Rho (per 1% change in interest rate)
        if is_american:
            if option_type == 'put':
                price_rho = american_put_fdm_cn_psor(S, K, r + h, sigma, T, 100, 100)
            else:
                price_rho = OptionsPricingEngine.binomial_price(S, K, T, r + h, sigma, 1000, option_type, True)
        else:
            price_rho = OptionsPricingEngine.black_scholes_price(S, K, T, r + h, sigma, option_type)
        rho = (price_rho - price) / h * 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

def create_pricing_comparison_chart(results_df):
    """Create a professional pricing comparison chart"""
    fig = go.Figure()
    
    methods = results_df['Method'].tolist()
    prices = results_df['Price'].tolist()
    changes = results_df.get('Change %', [0] * len(methods))
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    fig.add_trace(go.Bar(
        x=methods,
        y=prices,
        marker_color=colors,
        text=[f'${p:.4f}<br>{c:+.2f}%' if c != 0 else f'${p:.4f}' for p, c in zip(prices, changes)],
        textposition='auto',
        name='Option Price'
    ))
    
    fig.update_layout(
        title='Pricing Model Comparison',
        title_font_size=20,
        title_font_color='white',
        xaxis_title='Pricing Method',
        yaxis_title='Option Price ($)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        xaxis=dict(color='white', gridcolor='#374151'),
        yaxis=dict(color='white', gridcolor='#374151'),
        height=400
    )
    
    return fig

def create_greeks_radar_chart(greeks):
    """Create a radar chart for Greeks visualization"""
    greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    greek_values = [greeks['delta'], greeks['gamma'], abs(greeks['theta']), 
                   greeks['vega'], greeks['rho']]
    
    # Normalize values for radar chart
    max_val = max(abs(v) for v in greek_values if v != 0)
    if max_val > 0:
        normalized_values = [abs(v)/max_val for v in greek_values]
    else:
        normalized_values = [0] * len(greek_values)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=greek_names,
        fill='toself',
        name='Greeks',
        line_color='#3b82f6',
        fillcolor='rgba(59, 130, 246, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                color='white'
            ),
            angularaxis=dict(color='white')
        ),
        showlegend=False,
        title='Risk Sensitivities (Greeks)',
        title_font_size=18,
        title_font_color='white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    return fig

def main():
    # Initialize session state for tracking previous values
    if 'previous_params' not in st.session_state:
        st.session_state['previous_params'] = None
    if 'previous_results' not in st.session_state:
        st.session_state['previous_results'] = None
    
    # Header
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="terminal-title">🔷 OPTIONS PRICING TERMINAL</h1>
                <p class="terminal-subtitle">Professional Derivatives Valuation Platform</p>
                <div style="display: flex; gap: 20px; margin-top: 10px;">
                    <span class="status-connected">● LIVE MARKET DATA</span>
                    <span style="color: #60a5fa;">📊 REAL-TIME ANALYTICS</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="color: #9ca3af; font-size: 14px;">Market Status</div>
                <div class="live-clock">{}</div>
                <div style="color: #10b981; font-size: 12px; margin-top: 5px;">CONNECTED</div>
            </div>
        </div>
    </div>
    """.format(datetime.now().strftime("%H:%M:%S EST")), unsafe_allow_html=True)
    
    # Sidebar - Trading Parameters
    st.sidebar.markdown("### 📋 TRADING PARAMETERS")
    st.sidebar.markdown("---")
    
    # Market Data Inputs
    col1, col2 = st.sidebar.columns(2)
    with col1:
        spot_price = st.number_input("Spot Price (S)", value=100.0, min_value=0.1, step=0.1, format="%.2f")
    with col2:
        strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, step=0.1, format="%.2f")
    
    time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, min_value=0.001, step=0.1, format="%.3f")
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        risk_free_rate = st.number_input("Risk-free Rate (%)", value=5.0, min_value=0.0, step=0.1, format="%.2f") / 100
    with col4:
        volatility = st.number_input("Volatility (%)", value=20.0, min_value=0.1, step=0.1, format="%.2f") / 100
    
    # Option Specifications
    st.sidebar.markdown("### ⚙️ OPTION SPECIFICATIONS")
    col5, col6 = st.sidebar.columns(2)
    with col5:
        option_type = st.selectbox("Option Type", ["call", "put"])
    with col6:
        style = st.selectbox("Exercise Style", ["European", "American"])
    
    # Model Parameters
    st.sidebar.markdown("### 🔧 MODEL PARAMETERS")
    if style == "American":
        col7, col8 = st.sidebar.columns(2)
        with col7:
            lsmc_paths = st.number_input("LSMC Paths", value=10000, min_value=1000, step=1000)
        with col8:
            fdm_steps = st.number_input("FDM Steps", value=100, min_value=10, step=10)
    else:
        col7, col8 = st.sidebar.columns(2)
        with col7:
            mc_paths = st.number_input("MC Simulations", value=100000, min_value=1000, step=1000)
        with col8:
            binomial_steps = st.number_input("Binomial Steps", value=1000, min_value=10, step=10)
    
    # Calculate button
    calculate_button = st.sidebar.button("🚀 EXECUTE PRICING", use_container_width=True)
    
    # Current parameters
    current_params = {
        'spot_price': spot_price,
        'strike_price': strike_price
