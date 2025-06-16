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
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        h = 0.01
        
        # Get base price
        price = OptionsPricingEngine.black_scholes_price(S, K, T, r, sigma, option_type)
        
        # Delta
        price_up = OptionsPricingEngine.black_scholes_price(S + h, K, T, r, sigma, option_type)
        price_down = OptionsPricingEngine.black_scholes_price(S - h, K, T, r, sigma, option_type)
        delta = (price_up - price_down) / (2 * h)
        
        # Gamma
        gamma = (price_up - 2 * price + price_down) / (h**2)
        
        # Theta (per day)
        if T > h/365:
            price_theta = OptionsPricingEngine.black_scholes_price(S, K, T - h/365, r, sigma, option_type)
            theta = (price_theta - price) / (h/365) / 365
        else:
            theta = 0
        
        # Vega (per 1% change in volatility)
        price_vega = OptionsPricingEngine.black_scholes_price(S, K, T, r, sigma + h, option_type)
        vega = (price_vega - price) / h * 0.01
        
        # Rho (per 1% change in interest rate)
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
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    fig.add_trace(go.Bar(
        x=methods,
        y=prices,
        marker_color=colors,
        text=[f'${p:.4f}' for p in prices],
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
    max_val = max(abs(v) for v in greek_values)
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

def create_volatility_surface():
    """Create a 3D volatility surface visualization"""
    strikes = np.linspace(80, 120, 20)
    times = np.linspace(0.1, 2, 20)
    Strike, Time = np.meshgrid(strikes, times)
    
    # Simulate volatility surface (simplified model)
    base_vol = 0.2
    skew = -0.1
    term_structure = 0.05
    
    Volatility = base_vol + skew * (Strike - 100) / 100 + term_structure * np.sqrt(Time)
    
    fig = go.Figure(data=[go.Surface(
        z=Volatility,
        x=Strike,
        y=Time,
        colorscale='Viridis',
        opacity=0.8
    )])
    
    fig.update_layout(
        title='Implied Volatility Surface',
        title_font_size=18,
        title_font_color='white',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Maturity',
            zaxis_title='Implied Volatility',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='white'),
            yaxis=dict(color='white'),
            zaxis=dict(color='white')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500
    )
    
    return fig

def main():
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
    col7, col8 = st.sidebar.columns(2)
    with col7:
        mc_paths = st.number_input("MC Simulations", value=100000, min_value=1000, step=1000)
    with col8:
        binomial_steps = st.number_input("Binomial Steps", value=1000, min_value=10, step=10)
    
    # Calculate button
    calculate_button = st.sidebar.button("🚀 EXECUTE PRICING", use_container_width=True)
    
    if calculate_button or 'results' not in st.session_state:
        with st.spinner("⚡ Executing pricing models..."):
            # Initialize pricing engine
            engine = OptionsPricingEngine()
            
            # Calculate using different methods
            start_time = time.time()
            
            # Black-Scholes
            bs_start = time.time()
            bs_price = engine.black_scholes_price(spot_price, strike_price, time_to_maturity, 
                                                risk_free_rate, volatility, option_type)
            bs_time = (time.time() - bs_start) * 1000
            
            # Monte Carlo
            mc_start = time.time()
            mc_price = engine.monte_carlo_price(spot_price, strike_price, time_to_maturity,
                                              risk_free_rate, volatility, mc_paths, option_type)
            mc_time = (time.time() - mc_start) * 1000
            
            # Binomial
            bin_start = time.time()
            american = (style == "American")
            bin_price = engine.binomial_price(spot_price, strike_price, time_to_maturity,
                                            risk_free_rate, volatility, binomial_steps, 
                                            option_type, american)
            bin_time = (time.time() - bin_start) * 1000
            
            # Greeks
            greeks = engine.calculate_greeks(spot_price, strike_price, time_to_maturity,
                                           risk_free_rate, volatility, option_type)
            
            total_time = (time.time() - start_time) * 1000
            
            # Store results
            st.session_state['results'] = {
                'prices': {
                    'Black-Scholes': {'price': bs_price, 'time': bs_time},
                    'Monte Carlo': {'price': mc_price, 'time': mc_time},
                    'Binomial': {'price': bin_price, 'time': bin_time}
                },
                'greeks': greeks,
                'total_time': total_time
            }
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Main pricing results
        st.markdown("### 💰 VALUATION RESULTS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bs_price = results['prices']['Black-Scholes']['price']
            bs_time = results['prices']['Black-Scholes']['time']
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">BLACK-SCHOLES</div>
                <div class="metric-value">${bs_price:.4f}</div>
                <div class="metric-delta" style="color: #10b981;">⚡ {bs_time:.2f}ms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mc_price = results['prices']['Monte Carlo']['price']
            mc_time = results['prices']['Monte Carlo']['time']
            mc_diff = ((mc_price - bs_price) / bs_price * 100)
            diff_color = "#10b981" if mc_diff >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">MONTE CARLO</div>
                <div class="metric-value">${mc_price:.4f}</div>
                <div class="metric-delta" style="color: {diff_color};">
                    {mc_diff:+.2f}% | ⚡ {mc_time:.2f}ms
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            bin_price = results['prices']['Binomial']['price']
            bin_time = results['prices']['Binomial']['time']
            bin_diff = ((bin_price - bs_price) / bs_price * 100)
            diff_color = "#10b981" if bin_diff >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">BINOMIAL</div>
                <div class="metric-value">${bin_price:.4f}</div>
                <div class="metric-delta" style="color: {diff_color};">
                    {bin_diff:+.2f}% | ⚡ {bin_time:.2f}ms
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Greeks display
        st.markdown("### 📊 RISK SENSITIVITIES (GREEKS)")
        
        greeks = results['greeks']
        col1, col2, col3, col4, col5 = st.columns(5)
        
        greek_configs = [
            ('DELTA (Δ)', greeks['delta'], '#3b82f6', 'Price sensitivity'),
            ('GAMMA (Γ)', greeks['gamma'], '#10b981', 'Delta sensitivity'),
            ('THETA (Θ)', greeks['theta'], '#ef4444', 'Time decay'),
            ('VEGA (ν)', greeks['vega'], '#8b5cf6', 'Volatility sensitivity'),
            ('RHO (ρ)', greeks['rho'], '#f59e0b', 'Rate sensitivity')
        ]
        
        for i, (col, (name, value, color, desc)) in enumerate(zip([col1, col2, col3, col4, col5], greek_configs)):
            with col:
                st.markdown(f"""
                <div class="greek-item">
                    <div class="greek-name">{name}</div>
                    <div class="greek-value" style="color: {color};">{value:.4f}</div>
                    <div style="color: #9ca3af; font-size: 10px; margin-top: 5px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Charts section
        st.markdown("### 📈 ANALYTICS & VISUALIZATION")
        
        tab1, tab2, tab3 = st.tabs(["📊 Pricing Comparison", "🎯 Greeks Analysis", "📉 Volatility Surface"])
        
        with tab1:
            # Create pricing comparison chart
            results_df = pd.DataFrame([
                {'Method': 'Black-Scholes', 'Price': bs_price, 'Time (ms)': bs_time},
                {'Method': 'Monte Carlo', 'Price': mc_price, 'Time (ms)': mc_time},
                {'Method': 'Binomial', 'Price': bin_price, 'Time (ms)': bin_time}
            ])
            
            fig = create_pricing_comparison_chart(results_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics table
            st.markdown("#### ⚡ PERFORMANCE METRICS")
            
            performance_df = pd.DataFrame({
                'Method': ['Black-Scholes', 'Monte Carlo', 'Binomial'],
                'Price ($)': [f"{p:.4f}" for p in [bs_price, mc_price, bin_price]],
                'Execution Time (ms)': [f"{t:.2f}" for t in [bs_time, mc_time, bin_time]],
                'Accuracy vs BS (%)': ['Reference', f"{mc_diff:+.2f}%", f"{bin_diff:+.2f}%"],
                'Method Type': ['Analytical', 'Simulation', 'Numerical']
            })
            
            st.dataframe(performance_df, use_container_width=True)
        
        with tab2:
            # Greeks radar chart
            fig_radar = create_greeks_radar_chart(greeks)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Greeks interpretation
            st.markdown("#### 🔍 GREEKS INTERPRETATION")
            
            interpretations = {
                'Delta': f"A 1% move in underlying price changes option value by ${abs(greeks['delta']*spot_price*0.01):.4f}",
                'Gamma': f"Delta changes by {greeks['gamma']:.4f} for each $1 move in underlying",
                'Theta': f"Option loses ${abs(greeks['theta']):.4f} in value per day (time decay)",
                'Vega': f"1% increase in volatility increases option value by ${greeks['vega']:.4f}",
                'Rho': f"1% increase in interest rate changes option value by ${greeks['rho']:.4f}"
            }
            
            for greek, interpretation in interpretations.items():
                st.markdown(f"**{greek}:** {interpretation}")
        
        with tab3:
            # Volatility surface
            fig_surface = create_volatility_surface()
            st.plotly_chart(fig_surface, use_container_width=True)
            
            st.markdown("""
            #### 📊 VOLATILITY SURFACE ANALYSIS
            This 3D surface shows how implied volatility varies across different strikes and maturities.
            - **X-axis**: Strike prices relative to spot
            - **Y-axis**: Time to maturity
            - **Z-axis**: Implied volatility levels
            """)
        
        # Footer with execution summary
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #9ca3af; font-size: 12px;">
            <strong>EXECUTION SUMMARY</strong> | Total computation time: {results['total_time']:.2f}ms | 
            Models executed: 3 | Greeks calculated: 5 | 
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S EST")}
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
