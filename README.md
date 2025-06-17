![Dashboard screenshot](option_pricing.jpg)



# Option Pricing

This repository contains educational code for pricing European and American options.
It includes a Jupyter notebook with detailed explanations and two Streamlit dashboards
that allow interactive experimentation with various pricing models.

## Contents

- **Option_Pricing.ipynb** – step-by-step notebook covering:
  - Definitions of derivatives and options
  - Black–Scholes–Merton (BSM) formula
  - Monte Carlo simulation techniques
  - Binomial tree method
  - Finite-difference approach
  - Least-Squares Monte Carlo (LSMC)
  - Comparison of methods

  The notebook implements pricing functions in Python using `numpy`, `scipy`,
  and `matplotlib`. It demonstrates how each method converges and highlights
  when early exercise is optimal for American options.

- **dashboard.py** – lightweight Streamlit app to compute option prices using
  BSM, Monte Carlo, binomial trees, and (for American puts) finite-difference and
  LSMC methods. The app shows runtimes for each model so users can compare
  performance.

- **dashboard3.py** – advanced Streamlit dashboard styled like a trading
  terminal. It supports European and American options, displays results in
  interactive charts, and, for European options, shows the option Greeks. It
  relies on `pandas` and `plotly` for data tables and visualizations.

## Quick Start

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch one of the dashboards:

   ```bash
   streamlit run dashboard.py    # simple dashboard
   streamlit run dashboard3.py   # advanced dashboard
   ```

3. Open `Option_Pricing.ipynb` with Jupyter to read through the tutorial and run
   the examples interactively.

The dashboards expose the same underlying pricing routines as the notebook, so
you can experiment with parameters such as volatility, interest rate, or number
of simulation paths.

## Requirements

- numpy
- pandas
- matplotlib
- scipy
- networkx
- plotly
- streamlit
