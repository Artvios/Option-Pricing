import React, { useState, useCallback } from 'react';
import { Calculator, TrendingUp, BarChart3, Settings, Clock, DollarSign, Zap, Target } from 'lucide-react';

// Black-Scholes pricing function
const blackScholesPrice = (S, K, r, sigma, T, optionType = 'call') => {
  if (T <= 0) {
    return optionType.toLowerCase() === 'call' ? Math.max(S - K, 0) : Math.max(K - S, 0);
  }
  
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  
  const normCdf = (x) => {
    return 0.5 * (1 + erf(x / Math.sqrt(2)));
  };
  
  const erf = (x) => {
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return sign * y;
  };
  
  if (optionType.toLowerCase() === 'call') {
    return S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2);
  } else {
    return K * Math.exp(-r * T) * normCdf(-d2) - S * normCdf(-d1);
  }
};

// Monte Carlo pricing
const monteCarloPrice = (S, K, r, sigma, T, nPaths = 10000, optionType = 'call') => {
  let total = 0;
  
  for (let i = 0; i < nPaths; i++) {
    const z = Math.random() * 2 - 1 + Math.random() * 2 - 1 + Math.random() * 2 - 1;
    const ST = S * Math.exp((r - 0.5 * sigma ** 2) * T + sigma * Math.sqrt(T) * z);
    
    if (optionType.toLowerCase() === 'call') {
      total += Math.max(ST - K, 0);
    } else {
      total += Math.max(K - ST, 0);
    }
  }
  
  return Math.exp(-r * T) * (total / nPaths);
};

// Binomial pricing
const binomialPrice = (S, K, r, sigma, T, steps = 100, optionType = 'call', american = false) => {
  const dt = T / steps;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = 1 / u;
  const p = (Math.exp(r * dt) - d) / (u - d);
  const discount = Math.exp(-r * dt);
  
  let prices = new Array(steps + 1);
  
  // Initialize prices at maturity
  for (let i = 0; i <= steps; i++) {
    const ST = S * Math.pow(u, steps - i) * Math.pow(d, i);
    if (optionType === 'call') {
      prices[i] = Math.max(ST - K, 0);
    } else {
      prices[i] = Math.max(K - ST, 0);
    }
  }
  
  // Work backwards
  for (let step = steps - 1; step >= 0; step--) {
    for (let i = 0; i <= step; i++) {
      prices[i] = discount * (p * prices[i] + (1 - p) * prices[i + 1]);
      
      if (american) {
        const ST = S * Math.pow(u, step - i) * Math.pow(d, i);
        const exercise = optionType === 'call' ? ST - K : K - ST;
        prices[i] = Math.max(prices[i], exercise);
      }
    }
  }
  
  return prices[0];
};

// Greeks calculations
const calculateGreeks = (S, K, r, sigma, T, optionType) => {
  const h = 0.01;
  const price = blackScholesPrice(S, K, r, sigma, T, optionType);
  
  // Delta
  const priceUp = blackScholesPrice(S + h, K, r, sigma, T, optionType);
  const priceDown = blackScholesPrice(S - h, K, r, sigma, T, optionType);
  const delta = (priceUp - priceDown) / (2 * h);
  
  // Gamma
  const gamma = (priceUp - 2 * price + priceDown) / (h * h);
  
  // Theta
  const priceT = blackScholesPrice(S, K, r, sigma, T - h/365, optionType);
  const theta = (priceT - price) / (h/365);
  
  // Vega
  const priceV = blackScholesPrice(S, K, r, sigma + h, T, optionType);
  const vega = (priceV - price) / h;
  
  // Rho
  const priceR = blackScholesPrice(S, K, r + h, sigma, T, optionType);
  const rho = (priceR - price) / h;
  
  return { delta, gamma, theta, vega, rho };
};

const OptionPricingDashboard = () => {
  const [inputs, setInputs] = useState({
    spotPrice: 100,
    strikePrice: 100,
    timeToMaturity: 1.0,
    riskFreeRate: 0.05,
    volatility: 0.20,
    optionType: 'call',
    style: 'European',
    steps: 100,
    paths: 10000
  });
  
  const [results, setResults] = useState(null);
  const [previousResults, setPreviousResults] = useState(null);
  const [calculating, setCalculating] = useState(false);

  const calculate = useCallback(() => {
    setCalculating(true);
    
    // Small delay to show loading state
    setTimeout(() => {
      const { spotPrice: S, strikePrice: K, riskFreeRate: r, volatility: sigma, timeToMaturity: T, optionType, style, steps, paths } = inputs;
      
      const startTime = performance.now();
      
      // Black-Scholes
      const bsStart = performance.now();
      const bsPrice = blackScholesPrice(S, K, r, sigma, T, optionType);
      const bsTime = performance.now() - bsStart;
      
      // Monte Carlo
      const mcStart = performance.now();
      const mcPrice = monteCarloPrice(S, K, r, sigma, T, paths, optionType);
      const mcTime = performance.now() - mcStart;
      
      // Binomial
      const binStart = performance.now();
      const american = style === 'American';
      const binPrice = binomialPrice(S, K, r, sigma, T, steps, optionType, american);
      const binTime = performance.now() - binStart;
      
      // Greeks
      const greeks = calculateGreeks(S, K, r, sigma, T, optionType);
      
      const newResults = {
        'Black-Scholes': { price: bsPrice, time: bsTime },
        'Monte Carlo': { price: mcPrice, time: mcTime },
        'Binomial': { price: binPrice, time: binTime },
        greeks
      };
      
      setPreviousResults(results);
      setResults(newResults);
      setCalculating(false);
    }, 100);
  }, [inputs, results]);

  const formatPrice = (price) => price?.toFixed(4) || '0.0000';
  const formatTime = (time) => `${(time || 0).toFixed(2)}ms`;
  const formatChange = (current, previous) => {
    if (!previous) return null;
    const change = current - previous;
    const changeClass = change > 0 ? 'text-green-500' : change < 0 ? 'text-red-500' : 'text-gray-500';
    return (
      <span className={`text-sm font-medium ${changeClass}`}>
        {change > 0 ? '+' : ''}{change.toFixed(4)}
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg p-6 mb-6 shadow-2xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Calculator className="w-8 h-8 text-blue-400" />
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Options Pricing Terminal
              </h1>
              <p className="text-gray-300 mt-1">Professional derivatives valuation platform</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400">Real-time Analytics</div>
            <div className="text-lg font-mono text-green-400">{new Date().toLocaleTimeString()}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Panel */}
        <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
          <div className="flex items-center space-x-2 mb-6">
            <Settings className="w-5 h-5 text-blue-400" />
            <h2 className="text-xl font-semibold">Parameters</h2>
          </div>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Spot Price (S)</label>
                <input
                  type="number"
                  value={inputs.spotPrice}
                  onChange={(e) => setInputs({...inputs, spotPrice: parseFloat(e.target.value) || 0})}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Strike Price (K)</label>
                <input
                  type="number"
                  value={inputs.strikePrice}
                  onChange={(e) => setInputs({...inputs, strikePrice: parseFloat(e.target.value) || 0})}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Time to Maturity (Years)</label>
              <input
                type="number"
                step="0.1"
                value={inputs.timeToMaturity}
                onChange={(e) => setInputs({...inputs, timeToMaturity: parseFloat(e.target.value) || 0})}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Risk-free Rate (r)</label>
                <input
                  type="number"
                  step="0.01"
                  value={inputs.riskFreeRate}
                  onChange={(e) => setInputs({...inputs, riskFreeRate: parseFloat(e.target.value) || 0})}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Volatility (σ)</label>
                <input
                  type="number"
                  step="0.01"
                  value={inputs.volatility}
                  onChange={(e) => setInputs({...inputs, volatility: parseFloat(e.target.value) || 0})}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Option Type</label>
                <select
                  value={inputs.optionType}
                  onChange={(e) => setInputs({...inputs, optionType: e.target.value})}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                >
                  <option value="call">Call</option>
                  <option value="put">Put</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Style</label>
                <select
                  value={inputs.style}
                  onChange={(e) => setInputs({...inputs, style: e.target.value})}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                >
                  <option value="European">European</option>
                  <option value="American">American</option>
                </select>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Steps</label>
                <input
                  type="number"
                  value={inputs.steps}
                  onChange={(e) => setInputs({...inputs, steps: parseInt(e.target.value) || 0})}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">MC Paths</label>
                <input
                  type="number"
                  value={inputs.paths}
                  onChange={(e) => setInputs({...inputs, paths: parseInt(e.target.value) || 0})}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
            </div>
            
            <button
              onClick={calculate}
              disabled={calculating}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
            >
              {calculating ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Calculating...</span>
                </>
              ) : (
                <>
                  <Calculator className="w-4 h-4" />
                  <span>Calculate Options</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Pricing Results */}
          <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
            <div className="flex items-center space-x-2 mb-6">
              <DollarSign className="w-5 h-5 text-green-400" />
              <h2 className="text-xl font-semibold">Valuation Results</h2>
            </div>
            
            {results ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(results).filter(([key]) => key !== 'greeks').map(([method, data]) => (
                  <div key={method} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-blue-400">{method}</h3>
                      <Zap className="w-4 h-4 text-yellow-400" />
                    </div>
                    <div className="text-2xl font-mono font-bold text-white mb-1">
                      ${formatPrice(data.price)}
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">{formatTime(data.time)}</span>
                      {previousResults && previousResults[method] && 
                        formatChange(data.price, previousResults[method].price)
                      }
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-gray-400">
                <Calculator className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Click "Calculate Options" to see pricing results</p>
              </div>
            )}
          </div>

          {/* Greeks */}
          {results?.greeks && (
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <div className="flex items-center space-x-2 mb-6">
                <TrendingUp className="w-5 h-5 text-purple-400" />
                <h2 className="text-xl font-semibold">Risk Sensitivities (Greeks)</h2>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="bg-gray-700 rounded-lg p-4 text-center">
                  <div className="text-sm text-gray-400 mb-1">Delta (Δ)</div>
                  <div className="text-lg font-mono font-bold text-blue-400">
                    {results.greeks.delta.toFixed(4)}
                  </div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4 text-center">
                  <div className="text-sm text-gray-400 mb-1">Gamma (Γ)</div>
                  <div className="text-lg font-mono font-bold text-green-400">
                    {results.greeks.gamma.toFixed(4)}
                  </div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4 text-center">
                  <div className="text-sm text-gray-400 mb-1">Theta (Θ)</div>
                  <div className="text-lg font-mono font-bold text-red-400">
                    {results.greeks.theta.toFixed(4)}
                  </div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4 text-center">
                  <div className="text-sm text-gray-400 mb-1">Vega (ν)</div>
                  <div className="text-lg font-mono font-bold text-purple-400">
                    {results.greeks.vega.toFixed(4)}
                  </div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4 text-center">
                  <div className="text-sm text-gray-400 mb-1">Rho (ρ)</div>
                  <div className="text-lg font-mono font-bold text-yellow-400">
                    {results.greeks.rho.toFixed(4)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Performance Metrics */}
          {results && (
            <div className="bg-gray-800 rounded-lg p-6 shadow-xl">
              <div className="flex items-center space-x-2 mb-6">
                <BarChart3 className="w-5 h-5 text-orange-400" />
                <h2 className="text-xl font-semibold">Performance Analysis</h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3 text-orange-400">Method Comparison</h3>
                  <div className="space-y-2">
                    {Object.entries(results).filter(([key]) => key !== 'greeks').map(([method, data]) => {
                      const bsPrice = results['Black-Scholes'].price;
                      const diff = Math.abs(data.price - bsPrice);
                      const diffPercent = (diff / bsPrice * 100);
                      
                      return (
                        <div key={method} className="flex justify-between items-center py-2 px-3 bg-gray-700 rounded">
                          <span className="text-sm font-medium">{method}</span>
                          <span className="text-sm text-gray-400">
                            ±{diffPercent.toFixed(2)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold mb-3 text-orange-400">Execution Time</h3>
                  <div className="space-y-2">
                    {Object.entries(results).filter(([key]) => key !== 'greeks').map(([method, data]) => (
                      <div key={method} className="flex justify-between items-center py-2 px-3 bg-gray-700 rounded">
                        <span className="text-sm font-medium">{method}</span>
                        <span className="text-sm text-gray-400">
                          {formatTime(data.time)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default OptionPricingDashboard;
