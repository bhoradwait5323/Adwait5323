# Crypto Price Forecasting & Portfolio Optimization

An end-to-end pipeline that forecasts short-term cryptocurrency returns using an LSTM 
neural network, then uses those forecasts to build a risk-minimized portfolio via 
constrained optimization (SLSQP).

## Overview

This project combines two typically separate domains — time-series forecasting and 
portfolio theory — into one pipeline:

1. **Forecast**: An LSTM trained across 12 major cryptocurrencies predicts each coin's 
   expected next-day return based on 60 days of price action and technical indicators.
2. **Optimize**: Those predicted returns, combined with historical risk (covariance), 
   are fed into an SLSQP optimizer that allocates portfolio weights to minimize risk 
   for a target return.

## Data

- **Source**: Yahoo Finance (`yfinance`)
- **Assets**: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, DOT, TRX, LTC, AVAX, LINK
- **Range**: 5 years of daily OHLCV data
- **Features engineered**: SMA(20), EMA(20), RSI(14), MACD, Bollinger Bands

## Methodology

### Forecasting
- **Model**: 2-layer LSTM (64 → 32 units) with dropout, trained on 60-day sequences
- **Target**: next-day percentage return (not raw price — see Key Findings below)
- **Training**: one shared model trained across all 12 coins' combined sequences

### Portfolio Optimization
- **Method**: Sequential Least Squares Programming (SLSQP)
- **Objective**: minimize portfolio variance
- **Constraints**: weights sum to 1, portfolio return meets a target
- **Inputs**: LSTM-predicted returns + historical covariance matrix

## Key Findings

- **Predicting returns, not price, was essential.** An earlier version of the model 
  predicted scaled price levels, which produced unrealistic results for coins trading 
  far below their historical highs (e.g., DOT-USD). Switching the model's target to 
  next-day return fixed this and produced realistic, well-calibrated predictions.
- **Diversification vs. concentration tradeoff**: targeting the *average* predicted 
  return led the optimizer to diversify equally across all 12 coins, while targeting 
  the *best* predicted return concentrated 100% of the portfolio into a single coin — 
  a clean demonstration of the classic risk-return tradeoff.
- **High crypto correlation limits diversification benefit** compared to traditional 
  asset classes like stocks.

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, SciPy, pandas, yfinance, `ta`

## Limitations

- Small dataset (5 years, 12 coins) — results are illustrative, not investment advice
- No transaction costs, slippage, or rebalancing costs modeled
- Single train/test split (no walk-forward validation)
- LSTM tends to predict conservative, near-flat returns — a known tendency of MSE-trained 
  sequence models
- Static covariance matrix assumption (correlations shift in real markets)

## How to Run

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Open `notebook.ipynb` and run cells top to bottom

## Disclaimer

This project is for educational purposes only and does not constitute financial advice.
