# DirectionalLSTM — Confidence-Based Stock Direction Prediction

## Overview
DirectionalLSTM is a PyTorch-based deep learning system that predicts the next-day direction (up/down) of stock prices using sequential time-series data.

Instead of predicting exact prices, the model outputs probabilities of upward movement, enabling confidence-based trading decisions (BUY / SELL / HOLD). The system is evaluated using walk-forward validation, making it more realistic for non-stationary financial markets.

## Key Highlights
- Directional prediction (classification instead of regression)
- Probabilistic outputs for confidence-aware decisions
- Walk-forward (rolling window) validation
- Strict prevention of data leakage (scaling fit on train only)
- Confidence threshold filtering for trade selection
- Integration of market context (SPY returns and volatility)

## Model Architecture
- 2-layer LSTM (hidden size = 96)
- Dropout = 0.3
- Fully connected output layer
- Loss: Binary Cross Entropy with Logits
- Optimizer: AdamW (learning rate = 3e-4)
- Gradient clipping (max norm = 1.0)

## Features

### Stock Data (OHLCV)
- Open
- High
- Low
- Close
- Volume

### Technical Indicators
- Simple Moving Average (SMA 20)
- Relative Strength Index (RSI 14)
- MACD (line, signal, histogram)
- Average True Range (ATR 14)
- Volume percentage change

### Market Context
- SPY daily returns
- SPY rolling volatility (20-day standard deviation)

## Target Definition
Binary classification:

- 1 → Next-day return > 0 (UP)
- 0 → Next-day return ≤ 0 (DOWN)

This formulation avoids instability associated with direct price regression in non-stationary financial time-series data.

## Training Strategy
The model uses walk-forward validation:

- Training window: 1200 samples
- Testing window: 250 samples
- Rolling forward across the dataset

For each window:
1. Fit the scaler on training data only
2. Generate sequences (length = 60)
3. Train a new LSTM model
4. Evaluate on unseen forward data

This setup simulates real-world deployment and avoids look-ahead bias.

## Evaluation

### Metrics
- Directional accuracy
- Trade participation rate
- Filtered accuracy (confidence-based)

### Signal Logic
- BUY if probability > 0.55
- SELL if probability < 0.45
- HOLD otherwise

## Example Output
Using device: cuda

======================================================================
DIRECTIONAL SIGNAL RESULTS
----------------------------------------------------------------------
Ticker: GOOGL
Filtered Directional Accuracy : 61.40%
Trades Taken                 : 27.80%
Latest Probability (UP)       : 0.612
TRADING SIGNAL                : BUY
======================================================================

## Tech Stack
- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- yFinance

## Dataset
- Source: Yahoo Finance (via yFinance)
- Time range: 2016 to present
- Primary ticker: GOOGL
- Market proxy: SPY
- No proprietary or paid datasets required

## How to Run
pip install torch pandas numpy scikit-learn yfinance

python directional_lstm.py

## Important Notes
- This is a research and educational project, not a production trading system
- The model does not account for transaction costs, slippage, or liquidity constraints
- No risk management is implemented
- Results may not generalize to live trading environments

## Project Goals
This project explores:
- Proper machine learning evaluation for time-series data
- Avoiding data leakage in financial modeling
- Decision-making under uncertainty
- The relationship between confidence thresholds and trade frequency

## Future Improvements
- Evaluate across multiple stocks and sectors
- Add transaction cost and slippage modeling
- Compare against classical ML models (logistic regression, random forest, XGBoost)
- Implement hyperparameter tuning (grid search or Bayesian optimization)
- Explore transformer-based architectures
- Add interpretability (feature importance, SHAP analysis)
- Deploy as a web-based dashboard

## Summary
DirectionalLSTM demonstrates how deep learning can be applied to financial markets with a focus on realistic evaluation, probabilistic reasoning, and system-level design rather than naive price prediction.
