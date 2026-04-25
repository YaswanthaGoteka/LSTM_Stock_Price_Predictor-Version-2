# DirectionalLSTM — Confidence-Based Stock Direction Prediction

## Overview
DirectionalLSTM is a deep learning project that predicts the **next-day direction (UP/DOWN)** of stock prices using an LSTM neural network built with PyTorch.

Unlike traditional price prediction models, this project focuses on **probabilistic direction forecasting** and **confidence-based decision making**, allowing analysis of the trade-off between prediction accuracy and trade frequency in real, non-stationary financial markets.

The system uses **walk-forward (rolling window) validation** to better simulate real-world deployment and avoid look-ahead bias.

---

## Key Features
- Predicts next-day stock direction (UP / DOWN)
- LSTM-based sequential time-series modeling
- Walk-forward validation (no static train/test leakage)
- Probabilistic outputs instead of hard classifications
- Confidence-based filtering (BUY / SELL / HOLD)
- Incorporates market context (SPY returns & volatility)
- Analyzes accuracy vs trade participation trade-offs

---

## Model Architecture
- 2-layer LSTM
- 96 hidden units
- Dropout regularization
- Fully connected output layer
- Loss: Binary Cross Entropy with Logits
- Optimizer: AdamW
- Gradient clipping for stability

---

## Features Used

### Stock Data (Daily OHLCV)
- Open, High, Low, Close, Volume

### Technical Indicators
- Simple Moving Average (SMA)
- Relative Strength Index (RSI)
- MACD (Line, Signal, Histogram)
- Average True Range (ATR)
- Volume % Change

### Market Context
- SPY daily returns
- SPY rolling volatility

---

## Target Definition
Binary classification:

- `1` → Next-day return is positive (UP / BUY)
- `0` → Next-day return is negative or zero (DOWN / SELL/HOLD)

This avoids instability issues associated with direct price regression in financial time-series data.

---

## Evaluation Metrics
- Directional Accuracy
- Trade Participation Rate
- Confidence Threshold vs Accuracy Curve
- Filtered Accuracy (high-confidence trades only)

---

## Example Results
