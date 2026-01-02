# LSTM_Stock_Price_Predictor-Version-2 or DirectionalLSTM — Confidence-Based Stock Direction Prediction

DirectionalLSTM is a deep learning project that predicts the next-day direction (up or down) of a stock’s closing price using an LSTM neural network built in PyTorch. Instead of predicting exact prices, the model focuses on directional movement and confidence, allowing analysis of the trade-off between prediction accuracy and trade frequency in real, non-stationary financial markets.

The project uses walk-forward (rolling window) validation to better simulate real-world trading conditions and reduce look-ahead bias. The model outputs probabilities for upward price movement, which are then filtered using configurable confidence thresholds to generate BUY, SELL, or HOLD signals.

### Features:

* Predicts next-day stock price direction (UP / DOWN)

* Uses LSTM neural networks for sequential time-series modeling

* Implements walk-forward validation instead of static train-test splits

* Outputs probabilistic predictions rather than hard labels

* Supports confidence-based trade filtering

* Incorporates market context using SPY returns and volatility

* Designed to study accuracy vs trade participation trade-offs

* Built for experimentation, not guaranteed profitability


### Input Features:

* The model uses a combination of stock-specific and market-wide features:

Stock Data (Daily OHLCV):

* Open

* High

* Low

* Close

* Volume

### Technical Indicators (Most commonly used):

1. Simple Moving Average (SMA)

2. Relative Strength Index (RSI)

3. MACD, MACD Signal, MACD Histogram

4. Average True Range (ATR)

5. Volume percentage change

### Market Context:

- SPY daily returns

- SPY rolling volatility

- Target Definition

Binary classification

1 → Next-day return is positive (BUY)

0 → Next-day return is negative or zero (SELL or HOLD)

This formulation avoids many issues associated with direct price regression on non-stationary financial data.

### Model Architecture:

* 2-layer LSTM

* 96 hidden units

* Dropout regularization

* Fully connected output layer

* Binary Cross Entropy with Logits loss

* AdamW optimizer with gradient clipping

### Evaluation Metrics:

1. Directional accuracy

2. Trade participation rate

3. Confidence threshold vs accuracy analysis

4. Filtered accuracy (only high-confidence trades)

### Technologies Used:

1. Python

2. PyTorch (deep learning)

3. Pandas (data processing)

4. NumPy (numerical operations)

5. yFinance (market data)

6. Scikit-learn (scaling and evaluation)

### Dataset:

- Historical daily stock and market data downloaded from Yahoo Finance

- Time range: 2016–present

- Ticker (Used in code): GOOGL (stock), SPY (market proxy)

- No proprietary or paid datasets are required.

### How to Run:

1. Clone the repository or copy the script into a Python file

2. Install dependencies:

    - pip install torch pandas numpy scikit-learn yfinance

3. Run the script:
    - python directional_lstm.py

### Example Output:
Using device: cuda

Filtered Directional Accuracy : 61.4%

Trades Taken                 : 27.8%

Latest Probability (UP)       : 0.612

TRADING SIGNAL                : BUY

### Project Goal:

The goal of this project is not to claim market-beating performance, but to explore:

- Proper machine learning evaluation on time-series data

- Decision making under uncertainty

- The impact of confidence thresholds on real world ML systems

**The model was developed iteratively, starting from a simple LSTM baseline and refined through experimentation and AI-assisted exploration of alternative architectures and validation strategies. All experiments, evaluations, and analyses were conducted independently.**

### Future Improvements:

- Test across multiple stocks and sectors

- Add transaction costs and slippage modeling

- Compare against non-deep learning baselines

- Add interpretability (feature importance / ablation)

- Explore reinforcement learning approaches

- Deploy as a web-based dashboard

### About:

**Predicts stock price direction using LSTM neural networks, walk-forward validation, and confidence-based decision thresholds to study accuracy–trade frequency trade-offs in financial time-series data.**
