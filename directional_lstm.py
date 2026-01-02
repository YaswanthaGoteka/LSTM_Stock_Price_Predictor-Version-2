import torch                              # Import PyTorch core library for tensors and computation
import torch.nn as nn                     # Import PyTorch neural network modules
import numpy as np                        # Import NumPy for numerical array operations
import pandas as pd                       # Import Pandas for data manipulation and analysis
import yfinance as yf                     # Import yfinance to download financial market data
from sklearn.preprocessing import StandardScaler  # Import feature standardization utility
from sklearn.metrics import accuracy_score        # Import accuracy metric for classification

# Configuration
TICKER = "GOOGL"                          # Stock ticker symbol to analyze
MARKET = "SPY"                           # Market benchmark ETF for correlation/volatility
START_DATE = "2016-01-01"                # Start date for historical data
SEQ_LEN = 60                             # Number of time steps per LSTM input sequence
EPOCHS = 1000                            # Number of training epochs per walk-forward window
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU if available, else CPU

CONFIDENCE_THRESHOLD = 0.55              # Minimum probability required to take a trade
VOL_FILTER = 0.008                       # Volatility threshold (defined but not used in logic)

print(f"Using device: {DEVICE}")         # Display which device is being used for computation

# Data Download
df = yf.download(TICKER, start=START_DATE, auto_adjust=False, progress=False)  # Download stock price data
spy = yf.download(MARKET, start=START_DATE, auto_adjust=False, progress=False) # Download market benchmark data

df.columns = df.columns.get_level_values(0)    # Flatten multi-index columns for stock data
spy.columns = spy.columns.get_level_values(0)  # Flatten multi-index columns for market data

df = df[['Open','High','Low','Close','Volume']]  # Keep only relevant OHLCV columns
spy = spy[['Close']]                             # Keep only closing price for SPY

# Feature Engineering
df['SMA20'] = df['Close'].rolling(20).mean()    # 20-day simple moving average of price

delta = df['Close'].diff()                      # Day-to-day price change
gain = delta.clip(lower=0)                      # Positive price changes only
loss = -delta.clip(upper=0)                     # Negative price changes only
rs = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean()  # Relative strength
df['RSI14'] = 100 - (100 / (1 + rs))             # 14-period Relative Strength Index

ema12 = df['Close'].ewm(span=12, adjust=False).mean()  # 12-period exponential moving average
ema26 = df['Close'].ewm(span=26, adjust=False).mean()  # 26-period exponential moving average
df['MACD'] = ema12 - ema26                             # MACD line
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # MACD signal line
df['MACD_hist'] = df['MACD'] - df['MACD_signal']      # MACD histogram

tr = pd.concat([                                    # Compute true range components
    df['High'] - df['Low'],                         # High–Low range
    (df['High'] - df['Close'].shift()).abs(),       # High–Previous Close
    (df['Low'] - df['Close'].shift()).abs()         # Low–Previous Close
], axis=1).max(axis=1)                              # Take the maximum true range
df['ATR14'] = tr.ewm(alpha=1/14, adjust=False).mean()  # 14-period Average True Range

df['Volume_Change'] = df['Volume'].pct_change()    # Percentage change in trading volume

spy['SPY_Return'] = spy['Close'].pct_change()      # Daily market return
spy['SPY_Vol'] = spy['SPY_Return'].rolling(20).std()  # 20-day rolling market volatility

df = df.join(spy[['SPY_Return','SPY_Vol']], how='left')  # Merge market features into stock dataframe

# Classification Target
df['Return'] = df['Close'].pct_change().shift(-1)  # Next-day return (future-looking)
df['Target'] = (df['Return'] > 0).astype(int)      # Binary target: 1 if price goes up, else 0

df.dropna(inplace=True)                            # Remove rows with missing values

# Sequence Builder
def make_sequences(X, y, seq_len):                 # Function to build LSTM sequences
    xs, ys = [], []                                # Initialize feature and label lists
    for i in range(len(X) - seq_len):              # Loop over dataset with sliding window
        xs.append(X[i:i+seq_len])                  # Append feature sequence
        ys.append(y[i+seq_len])                    # Append label after sequence end
    return np.array(xs), np.array(ys)              # Convert lists to NumPy arrays

features = df.drop(columns=['Target','Return']).values.astype(np.float32)  # Feature matrix
target = df['Target'].values.astype(np.float32)    # Target labels

WINDOW_TRAIN = 1200                                # Number of samples used for training window
WINDOW_TEST = 250                                  # Number of samples used for testing window

signals, probs, truths = [], [], []                 # Containers for predictions and ground truth

# Model
class DirectionalRNN(nn.Module):                   # Define LSTM-based classification model
    def __init__(self, input_size):
        super().__init__()                          # Initialize base nn.Module
        self.lstm = nn.LSTM(                        # LSTM layer
            input_size,                             # Number of input features
            96,                                     # Hidden state size
            num_layers=2,                           # Number of stacked LSTM layers
            batch_first=True,                       # Input shape: (batch, seq, features)
            dropout=0.3                             # Dropout for regularization
        )
        self.fc = nn.Linear(96, 1)                  # Fully connected output layer

    def forward(self, x):
        x, _ = self.lstm(x)                         # Pass input through LSTM
        return self.fc(x[:, -1])                    # Use final timestep output for prediction

# Walk-Forward Training
for start in range(0, len(df) - WINDOW_TRAIN - WINDOW_TEST, WINDOW_TEST):
    train_slice = slice(start, start + WINDOW_TRAIN)  # Define training window
    test_slice = slice(start + WINDOW_TRAIN, start + WINDOW_TRAIN + WINDOW_TEST)  # Define test window

    scaler = StandardScaler().fit(features[train_slice])  # Fit scaler on training data only
    X_scaled = scaler.transform(features).astype(np.float32)  # Scale full dataset

    X_seq, y_seq = make_sequences(X_scaled, target, SEQ_LEN)  # Convert data to LSTM sequences

    X_train = torch.tensor(X_seq[train_slice.start:train_slice.stop], device=DEVICE)  # Training features
    y_train = torch.tensor(
        y_seq[train_slice.start:train_slice.stop],
        device=DEVICE
    ).unsqueeze(1)                               # Training labels (reshaped for loss function)

    X_test = torch.tensor(X_seq[test_slice.start:test_slice.stop], device=DEVICE)  # Test features
    y_test = y_seq[test_slice.start:test_slice.stop]  # Test labels (NumPy)

    model = DirectionalRNN(X_train.shape[2]).to(DEVICE)  # Initialize model
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)  # Optimizer
    loss_fn = nn.BCEWithLogitsLoss()              # Binary classification loss

    for _ in range(EPOCHS):                       # Training loop
        opt.zero_grad(set_to_none=True)           # Reset gradients
        loss = loss_fn(model(X_train), y_train)  # Compute loss
        loss.backward()                           # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        opt.step()                                # Update model weights

    model.eval()                                  # Switch model to evaluation mode
    with torch.no_grad():                         # Disable gradient computation
        logits = model(X_test).cpu().numpy().flatten()  # Raw model outputs
        prob = 1 / (1 + np.exp(-logits))          # Convert logits to probabilities

    probs.extend(prob)                            # Store predicted probabilities
    truths.extend(y_test)                         # Store true labels

# Signal Logic
probs = np.array(probs)                           # Convert probabilities to NumPy array
truths = np.array(truths)                         # Convert labels to NumPy array

trade_mask = probs > CONFIDENCE_THRESHOLD         # Identify confident trade signals
direction_pred = (probs > 0.5).astype(int)        # Direction prediction (up/down)

filtered_acc = accuracy_score(                    # Accuracy only on confident trades
    truths[trade_mask],
    direction_pred[trade_mask]
)

latest_signal = (                                 # Determine final trading signal
    "BUY" if probs[-1] > CONFIDENCE_THRESHOLD else
    "SELL" if probs[-1] < (1 - CONFIDENCE_THRESHOLD) else
    "HOLD"
)

print("\n" + "="*70)
print("DIRECTIONAL SIGNAL RESULTS")
print("-"*70)
print(f'Ticker: {TICKER}')
print(f"Filtered Directional Accuracy : {filtered_acc:.2%}")
print(f"Trades Taken                 : {trade_mask.mean():.2%}")
print(f"Latest Probability (UP)       : {probs[-1]:.3f}")
print(f"TRADING SIGNAL                : {latest_signal}")
print("="*70)
