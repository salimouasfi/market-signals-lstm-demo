import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.rolling(window=period, min_periods=period).mean()
    loss = down.rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    return df

def make_features_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret1'] = df['Close'].pct_change()
    df['rsi14'] = compute_rsi(df['Close'], 14)
    df['mom5'] = df['Close'].pct_change(5)
    df['vol5'] = df['ret1'].rolling(5).std()
    # Label: direction J+1
    df['ret1_fwd'] = df['ret1'].shift(-1)
    df['label'] = (df['ret1_fwd'] > 0).astype(int)
    df = df.dropna()
    return df

def to_sequences(values: np.ndarray, labels: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

def time_split(df: pd.DataFrame, train_end: str, val_end: str):
    tr = df[df.index <= pd.to_datetime(train_end)].copy()
    va = df[(df.index > pd.to_datetime(train_end)) & (df.index <= pd.to_datetime(val_end))].copy()
    te = df[df.index > pd.to_datetime(val_end)].copy()
    return tr, va, te

def build_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def backtest(returns: pd.Series, proba: pd.Series, threshold: float = 0.5, cost_bps: float = 10.0):
    # Position: +1 if proba>thr else -1
    pos = proba.apply(lambda p: 1 if p > threshold else -1).shift(1).fillna(0)
    # Trade cost when position changes
    trades = pos.diff().abs().fillna(0) / 2.0  # change of 2 => 1 trade
    daily_cost = trades * (cost_bps / 10000.0)
    strat_ret = pos * returns - daily_cost
    equity = (1 + strat_ret).cumprod()
    sharpe = np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-12)
    peak = equity.cummax()
    mdd = ((equity - peak) / peak).min()
    return strat_ret, equity, float(sharpe), float(mdd)

def plot_equity(equity: pd.Series, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10,5))
    equity.plot()
    plt.title('Equity Curve')
    plt.ylabel('Cumulative Return (×)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='2024-12-31')
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--train_end', type=str, default='2021-12-31')
    parser.add_argument('--val_end', type=str, default='2022-12-31')
    parser.add_argument('--cost_bps', type=float, default=10.0)
    args = parser.parse_args()

    print(f"Loading data for {args.ticker} ...")
    raw = load_data(args.ticker, args.start, args.end)
    df = make_features_labels(raw)

    features = ['ret1', 'rsi14', 'mom5', 'vol5']
    tr, va, te = time_split(df, args.train_end, args.val_end)

    scaler = StandardScaler()
    tr_vals = scaler.fit_transform(tr[features].values)
    va_vals = scaler.transform(va[features].values)
    te_vals = scaler.transform(te[features].values)

    X_tr, y_tr = to_sequences(tr_vals, tr['label'].values, args.window)
    X_va, y_va = to_sequences(va_vals, va['label'].values, args.window)
    X_te, y_te = to_sequences(te_vals, te['label'].values, args.window)

    print(f"Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    model = build_model(input_shape=(args.window, len(features)))
    cb_early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=50,
        batch_size=64,
        callbacks=[cb_early],
        verbose=1
    )

    proba_te = model.predict(X_te, verbose=0).ravel()
    ap = float(average_precision_score(y_te, proba_te))
    print(f"Average Precision (AUC-PR) on Test: {ap:.4f}")

    # Align probabilities back to dates
    te_idx = te.index[args.window:]  # because of sequence window
    proba_series = pd.Series(proba_te, index=te_idx, name='proba')
    ret_series = df.loc[te_idx, 'ret1_fwd']  # forward return aligns with label definition

    strat_ret, equity, sharpe, mdd = backtest(ret_series, proba_series, threshold=0.5, cost_bps=args.cost_bps)
    print(f"Sharpe (annualisé): {sharpe:.2f} | Max Drawdown: {mdd:.2%}")

    out_plot = os.path.join('outputs', 'equity_curve.png')
    plot_equity(equity, out_plot)
    print(f"Saved equity curve to {out_plot}")

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
