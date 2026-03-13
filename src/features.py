import pandas as pd
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["rolling_volatility"] = (
        df["log_return"]
        .rolling(config.VOL_WINDOW)
        .std()
    )

    sma_short = df["Close"].rolling(config.SMA_SHORT_WINDOW).mean()
    sma_long = df["Close"].rolling(config.SMA_LONG_WINDOW).mean()

    df["sma_ratio"] = sma_short / sma_long

    df["rsi"] = compute_rsi(df["Close"], config.RSI_WINDOW)

    df["log_return"] = (
        np.log(df["Close"] / df["Close"].shift(config.RETURN_WINDOW))
    )

    df = df.dropna()

    return df

from src.data import load_spy_data
from src.features import create_features

df = load_spy_data()
df_feat = create_features(df)

print(df_feat.head())
print(df_feat.columns)
print(df_feat.shape)