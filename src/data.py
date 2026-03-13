import pandas as pd
import yfinance as yf
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


@st.cache_data
def load_spy_data() -> pd.DataFrame:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)


    if config.DATA_CACHE_FILE.exists():
        print(f"Loading data from cache: {config.DATA_CACHE_FILE}")

        df = pd.read_csv(
            config.DATA_CACHE_FILE,
            index_col=0,
            parse_dates=True
        )

        df.index.name = "Date"
        return df

  
    print(f"Downloading {config.TICKER} data from {config.START_DATE}...")

    df = yf.download(
        config.TICKER,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=False
    )

  
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.ffill()

    df.index.name = "Date"

    df.to_csv(config.DATA_CACHE_FILE)

    print(f"Data cached to: {config.DATA_CACHE_FILE}")

    return df

if __name__ == "__main__":
    df = load_spy_data()
    print(df.head())
    print("\nShape:", df.shape)