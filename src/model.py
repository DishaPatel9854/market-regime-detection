import pandas as pd
import numpy as np
import sys
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def build_kmeans_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix used for KMeans."""

    features = [
        "rolling_volatility",
        "sma_ratio",
        "rsi",
        "log_return"
    ]

    return df[features].copy()


def build_hmm_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix used for HMM."""

    features = [
        "log_return",
        "rolling_volatility"
    ]

    return df[features].copy()

def train_kmeans(X: pd.DataFrame):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=config.N_CLUSTERS,
        random_state=config.RANDOM_SEED,
        n_init=20
    )

    model.fit(X_scaled)

    return model, scaler



def train_hmm(X: pd.DataFrame):

    model = GaussianHMM(
        n_components=config.HMM_STATES,
        covariance_type="full",
        n_iter=500,
        random_state=config.RANDOM_SEED
    )

    model.fit(X)

    return model



def assign_regime_labels(df: pd.DataFrame, labels):

    df = df.copy()
    df["cluster"] = labels

    mean_returns = df.groupby("cluster")["log_return"].mean()
    sorted_clusters = mean_returns.sort_values().index.tolist()

    regime_names = ["Bear", "Sideways", "Bull"]

    regime_map = {}

    for i, cluster in enumerate(sorted_clusters):
        regime_map[cluster] = regime_names[i]

    df["regime"] = df["cluster"].map(regime_map)

    return df, regime_map


def predict_regime(model, X, scaler=None):

    if scaler is not None:
        X = scaler.transform(X)

    labels = model.predict(X)

    return labels


def save_models(kmeans_model, hmm_model, scaler):

    joblib.dump(kmeans_model, config.KMEANS_MODEL_FILE)
    joblib.dump(hmm_model, config.HMM_MODEL_FILE)
    joblib.dump(scaler, config.SCALER_FILE)