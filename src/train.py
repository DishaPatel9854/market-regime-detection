import sys
from pathlib import Path
import pandas as pd
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent))

from src.data import load_spy_data
from src.features import create_features
from src.model import (
    assign_regime_labels,
    build_kmeans_matrix,
    build_hmm_matrix,
    train_kmeans,
    train_hmm,
    save_models
)

import config


def main():

    print("\nLoading data...")
    df = load_spy_data()

    print("Creating features...")
    df_feat = create_features(df)

    print("Preparing KMeans features...")
    X_kmeans = build_kmeans_matrix(df_feat)

    print("Training KMeans model...")
    kmeans_model, scaler = train_kmeans(X_kmeans)

    print("Preparing HMM features...")
    X_hmm = build_hmm_matrix(df_feat)

    print("Training HMM model...")
    hmm_model = train_hmm(X_hmm)

    print("Saving models...")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    save_models(kmeans_model, hmm_model, scaler)

    print("\nTraining complete.")
    print("Models saved to:", config.MODEL_DIR)

    print("Precomputing regimes...")

    X_kmeans = build_kmeans_matrix(df_feat)
    labels_kmeans = kmeans_model.predict(scaler.transform(X_kmeans))
    df_kmeans, regime_map = assign_regime_labels(df_feat, labels_kmeans)

    X_hmm = build_hmm_matrix(df_feat)
    labels_hmm = hmm_model.predict(X_hmm)

    df_hmm = df_feat.copy()
    df_hmm["cluster"] = labels_hmm

    mean_returns = df_hmm.groupby("cluster")["log_return"].mean()
    sorted_clusters = mean_returns.sort_values().index.tolist()

    regime_names = ["Bear", "Sideways", "Bull"]
    hmm_map = {cluster: regime_names[i] for i, cluster in enumerate(sorted_clusters)}

    df_kmeans["hmm_regime"] = pd.Series(labels_hmm, index=df_feat.index).map(hmm_map)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_kmeans.to_csv(config.OUTPUT_DIR / "regime_results.csv")
    print("Regime results saved.")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()







