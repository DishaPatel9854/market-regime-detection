from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# ── Data ─────────────────────────────────────────────────────────────────────
TICKER     = "SPY"
START_DATE = "2005-01-01"
END_DATE   = None         

# ── Feature Engineering Windows ──────────────────────────────────────────────
VOL_WINDOW        = 20    
SMA_SHORT_WINDOW  = 20    
SMA_LONG_WINDOW   = 50   
RSI_WINDOW        = 14    
RETURN_WINDOW     = 5     

# ── Model ─────────────────────────────────────────────────────────────────────
N_CLUSTERS  = 3         
HMM_STATES  = 3       
RANDOM_SEED = 42

# ── Regime Labels (assigned post-clustering by mean return rank) ──────────────
REGIME_LABELS = {
    0: "Bear",
    1: "Sideways",
    2: "Bull"
}

REGIME_COLORS = {
    "Bear":     "#ef4444",   
    "Sideways": "#f59e0b",   
    "Bull":     "#22c55e",   
}

# ── Saved Model Filenames ─────────────────────────────────────────────────────
KMEANS_MODEL_FILE = MODEL_DIR / "kmeans_model.pkl"
HMM_MODEL_FILE    = MODEL_DIR / "hmm_model.pkl"
SCALER_FILE       = MODEL_DIR / "scaler.pkl"
DATA_CACHE_FILE   = DATA_DIR  / "spy_data.csv"

KMEANS_FEATURES = [
    "rolling_volatility",
    "sma_ratio",
    "rsi",
    "log_return"
]

HMM_FEATURES = [
    "log_return",
    "rolling_volatility"
]