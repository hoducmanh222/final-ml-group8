"""Configuration settings for weather forecasting system."""

import numpy as np
import random

# ============================================================
# RANDOM SEED CONFIGURATION
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ============================================================
# DATA CONFIGURATION
# ============================================================
DATA_PATH_DAILY = "data/weather_hcm_daily.csv"  
DATA_PATH_HOURLY = "data/weather_hcm_hourly.csv"
# ============================================================
# FEATURE ENGINEERING - DAILY DATA
# ============================================================
DAILY_HORIZONS = [1, 2, 3, 4, 5]  # Forecast 5 days ahead
DAILY_LAG_VALUES = [1, 2, 3, 7, 14]  # Lag features
DAILY_ROLLING_WINDOWS = [3, 7, 14]  # Rolling window sizes
DAILY_SEQUENCE_LENGTH = 30  # For sequence models

# ============================================================
# FEATURE ENGINEERING - HOURLY DATA
# ============================================================
HOURLY_HORIZONS = [1, 2, 3, 6, 12, 18, 24, 48, 72, 96, 120]
HOURLY_LAG_VALUES = [1, 2, 3, 6, 12, 18, 24, 48, 96, 120, 144]
HOURLY_ROLLING_WINDOWS = [3, 6, 12, 18, 24, 48, 72, 96, 120, 144]

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================
TRAIN_TEST_SPLIT_RATIO = 0.85
LEAKAGE_BUFFER_DAILY = 14  # 14 days buffer
LEAKAGE_BUFFER_HOURLY = 336  # 14 days (336 hours) buffer

# ============================================================
# CROSS-VALIDATION
# ============================================================
CV_N_SPLITS = 5

# ============================================================
# MODEL HYPERPARAMETERS (BEST FOUND)
# ============================================================

# HistGradientBoosting - Daily Data
HGB_DAILY_PARAMS = {
    'learning_rate': 0.02014148621347681,
    'max_depth': 3,
    'max_iter': 493,
    'min_samples_leaf': 25,
    'l2_regularization': 9.852173472173725e-06
}

# HistGradientBoosting - Hourly Data
HGB_HOURLY_PARAMS = {
    'learning_rate': 0.07172313382240242,
    'max_depth': 12,
    'min_samples_leaf': 131,
    'l2_regularization': 0.0026568368934158193
}

# Random Forest - Manual Tune
RF_MANUAL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_leaf': 10,
    'max_features': 0.8
}

# Huber Regressor - Optuna Tuned
HUBER_OPTUNA_PARAMS = {
    'alpha': 0.6258952607607242,
    'epsilon': 1.5599425526665442
}

# ============================================================
# RETRAINING SYSTEM CONFIGURATION
# ============================================================
RETRAINING_CONFIG = {
    'alert_threshold': 0.15,  # 15% degradation triggers alert
    'retrain_window': 90,  # Use 90 days for retraining
    'validation_window': 30,  # 30 days for validation
    'max_history': 180  # Keep 180 days of history
}

# ============================================================
# ONNX EXPORT CONFIGURATION
# ============================================================
ONNX_CONFIG = {
    'target_opset': 15,
    'sample_size': 200,  # Number of samples for benchmarking
    'output_dir': 'models/'
}

# ============================================================
# PREPROCESSING COLUMN TYPES
# ============================================================
CATEGORICAL_COLS = ['preciptype', 'conditions', 'icon']
TEXT_COLS = ['description']

# Columns to drop
DROP_COLS = ['name', 'address', 'resolvedAddress', 'latitude', 'longitude', 'source']
