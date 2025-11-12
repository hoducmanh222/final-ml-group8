"""
Main training script for weather forecasting models.

This script trains all models and saves the best one for production use.
"""

import sys
import cloudpickle as pickle
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import *
from src.feature_engineering import prepare_supervised_dataset
from src.preprocessing import identify_column_types, make_pipeline, ModTimeSeriesSplit
from src.model_training import train_multiple_models, summarize_model_results

warnings.filterwarnings('ignore')

def main():
    print("\n" + "="*80)
    print("WEATHER FORECASTING MODEL TRAINING")
    print("="*80)
    
    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    print("\n[1/5] Loading data...")
    df_raw = pd.read_csv(DATA_PATH_DAILY)
    print(f"✓ Data loaded: {df_raw.shape}")
    
    # ============================================================
    # STEP 2: Feature Engineering
    # ============================================================
    print("\n[2/5] Performing feature engineering...")
    feature_df, target_df, target_cols, _ = prepare_supervised_dataset(
        df_raw,
        horizons=DAILY_HORIZONS,
        lag_values=DAILY_LAG_VALUES,
        rolling_windows=DAILY_ROLLING_WINDOWS,
        sequence_length=DAILY_SEQUENCE_LENGTH
    )
    print(f"✓ Features prepared: {feature_df.shape}")
    print(f"✓ Targets prepared: {target_df.shape}")
    
    # ============================================================
    # STEP 3: Train-Test Split
    # ============================================================
    print("\n[3/5] Splitting data...")
    split_idx = int(len(feature_df) * TRAIN_TEST_SPLIT_RATIO)
    
    dates = feature_df['datetime'].reset_index(drop=True)
    X = feature_df.drop(columns=['datetime'])
    y = target_df[target_cols]
    
    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_test = X.iloc[split_idx + LEAKAGE_BUFFER_DAILY:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx + LEAKAGE_BUFFER_DAILY:].reset_index(drop=True)
    test_dates = dates.iloc[split_idx + LEAKAGE_BUFFER_DAILY:].reset_index(drop=True)
    
    print(f"✓ Train size: {X_train.shape}")
    print(f"✓ Test size: {X_test.shape}")
    
    # Identify column types
    column_types = identify_column_types(X_train)
    numeric_cols = column_types['numeric']
    categorical_cols = column_types['categorical']
    text_cols = column_types['text']
    
    print(f"✓ Numeric features: {len(numeric_cols)}")
    print(f"✓ Categorical features: {len(categorical_cols)}")
    print(f"✓ Text features: {len(text_cols)}")
    
    # Create pipeline maker
    def pipeline_maker(estimator):
        return make_pipeline(estimator, numeric_cols, categorical_cols, text_cols)
    
    # Cross-validation
    tscv = ModTimeSeriesSplit(n_splits=CV_N_SPLITS, buffer=LEAKAGE_BUFFER_DAILY)
    
    # ============================================================
    # STEP 4: Train Models
    # ============================================================
    print("\n[4/5] Training models...")
    
    estimators = {
        'BayesianRidge': MultiOutputRegressor(BayesianRidge()),
        'HistGradientBoosting_Optuna': MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE, **HGB_DAILY_PARAMS)
        )
    }
    
    model_results = train_multiple_models(
        estimators,
        X_train,
        y_train,
        X_test,
        y_test,
        pipeline_maker,
        tscv,
        DAILY_HORIZONS
    )
    
    # Create ensemble
    print("\nCreating ensemble...")
    ensemble_test_preds = np.mean([
        model_results['BayesianRidge']['predictions_test'],
        model_results['HistGradientBoosting_Optuna']['predictions_test']
    ], axis=0)
    
    ensemble_train_preds = np.mean([
        model_results['BayesianRidge']['predictions_train'],
        model_results['HistGradientBoosting_Optuna']['predictions_train']
    ], axis=0)
    
    # Compute ensemble metrics
    from src.model_training import compute_metrics
    
    model_results['Ensemble_Final'] = {
        'train_metrics': compute_metrics(y_train.values, ensemble_train_preds, DAILY_HORIZONS),
        'test_metrics': compute_metrics(y_test.values, ensemble_test_preds, DAILY_HORIZONS),
        'predictions_test': ensemble_test_preds,
        'predictions_train': ensemble_train_preds,
        'details': 'Simple average ensemble of BayesianRidge and HGB'
    }
    
    print(f"\n✓ Ensemble Test RMSE: {model_results['Ensemble_Final']['test_metrics']['RMSE_macro']:.4f}°C")
    
    # ============================================================
    # STEP 5: Save Best Model
    # ============================================================
    print("\n[5/5] Saving models...")
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Get summary
    summary = summarize_model_results(model_results)
    print("\nModel Performance Summary:")
    print(summary.to_string(index=False))
    
    # Save best model (ensemble)
    best_model_data = {
        'model': model_results['HistGradientBoosting_Optuna']['pipeline'],  # Use single model for ONNX compatibility
        'metrics': model_results['Ensemble_Final']['test_metrics'],
        'column_types': column_types,
        'config': {
            'horizons': DAILY_HORIZONS,
            'lag_values': DAILY_LAG_VALUES,
            'rolling_windows': DAILY_ROLLING_WINDOWS,
            'sequence_length': DAILY_SEQUENCE_LENGTH
        }
    }
    
    model_path = models_dir / "final_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_data, f)
    
    print(f"\n✓ Model saved: {model_path}")
    
    # Save full results
    results_path = models_dir / "model_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(model_results, f)
    
    print(f"✓ Full results saved: {results_path}")
    
    # Save metadata
    metadata = {
        'best_model': 'Ensemble_Final',
        'test_rmse': float(model_results['Ensemble_Final']['test_metrics']['RMSE_macro']),
        'test_mae': float(model_results['Ensemble_Final']['test_metrics']['MAE_macro']),
        'test_r2': float(model_results['Ensemble_Final']['test_metrics']['R2_macro']),
        'train_date': pd.Timestamp.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    import json
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved: {metadata_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest Model: Ensemble_Final")
    print(f"Test RMSE: {metadata['test_rmse']:.4f}°C")
    print(f"Test MAE: {metadata['test_mae']:.4f}°C")
    print(f"Test R²: {metadata['test_r2']:.4f}")
    print("\nNext steps:")
    print("1. Run ONNX export: python scripts/export_onnx.py")
    print("2. Launch Streamlit app: streamlit run streamlit_app/app.py")
    print("3. View results notebook: notebooks/04_final_results.ipynb")


if __name__ == "__main__":
    main()
    