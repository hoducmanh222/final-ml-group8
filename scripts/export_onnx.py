"""Export trained models to ONNX format with benchmarking."""

import sys
import os
import joblib
import time
import numpy as np
import pandas as pd
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime import InferenceSession
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *

def export_model_to_onnx(model_path, output_path, X_sample, model_name="model"):
    """
    Export a scikit-learn model to ONNX format.
    
    Parameters:
    -----------
    model_path : str
        Path to the pickle model file
    output_path : str
        Path to save the ONNX model
    X_sample : array-like
        Sample input data for determining input shape
    model_name : str
        Name of the model for logging
    """
    print(f"\n{'='*80}")
    print(f"Exporting {model_name} to ONNX")
    print(f"{'='*80}")
    
    # Load pickle model
    print("Loading pickle model...")
    model_data = joblib.load(model_path)
    
    if isinstance(model_data, dict):
        pipeline = model_data.get('pipeline') or model_data.get('model')
    else:
        pipeline = model_data
    
    # Extract regressor from pipeline
    if hasattr(pipeline, 'named_steps'):
        preprocess = pipeline.named_steps['preprocess']
        regressor = pipeline.named_steps['regressor']
        
        # Transform sample data
        X_transformed = preprocess.transform(X_sample)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
    else:
        regressor = pipeline
        X_transformed = X_sample
    
    # Define ONNX input type
    n_features = X_transformed.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    
    print(f"Model input shape: [None, {n_features}]")
    
    # Convert to ONNX
    print("Converting to ONNX...")
    try:
        onnx_model = convert_sklearn(
            regressor,
            initial_types=initial_type,
            target_opset=ONNX_CONFIG['target_opset']
        )
        
        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"✅ ONNX model saved: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        return True, X_transformed, preprocess if hasattr(pipeline, 'named_steps') else None
        
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        return False, None, None


def benchmark_models(
    pickle_path,
    onnx_path,
    X_test,
    y_test,
    preprocessor=None,
    n_samples=200
):
    """
    Benchmark pickle vs ONNX model performance.
    
    Parameters:
    -----------
    pickle_path : str
        Path to pickle model
    onnx_path : str
        Path to ONNX model
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
    preprocessor : sklearn transformer, optional
        Preprocessor to transform X_test
    n_samples : int
        Number of samples for benchmarking
        
    Returns:
    --------
    summary : DataFrame
        Benchmark comparison table
    """
    print(f"\n{'='*80}")
    print("Benchmarking: Pickle vs ONNX")
    print(f"{'='*80}")
    
    # Load pickle model
    model_data = joblib.load(pickle_path)
    
    if isinstance(model_data, dict):
        pickle_model = model_data.get('pipeline') or model_data.get('model')
    else:
        pickle_model = model_data
    
    # Prepare test samples
    X_sample = X_test[:n_samples]
    y_true = y_test[:n_samples]
    
    # Transform if needed
    if preprocessor is not None:
        X_sample_transformed = preprocessor.transform(X_sample)
        if hasattr(X_sample_transformed, 'toarray'):
            X_sample_transformed = X_sample_transformed.toarray()
        X_sample_transformed = X_sample_transformed.astype(np.float32)
    else:
        X_sample_transformed = X_sample.astype(np.float32)
    
    # === Sklearn Inference ===
    print("\nRunning Sklearn inference...")
    start = time.time()
    y_pred_sklearn = pickle_model.predict(X_sample)
    t_sklearn = time.time() - start
    print(f"  Time: {t_sklearn:.4f}s")
    
    # === ONNX Inference ===
    print("Running ONNX inference...")
    sess = InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    
    start = time.time()
    y_pred_onnx = sess.run(None, {input_name: X_sample_transformed})[0]
    t_onnx = time.time() - start
    print(f"  Time: {t_onnx:.4f}s")
    
    # === Metrics Comparison ===
    # Sklearn metrics
    r2_sklearn = r2_score(y_true, y_pred_sklearn)
    rmse_sklearn = np.sqrt(mean_squared_error(y_true, y_pred_sklearn))  # Changed this line
    mae_sklearn = mean_absolute_error(y_true, y_pred_sklearn)
    
    # ONNX metrics
    r2_onnx = r2_score(y_true, y_pred_onnx)
    rmse_onnx = np.sqrt(mean_squared_error(y_true, y_pred_onnx))  # Changed this line
    mae_onnx = mean_absolute_error(y_true, y_pred_onnx)
    
    # File sizes
    size_pkl = os.path.getsize(pickle_path) / 1024
    size_onnx = os.path.getsize(onnx_path) / 1024
    
    # Speed-up factor
    speedup = (t_sklearn / t_onnx) if t_onnx > 0 else np.inf
    
    # Create summary
    summary = pd.DataFrame({
        "Format": ["Pickle (.pkl)", "ONNX (.onnx)"],
        "Size (KB)": [size_pkl, size_onnx],
        "Inference Time (s)": [t_sklearn, t_onnx],
        "Speed-up Factor": [1.0, speedup],
        "R² Score": [r2_sklearn, r2_onnx],
        "RMSE": [rmse_sklearn, rmse_onnx],
        "MAE": [mae_sklearn, mae_onnx]
    })
    
    print("\n📊 ONNX vs Pickle Benchmark Summary:")
    print(summary.to_string(index=False))
    
    # Analysis
    print(f"\n{'='*80}")
    print("Analysis:")
    print(f"  • Speed improvement: {speedup:.2f}x faster")
    print(f"  • Size reduction: {(1 - size_onnx/size_pkl)*100:.1f}% smaller")
    print(f"  • Accuracy difference (RMSE): {abs(rmse_sklearn - rmse_onnx):.6f}")
    
    if abs(rmse_sklearn - rmse_onnx) < 0.001:
        print("  ✅ ONNX model maintains accuracy!")
    else:
        print("  ⚠️ Small accuracy difference detected")
    
    print(f"{'='*80}\n")
    
    return summary


def main():
    """Main export and benchmark pipeline."""
    print("\n" + "="*80)
    print("MODEL EXPORT TO ONNX")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Paths
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Example: Export the final ensemble model
    model_name = "final_model"
    pickle_path = models_dir / f"{model_name}.pkl"
    onnx_path = models_dir / f"{model_name}.onnx"
    
    if not pickle_path.exists():
        print(f"❌ Model not found: {pickle_path}")
        print("Please train the model first using scripts/train_models.py")
        return
    
    # Load sample data for shape inference
    print("\nLoading sample data...")
    try:
        import pandas as pd
        from src.feature_engineering import prepare_supervised_dataset
        
        # Fetch data
        df = pd.read_csv(DATA_PATH_DAILY)
        
        # Prepare features
        feature_df, target_df, target_cols, _ = prepare_supervised_dataset(
            df,
            horizons=DAILY_HORIZONS,
            lag_values=DAILY_LAG_VALUES,
            rolling_windows=DAILY_ROLLING_WINDOWS,
            sequence_length=DAILY_SEQUENCE_LENGTH
        )
        
        # Split data
        split_idx = int(len(feature_df) * TRAIN_TEST_SPLIT_RATIO)
        X_train = feature_df.drop(columns=['datetime']).iloc[:split_idx]
        y_train = target_df[target_cols].iloc[:split_idx]
        X_test = feature_df.drop(columns=['datetime']).iloc[split_idx + LEAKAGE_BUFFER_DAILY:]
        y_test = target_df[target_cols].iloc[split_idx + LEAKAGE_BUFFER_DAILY:]
        
        print(f"✅ Data loaded: {len(X_test)} test samples")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Export to ONNX
    success, X_transformed, preprocessor = export_model_to_onnx(
        pickle_path,
        onnx_path,
        X_train,
        model_name
    )
    
    if not success:
        print("Export failed. Exiting.")
        return
    
    # Benchmark
    print("\nStarting benchmark...")
    benchmark_models(
        pickle_path,
        onnx_path,
        X_test,
        y_test,
        preprocessor=preprocessor,
        n_samples=ONNX_CONFIG['sample_size']
    )
    
    print("\n✅ ONNX export and benchmark complete!")


if __name__ == "__main__":
    main()