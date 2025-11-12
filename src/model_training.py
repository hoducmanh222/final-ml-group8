"""Model training and evaluation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from tqdm.auto import tqdm


def compute_metrics(y_true, y_pred, horizons  =  [1, 2, 3, 4, 5]):
    """
    Compute comprehensive evaluation metrics for multi-horizon forecasting.
    
    Parameters:
    -----------
    y_true : array-like
        True values (n_samples, n_horizons)
    y_pred : array-like
        Predicted values (n_samples, n_horizons)
    horizons : list
        List of forecast horizons
        
    Returns:
    --------
    metrics : dict
        Dictionary of metrics for each horizon and macro averages
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {}
    
    rmse_vals, mae_vals, mape_vals, r2_vals = [], [], [], []
    
    for idx, horizon in enumerate(horizons):
        actual = y_true[:, idx]
        pred = y_pred[:, idx]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        
        # MAPE (avoid division by zero)
        mask = np.abs(actual) > 1e-6
        mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100 if mask.any() else np.nan
        
        r2 = r2_score(actual, pred)
        
        # Store per-horizon metrics
        metrics[f'RMSE_h{horizon}'] = rmse
        metrics[f'MAE_h{horizon}'] = mae
        metrics[f'MAPE_h{horizon}'] = mape
        metrics[f'R2_h{horizon}'] = r2
        
        # Collect for macro averages
        rmse_vals.append(rmse)
        mae_vals.append(mae)
        if not np.isnan(mape):
            mape_vals.append(mape)
        r2_vals.append(r2)
    
    # Macro averages
    metrics['RMSE_macro'] = float(np.mean(rmse_vals))
    metrics['MAE_macro'] = float(np.mean(mae_vals))
    metrics['MAPE_macro'] = float(np.mean(mape_vals)) if mape_vals else np.nan
    metrics['R2_macro'] = float(np.mean(r2_vals))
    
    return metrics


def cross_val_evaluate(pipeline, X_data, y_data, cv, horizons):
    """
    Perform cross-validation evaluation.
    
    Parameters:
    -----------
    pipeline : sklearn Pipeline
        Model pipeline
    X_data : DataFrame
        Features
    y_data : DataFrame
        Targets
    cv : cross-validator
        Cross-validation splitter
    horizons : list
        Forecast horizons
        
    Returns:
    --------
    fold_metrics : list
        Metrics for each fold
    """
    fold_metrics = []
    for train_idx, val_idx in cv.split(X_data):
        X_tr, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_tr, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]
        
        model = clone(pipeline)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        fold_metrics.append(compute_metrics(y_val.values, preds, horizons))
    
    return fold_metrics


def aggregate_metrics(fold_metrics):
    """
    Aggregate metrics across cross-validation folds.
    
    Parameters:
    -----------
    fold_metrics : list
        List of metric dictionaries from each fold
        
    Returns:
    --------
    aggregated : dict
        Mean metrics across all folds
    """
    if not fold_metrics:
        return None
    
    keys = fold_metrics[0].keys()
    return {k: float(np.nanmean([fold[k] for fold in fold_metrics])) for k in keys}


def train_model(
    model_name: str,
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    pipeline_maker,
    cv,
    horizons: List[int],
    details: str = ""
) -> Dict:
    """
    Train a model and evaluate on train/test sets with cross-validation.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    estimator : sklearn estimator
        Model to train
    X_train, y_train : DataFrame
        Training data
    X_test, y_test : DataFrame
        Test data
    pipeline_maker : callable
        Function that creates a pipeline given an estimator
    cv : cross-validator
        Cross-validation splitter
    horizons : list
        Forecast horizons
    details : str
        Model description
        
    Returns:
    --------
    result : dict
        Dictionary with pipeline, metrics, and predictions
    """
    pipeline = pipeline_maker(estimator)
    
    # Cross-validation
    cv_metrics = aggregate_metrics(cross_val_evaluate(pipeline, X_train, y_train, cv, horizons))
    
    # Train on full training set
    pipeline.fit(X_train, y_train)
    
    # Predictions
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    result = {
        'pipeline': pipeline,
        'cv_metrics': cv_metrics,
        'train_metrics': compute_metrics(y_train.values, train_pred, horizons),
        'test_metrics': compute_metrics(y_test.values, test_pred, horizons),
        'predictions_test': test_pred,
        'predictions_train': train_pred,
        'details': details or f'{model_name} trained with sklearn pipeline'
    }
    
    return result


def train_multiple_models(
    estimators: Dict,
    X_train,
    y_train,
    X_test,
    y_test,
    pipeline_maker,
    cv,
    horizons: List[int]
) -> Dict:
    """
    Train multiple models and collect results.
    
    Parameters:
    -----------
    estimators : dict
        Dictionary of {model_name: estimator}
    X_train, y_train : DataFrame
        Training data
    X_test, y_test : DataFrame
        Test data
    pipeline_maker : callable
        Function that creates a pipeline given an estimator
    cv : cross-validator
        Cross-validation splitter
    horizons : list
        Forecast horizons
        
    Returns:
    --------
    model_results : dict
        Dictionary of {model_name: result_dict}
    """
    model_results = {}
    
    for name, estimator in tqdm(estimators.items(), desc='Training models'):
        result = train_model(
            name, estimator, X_train, y_train, X_test, y_test,
            pipeline_maker, cv, horizons
        )
        model_results[name] = result
        
        print(f"{name} CV RMSE: {result['cv_metrics']['RMSE_macro']:.4f}, "
              f"Test RMSE: {result['test_metrics']['RMSE_macro']:.4f}")
    
    return model_results


def summarize_model_results(model_results: Dict) -> pd.DataFrame:
    """
    Create summary DataFrame of model results.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary of model results
        
    Returns:
    --------
    summary_df : DataFrame
        Summary table sorted by test RMSE
    """
    summary_rows = []
    for name, res in model_results.items():
        if 'test_metrics' in res and res['test_metrics'] is not None:
            test_metrics = res['test_metrics']
            summary_rows.append({
                'Model': name,
                'RMSE_macro': test_metrics.get('RMSE_macro', np.nan),
                'MAE_macro': test_metrics.get('MAE_macro', np.nan),
                'MAPE_macro': test_metrics.get('MAPE_macro', np.nan),
                'R2_macro': test_metrics.get('R2_macro', np.nan)
            })
    
    summary_df = pd.DataFrame(summary_rows).sort_values('RMSE_macro')
    return summary_df


def get_horizon_metrics(model_results: Dict, horizons: List[int]) -> pd.DataFrame:
    """
    Create DataFrame of per-horizon RMSE for all models.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary of model results
    horizons : list
        Forecast horizons
        
    Returns:
    --------
    horizon_df : DataFrame
        Per-horizon metrics table
    """
    horizon_metrics = []
    for name, res in model_results.items():
        if 'test_metrics' in res and res['test_metrics'] is not None:
            row = {'Model': name}
            for horizon in horizons:
                row[f'RMSE_h{horizon}'] = res['test_metrics'].get(f'RMSE_h{horizon}', np.nan)
            horizon_metrics.append(row)
    
    horizon_df = pd.DataFrame(horizon_metrics).set_index('Model')
    return horizon_df
