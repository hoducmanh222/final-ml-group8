import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from model_training import compute_metrics
from config import DAILY_HORIZONS

def analyze_model_comprehensive(model_name, model_results, X_train, X_test, y_train, y_test,
                                test_dates, horizons=DAILY_HORIZONS):
    """
    Comprehensive model analysis following the 6-part framework.

    Parameters:
    -----------
    model_name : str
        Name of the model to analyze
    model_results : dict
        Dictionary containing all model results
    X_train, X_test : DataFrame
        Training and test features
    y_train, y_test : DataFrame
        Training and test targets
    test_dates : Series
        Dates corresponding to test set
    horizons : list
        Forecast horizons
    """

    if model_name not in model_results:
        print(f"Model '{model_name}' not found in results.")
        return

    result = model_results[model_name]

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS: {model_name}")
    print(f"{'='*80}\n")

    # Section 2: Model Specification
    print("## 2. MODEL SPECIFICATION")
    print("=" * 60)
    print_model_specification(model_name, result)

    # Section 3: Quantitative Performance
    print("\n## 3. QUANTITATIVE PERFORMANCE")
    print("=" * 60)
    analyze_quantitative_performance(model_name, result, horizons)

    # Section 4: Error Analysis
    print("\n## 4. ERROR ANALYSIS")
    print("=" * 60)
    perform_error_analysis(model_name, result, y_test, test_dates, horizons)

    # Section 5: Bias-Variance Decomposition
    print("\n## 5. BIAS-VARIANCE DECOMPOSITION")
    print("=" * 60)
    perform_bias_variance_decomposition(model_name, result, y_train, y_test, horizons)


def print_model_specification(model_name, result):
    """Print model technical configuration."""
    print(f"Model: {model_name}")
    print(f"Details: {result.get('details', 'N/A')}")

    # Extract hyperparameters if available
    if 'pipeline' in result:
        pipeline = result['pipeline']
        try:
            estimator = pipeline.named_steps['regressor']
            if hasattr(estimator, 'estimators_'):
                base_estimator = estimator.estimators_[0]
                print(f"\nHyperparameters:")
                for param, value in base_estimator.get_params().items():
                    if not param.startswith('_'):
                        print(f"  - {param}: {value}")
        except:
            pass


def compare_optuna_performance(base_model_name, optuna_model_name, model_results, horizons=DAILY_HORIZONS):
    """
    Compare performance with and without Optuna tuning.
    """
    print("\n" + "="*80)
    print("OPTUNA HYPERPARAMETER TUNING COMPARISON")
    print("="*80)

    if base_model_name not in model_results or optuna_model_name not in model_results:
        print("Required models not found for comparison.")
        return

    base_metrics = model_results[base_model_name]['test_metrics']
    optuna_metrics = model_results[optuna_model_name]['test_metrics']

    comparison_df = pd.DataFrame({
        'Metric': ['RMSE_macro', 'MAE_macro', 'MAPE_macro', 'R2_macro'],
        'Base Model': [base_metrics['RMSE_macro'], base_metrics['MAE_macro'],
                       base_metrics['MAPE_macro'], base_metrics['R2_macro']],
        'Optuna Tuned': [optuna_metrics['RMSE_macro'], optuna_metrics['MAE_macro'],
                         optuna_metrics['MAPE_macro'], optuna_metrics['R2_macro']]
    })

    comparison_df['Improvement (%)'] = ((comparison_df['Base Model'] - comparison_df['Optuna Tuned'])
                                        / comparison_df['Base Model'] * 100)

    display(comparison_df)

    # Per-horizon comparison
    horizon_comparison = []
    for h in horizons:
        horizon_comparison.append({
            'Horizon': f't+{h}',
            'Base RMSE': base_metrics[f'RMSE_h{h}'],
            'Optuna RMSE': optuna_metrics[f'RMSE_h{h}'],
            'Improvement (%)': ((base_metrics[f'RMSE_h{h}'] - optuna_metrics[f'RMSE_h{h}'])
                                / base_metrics[f'RMSE_h{h}'] * 100)
        })

    horizon_df = pd.DataFrame(horizon_comparison)
    print("\nPer-Horizon RMSE Comparison:")
    display(horizon_df)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Macro metrics comparison
    metrics_to_plot = ['RMSE_macro', 'MAE_macro', 'R2_macro']
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    base_vals = [base_metrics[m] for m in metrics_to_plot]
    optuna_vals = [optuna_metrics[m] for m in metrics_to_plot]

    axes[0].bar(x - width/2, base_vals, width, label='Base Model', alpha=0.8)
    axes[0].bar(x + width/2, optuna_vals, width, label='Optuna Tuned', alpha=0.8)
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Macro Metrics Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_to_plot)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-horizon RMSE
    axes[1].plot(horizons, [base_metrics[f'RMSE_h{h}'] for h in horizons],
                 marker='o', label='Base Model', linewidth=2)
    axes[1].plot(horizons, [optuna_metrics[f'RMSE_h{h}'] for h in horizons],
                 marker='s', label='Optuna Tuned', linewidth=2)
    axes[1].set_xlabel('Forecast Horizon (days)')
    axes[1].set_ylabel('RMSE (°C)')
    axes[1].set_title('RMSE by Forecast Horizon')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Decision recommendation
    avg_improvement = comparison_df['Improvement (%)'].mean()
    print(f"\n{'='*80}")
    print("RECOMMENDATION:")
    if avg_improvement > 2:
        print(f"✓ Use Optuna tuning (Average improvement: {avg_improvement:.2f}%)")
    elif avg_improvement > 0:
        print(f"~ Consider Optuna tuning (Marginal improvement: {avg_improvement:.2f}%)")
    else:
        print(f"✗ Base model sufficient (No significant improvement: {avg_improvement:.2f}%)")
    print(f"{'='*80}\n")


def analyze_quantitative_performance(model_name, result, horizons):
    """Analyze quantitative performance metrics."""

    # 3a. Performance Metrics
    print("\n### 3a. Performance Metrics")
    print("-" * 60)

    test_metrics = result['test_metrics']
    train_metrics = result.get('train_metrics', {})
    cv_metrics = result.get('cv_metrics', {})


    print("\nMacro Metrics (averaged across all horizons):")
    for metric in ['RMSE_macro', 'MAE_macro', 'MAPE_macro', 'R2_macro']:
        metric_base = metric.replace('_macro', '')
        print(f"\n{metric_base}:")
        print(f"  Value: {test_metrics[metric]:.4f}")

    # Per-horizon metrics
    print("\n\nPer-Horizon Metrics:")
    horizon_metrics_df = pd.DataFrame([
        {
            'Horizon': f't+{h}',
            'RMSE': test_metrics[f'RMSE_h{h}'],
            'MAE': test_metrics[f'MAE_h{h}'],
            'MAPE': test_metrics[f'MAPE_h{h}'],
            'R²': test_metrics[f'R2_h{h}']
        }
        for h in horizons
    ])
    display(horizon_metrics_df)

    # 3b. Train-Test Comparison
    print("\n### 3b. Train-Test Comparison")
    print("-" * 60)

    if train_metrics:
        comparison_df = pd.DataFrame({
            'Metric': ['RMSE_macro', 'MAE_macro', 'R2_macro'],
            'Train': [train_metrics.get('RMSE_macro', np.nan),
                     train_metrics.get('MAE_macro', np.nan),
                     train_metrics.get('R2_macro', np.nan)],
            'Test': [test_metrics['RMSE_macro'],
                    test_metrics['MAE_macro'],
                    test_metrics['R2_macro']]
        })
        comparison_df['Difference'] = comparison_df['Test'] - comparison_df['Train']
        comparison_df['Overfitting?'] = comparison_df['Difference'].apply(
            lambda x: 'Yes' if abs(x) > 0.5 else 'No'
        )
        display(comparison_df)

        # Interpretation
        rmse_diff = test_metrics['RMSE_macro'] - train_metrics.get('RMSE_macro', 0)
        if rmse_diff > 0.5:
            print("\n⚠ Potential overfitting detected (large train-test gap)")
        elif rmse_diff < -0.2:
            print("\n⚠ Unusual behavior (test performs better than train)")
        else:
            print("\n✓ Good generalization (similar train-test performance)")
    else:
        print("Train metrics not available.")

    # 3c. Cross-Validation Metrics
    print("\n### 3c. Cross-Validation Performance")
    print("-" * 60)

    if cv_metrics:
        print(f"CV RMSE (mean): {cv_metrics['RMSE_macro']:.4f}")
        print(f"CV MAE (mean): {cv_metrics['MAE_macro']:.4f}")
        print(f"CV R² (mean): {cv_metrics['R2_macro']:.4f}")
        print("\n[Note: Standard deviation would require storing individual fold results]")
    else:
        print("Cross-validation metrics not available.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-horizon performance
    axes[0].plot(horizons, [test_metrics[f'RMSE_h{h}'] for h in horizons],
                marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Forecast Horizon (days)', fontsize=11)
    axes[0].set_ylabel('RMSE (°C)', fontsize=11)
    axes[0].set_title(f'{model_name}: RMSE by Horizon', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Train vs Test comparison (if available)
    if train_metrics:
        metrics_names = ['RMSE', 'MAE', 'R²']
        train_vals = [train_metrics.get('RMSE_macro', 0),
                     train_metrics.get('MAE_macro', 0),
                     train_metrics.get('R2_macro', 0)]
        test_vals = [test_metrics['RMSE_macro'],
                    test_metrics['MAE_macro'],
                    test_metrics['R2_macro']]

        x = np.arange(len(metrics_names))
        width = 0.35

        axes[1].bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
        axes[1].bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
        axes[1].set_xlabel('Metrics', fontsize=11)
        axes[1].set_ylabel('Value', fontsize=11)
        axes[1].set_title('Train vs Test Performance', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metrics_names)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def perform_error_analysis(model_name, result, y_test, test_dates, horizons):
    """Perform comprehensive error analysis."""

    predictions = result['predictions_test']

    # 4a. Residual Plot (Time Series)
    print("\n### 4a. Residual Analysis Over Time")
    print("-" * 60)

    residuals = y_test.values - predictions

    # Calculate seasonal statistics
    test_dates_series = pd.to_datetime(test_dates)
    months = test_dates_series.dt.month

    seasonal_errors = pd.DataFrame({
        'Month': months,
        'Residual_h1': residuals[:, 0]
    })
    monthly_stats = seasonal_errors.groupby('Month')['Residual_h1'].agg(['mean', 'std', 'count'])

    print("\nMonthly Residual Statistics (Horizon t+1):")
    display(monthly_stats)

    # Identify problematic periods
    abs_residuals = np.abs(residuals)
    error_threshold = np.percentile(abs_residuals[:, 0], 90)
    high_error_periods = test_dates_series[abs_residuals[:, 0] > error_threshold]

    print(f"\nHigh Error Periods (top 10% errors > {error_threshold:.2f}°C):")
    print(f"Total occurrences: {len(high_error_periods)}")
    if len(high_error_periods) > 0:
        print(f"Date range: {high_error_periods.min()} to {high_error_periods.max()}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Time series residual plot (horizon 1)
    axes[0, 0].scatter(test_dates, residuals[:, 0], alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Date', fontsize=10)
    axes[0, 0].set_ylabel('Residual (°C)', fontsize=10)
    axes[0, 0].set_title(f'{model_name}: Residuals Over Time (t+1)', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals by month (boxplot)
    monthly_residuals = [residuals[months == m, 0] for m in range(1, 13)]
    axes[0, 1].boxplot(monthly_residuals, labels=range(1, 13))
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Month', fontsize=10)
    axes[0, 1].set_ylabel('Residual (°C)', fontsize=10)
    axes[0, 1].set_title('Seasonal Error Pattern (t+1)', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Residual by horizon
    for i, h in enumerate(horizons):
        axes[1, 0].scatter([h] * len(residuals), residuals[:, i], alpha=0.3, s=10, label=f't+{h}')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Forecast Horizon', fontsize=10)
    axes[1, 0].set_ylabel('Residual (°C)', fontsize=10)
    axes[1, 0].set_title('Residuals by Forecast Horizon', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Absolute error by horizon
    mean_abs_errors = [np.mean(np.abs(residuals[:, i])) for i in range(len(horizons))]
    axes[1, 1].bar(horizons, mean_abs_errors, alpha=0.7, color='coral')
    axes[1, 1].set_xlabel('Forecast Horizon (days)', fontsize=10)
    axes[1, 1].set_ylabel('Mean Absolute Error (°C)', fontsize=10)
    axes[1, 1].set_title('Average Error by Horizon', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # 4b. Distribution of Residuals
    print("\n### 4b. Residual Distribution Analysis")
    print("-" * 60)

    fig, axes = plt.subplots(1, len(horizons), figsize=(4*len(horizons), 4))
    if len(horizons) == 1:
        axes = [axes]

    for i, h in enumerate(horizons):
        res_h = residuals[:, i]

        # Statistics
        mean_res = np.mean(res_h)
        std_res = np.std(res_h)
        skewness = pd.Series(res_h).skew()
        kurtosis = pd.Series(res_h).kurtosis()

        # Histogram with normal curve
        axes[i].hist(res_h, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

        # Overlay normal distribution
        xmin, xmax = axes[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = np.exp(-0.5 * ((x - mean_res) / std_res) ** 2) / (std_res * np.sqrt(2 * np.pi))
        axes[i].plot(x, p, 'r-', linewidth=2, label='Normal')

        axes[i].axvline(mean_res, color='green', linestyle='--', linewidth=2, label=f'Mean={mean_res:.3f}')
        axes[i].set_xlabel('Residual (°C)', fontsize=10)
        axes[i].set_ylabel('Density', fontsize=10)
        axes[i].set_title(f'Horizon t+{h}\nSkew={skewness:.2f}, Kurt={kurtosis:.2f}',
                         fontsize=10, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

        print(f"\nHorizon t+{h}:")
        print(f"  Mean: {mean_res:.4f}°C")
        print(f"  Std Dev: {std_res:.4f}°C")
        print(f"  Skewness: {skewness:.4f} {'(right-skewed)' if skewness > 0 else '(left-skewed)'}")
        print(f"  Kurtosis: {kurtosis:.4f} {'(heavy-tailed)' if kurtosis > 0 else '(light-tailed)'}")

        # Bias assessment
        if abs(mean_res) < 0.1:
            print(f"  ✓ Low bias (|mean| < 0.1°C)")
        elif abs(mean_res) < 0.5:
            print(f"  ~ Moderate bias")
        else:
            print(f"  ✗ High bias (|mean| > 0.5°C)")

    plt.tight_layout()
    plt.show()


def perform_bias_variance_decomposition(model_name, result, y_train, y_test, horizons):
    """Perform bias-variance decomposition analysis."""

    print("\n### Bias-Variance Decomposition")
    print("-" * 60)

    predictions_test = result['predictions_test']
    predictions_train = result.get('predictions_train', None)

    decomposition_results = []

    for i, h in enumerate(horizons):
        y_true = y_test.values[:, i]
        y_pred = predictions_test[:, i]

        # Overall MSE
        mse = np.mean((y_true - y_pred) ** 2)

        # Bias² (squared difference between mean prediction and mean truth)
        bias_squared = (np.mean(y_pred) - np.mean(y_true)) ** 2

        # Variance (variance of predictions)
        variance = np.var(y_pred)

        # Irreducible error (variance of true values)
        irreducible_error = np.var(y_true)

        decomposition_results.append({
            'Horizon': f't+{h}',
            'Total MSE': mse,
            'Bias²': bias_squared,
            'Variance': variance,
            'Irreducible Error': irreducible_error,
            'Bias² %': (bias_squared / mse * 100) if mse > 0 else 0,
            'Variance %': (variance / mse * 100) if mse > 0 else 0
        })

    decomp_df = pd.DataFrame(decomposition_results)
    display(decomp_df)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar chart
    bias_vals = decomp_df['Bias²'].values
    var_vals = decomp_df['Variance'].values

    axes[0].bar(decomp_df['Horizon'], bias_vals, label='Bias²', alpha=0.8)
    axes[0].bar(decomp_df['Horizon'], var_vals, bottom=bias_vals, label='Variance', alpha=0.8)
    axes[0].set_xlabel('Forecast Horizon', fontsize=11)
    axes[0].set_ylabel('Error Component', fontsize=11)
    axes[0].set_title('Bias-Variance Decomposition', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Percentage contribution
    axes[1].bar(decomp_df['Horizon'], decomp_df['Bias² %'], label='Bias² %', alpha=0.8)
    axes[1].bar(decomp_df['Horizon'], decomp_df['Variance %'],
               bottom=decomp_df['Bias² %'], label='Variance %', alpha=0.8)
    axes[1].set_xlabel('Forecast Horizon', fontsize=11)
    axes[1].set_ylabel('Percentage Contribution (%)', fontsize=11)
    axes[1].set_title('Relative Error Contribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Interpretation
    print("\n### Interpretation:")
    avg_bias_pct = decomp_df['Bias² %'].mean()
    avg_var_pct = decomp_df['Variance %'].mean()

    if avg_bias_pct > 60:
        print("⚠ High bias detected - model is underfitting")
        print("  Recommendation: Increase model complexity or add more features")
    elif avg_var_pct > 60:
        print("⚠ High variance detected - model is overfitting")
        print("  Recommendation: Regularization, reduce complexity, or gather more data")
    else:
        print("✓ Balanced bias-variance trade-off")
        print(f"  Bias contribution: {avg_bias_pct:.1f}%")
        print(f"  Variance contribution: {avg_var_pct:.1f}%")