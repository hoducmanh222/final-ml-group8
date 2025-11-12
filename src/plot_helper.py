import matplotlib.pyplot as plt
import  numpy as np
import pandas as pd
from IPython.display import display

def base_model_plot(model_result, model_name, y_test, test_dates):
    preds = model_result[model_name]['predictions_test']
    residuals = y_test.values - preds

    # Top 10% lỗi lớn
    error_threshold = np.percentile(np.abs(residuals[:, 0]), 90)
    high_error_mask = np.abs(residuals[:, 0]) > error_threshold

    plt.figure(figsize=(14, 6))

    # --- Actual vs Predicted (màu nhẹ, line mảnh) ---
    plt.plot(
        test_dates, y_test.values[:, 0],
        color='#2a9d8f', label='Actual', linewidth=1.2, alpha=0.9
    )
    plt.plot(
        test_dates, preds[:, 0],
        color='#e76f51', label='Predicted', linewidth=1.2, alpha=0.9
    )

    # --- Scatter: Top 10% errors (nổi bật) ---
    plt.scatter(
        test_dates[high_error_mask], preds[:, 0][high_error_mask],
        color='#d62828', s=35, label='Top 10% errors',
        zorder=5, alpha=0.8, edgecolors='white', linewidth=0.6
    )


    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title(f'{model_name}: Predictions vs Actual (Horizon t+1)',
            fontsize=14, fontweight='bold', pad=10)

    plt.legend(frameon=False, fontsize=10)
    plt.grid(True, alpha=0.15, linestyle='--')
    plt.gca().set_facecolor('white')
    plt.tight_layout()


    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tick_params(axis='both', which='both', length=0) 


    for spine in plt.gca().spines.values():
        spine.set_alpha(0.3)

    plt.show()

def final_summary(model_results, y_test, test_dates, HORIZONS):
    summary_rows = []
    for name, res in model_results.items():
        test_metrics = res['test_metrics']
        summary_rows.append({
            'Model': name,
            'RMSE_macro': test_metrics['RMSE_macro'],
            'MAE_macro': test_metrics['MAE_macro'],
            'MAPE_macro': test_metrics['MAPE_macro'],
            'R2_macro': test_metrics['R2_macro']
        })
    summary_df = pd.DataFrame(summary_rows).sort_values('RMSE_macro')
    display(summary_df)

    horizon_metrics = []
    for name, res in model_results.items():
        row = {'Model': name}
        for horizon in HORIZONS:
            row[f'RMSE_h{horizon}'] = res['test_metrics'][f'RMSE_h{horizon}']
        horizon_metrics.append(row)
    horizon_df = pd.DataFrame(horizon_metrics).set_index('Model').loc[summary_df['Model']]
    display(horizon_df)

    best_model_name = summary_df.iloc[0]['Model']
    best_predictions = model_results[best_model_name]['predictions_test']
    print(f'Best model on test: {best_model_name}')

    slice_idx = max(0, len(test_dates) - 365)
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_horizon = 1
    ax.plot(test_dates.iloc[slice_idx:], y_test[f'target_temp_t+{plot_horizon}'].iloc[slice_idx:], label='Actual', alpha=0.7)
    ax.plot(test_dates.iloc[slice_idx:], best_predictions[slice_idx:, plot_horizon - 1], label=f'{best_model_name} prediction', alpha=0.9)
    ax.set_title(f'Horizon +{plot_horizon} day forecast comparison (last year)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Date')
    ax.legend()
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_horizon = 5
    ax.plot(test_dates.iloc[slice_idx:], y_test[f'target_temp_t+{plot_horizon}'].iloc[slice_idx:], label='Actual', alpha=0.7)
    ax.plot(test_dates.iloc[slice_idx:], best_predictions[slice_idx:, plot_horizon - 1], label=f'{best_model_name} prediction', alpha=0.9)
    ax.set_title(f'Horizon +{plot_horizon} day forecast comparison (last year)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Date')
    ax.legend()
    plt.tight_layout()

    if 'Seq2Seq_LSTM' in model_results:
        history_df = pd.DataFrame(model_results['Seq2Seq_LSTM']['training_history'])
        display(history_df.tail())