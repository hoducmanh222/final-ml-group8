"""Feature engineering utilities for weather forecasting."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def prepare_supervised_dataset(
    df: pd.DataFrame,
    horizons: List[int],
    lag_values: List[int],
    rolling_windows: List[int],
    sequence_length: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict]:
    """
    Create supervised learning dataset for multi-step forecasting.
    
    Parameters:
    -----------
    df : DataFrame
        Raw weather data
    horizons : list
        Forecast horizons (e.g., [1,2,3,4,5] for 5 days)
    lag_values : list
        Lag periods for features
    rolling_windows : list
        Window sizes for rolling statistics
    sequence_length : int
        Sequence length for time-series models
        
    Returns:
    --------
    feature_df : DataFrame
        Engineered features
    target_df : DataFrame
        Target variables
    target_cols : list
        Names of target columns
    sequence_package : dict
        Sequence data for LSTM/RNN models
    """
    df_work = df.copy()
    df_work['datetime'] = pd.to_datetime(df_work['datetime'])
    df_work = df_work.sort_values('datetime').reset_index(drop=True)
    df_work['row_id'] = df_work.index

    # Fill missing text columns
    text_columns = ['description', 'conditions', 'icon']
    for col in text_columns:
        if col in df_work.columns:
            df_work[col] = df_work[col].fillna('').astype(str)
    
    df_work['preciptype'] = df_work['preciptype'].fillna('none')
    df_work['severerisk'] = df_work['severerisk'].fillna(0.0)

    # Parse sunrise/sunset
    if 'sunrise' in df_work.columns and 'sunset' in df_work.columns:
        df_work['sunrise'] = pd.to_datetime(df_work['sunrise'])
        df_work['sunset'] = pd.to_datetime(df_work['sunset'])
        df_work['sunrise_minutes'] = df_work['sunrise'].dt.hour * 60 + df_work['sunrise'].dt.minute
        df_work['sunset_minutes'] = df_work['sunset'].dt.hour * 60 + df_work['sunset'].dt.minute
        df_work['day_length_minutes'] = (df_work['sunset'] - df_work['sunrise']).dt.total_seconds() / 60

    # Time features
    df_work['day_of_year'] = df_work['datetime'].dt.dayofyear
    df_work['month'] = df_work['datetime'].dt.month
    df_work['year'] = df_work['datetime'].dt.year
    df_work['day_of_week'] = df_work['datetime'].dt.dayofweek
    
    # Cyclical encoding
    df_work['sin_day_of_year'] = np.sin(2 * np.pi * df_work['day_of_year'] / 366)
    df_work['cos_day_of_year'] = np.cos(2 * np.pi * df_work['day_of_year'] / 366)
    df_work['sin_month'] = np.sin(2 * np.pi * df_work['month'] / 12)
    df_work['cos_month'] = np.cos(2 * np.pi * df_work['month'] / 12)

    # Temperature differences
    df_work['temp_diff_1'] = df_work['temp'].diff(1)
    df_work['temp_diff_7'] = df_work['temp'].diff(7)

    # Drop constant columns
    drop_constant = ['name', 'address', 'resolvedAddress', 'latitude', 'longitude', 'source']
    df_work = df_work.drop(columns=[col for col in drop_constant if col in df_work.columns])

    # Base numeric columns for lag/rolling features
    base_numeric_cols = ['temp', 'tempmax', 'tempmin', 'feelslikemax', 'feelslikemin', 'feelslike',
                         'humidity', 'dew', 'precip', 'precipprob', 'precipcover', 'windgust',
                         'windspeed', 'windspeedmax', 'windspeedmean', 'windspeedmin', 'winddir',
                         'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
                         'uvindex', 'severerisk', 'moonphase']
    base_numeric_cols = [col for col in base_numeric_cols if col in df_work.columns]

    base_numeric = df_work[['row_id', 'datetime'] + base_numeric_cols].copy().set_index('row_id')

    # Create lag features
    lag_frames = []
    for lag in lag_values:
        lag_df = df_work[base_numeric_cols].shift(lag)
        lag_df.columns = [f'{col}_lag_{lag}' for col in base_numeric_cols]
        lag_frames.append(lag_df)

    # Create rolling features
    roll_frames = []
    for window in rolling_windows:
        roll_df = pd.DataFrame(index=df_work.index)
        roll_df[f'temp_rollmean_{window}'] = df_work['temp'].rolling(window).mean()
        roll_df[f'temp_rollstd_{window}'] = df_work['temp'].rolling(window).std()
        roll_df[f'humidity_rollmean_{window}'] = df_work['humidity'].rolling(window).mean()
        roll_df[f'precip_rollsum_{window}'] = df_work['precip'].rolling(window).sum()
        roll_df[f'windspeed_rollmean_{window}'] = df_work['windspeed'].rolling(window).mean()
        roll_df[f'cloudcover_rollmean_{window}'] = df_work['cloudcover'].rolling(window).mean()
        roll_frames.append(roll_df)

    # Concatenate all features
    df_work = pd.concat([df_work] + lag_frames + roll_frames, axis=1)

    # Create target variables
    target_cols = []
    for horizon in horizons:
        col_name = f'target_temp_t+{horizon}'
        df_work[col_name] = df_work['temp'].shift(-horizon)
        target_cols.append(col_name)

    # Drop unnecessary columns
    if 'sunrise' in df_work.columns:
        df_work = df_work.drop(columns=['sunrise', 'sunset'])

    # Calculate valid range
    min_history = max(max(lag_values), sequence_length)
    valid_end_index = len(df_work) - max(horizons)
    
    if valid_end_index <= min_history:
        raise ValueError('Not enough rows for the requested configuration.')

    # Slice valid data
    df_model = df_work.iloc[min_history:valid_end_index].copy()

    # Prepare sequence data
    seq_samples, seq_targets, seq_row_ids = [], [], []
    for row_id in df_model['row_id']:
        start_idx = row_id - sequence_length + 1
        end_idx = row_id
        future_idx = [row_id + h for h in horizons]
        
        if start_idx < 0 or max(future_idx) >= len(base_numeric):
            continue
            
        seq_block = base_numeric.loc[start_idx:end_idx, base_numeric_cols].values
        target_block = base_numeric.loc[future_idx, 'temp'].values
        
        if seq_block.shape[0] == sequence_length:
            seq_samples.append(seq_block)
            seq_targets.append(target_block)
            seq_row_ids.append(row_id)

    if not seq_samples:
        raise ValueError('Sequence preparation failed. No sequences generated.')

    seq_samples = np.array(seq_samples)
    seq_targets = np.array(seq_targets)

    df_model = df_model.set_index('row_id').loc[seq_row_ids].reset_index()

    # Separate features and targets
    target_df = df_model[['row_id'] + target_cols].copy()
    feature_df = df_model.drop(columns=target_cols).copy()

    sequence_package = {
        'row_ids': seq_row_ids,
        'X_seq': seq_samples,
        'y_seq': seq_targets,
        'feature_names': base_numeric_cols
    }

    return feature_df, target_df, target_cols, sequence_package


import pandas as pd
import numpy as np
# Giả định: DataFrame df đã được load và có cột 'temp' và 'datetime'

# Hourly-specific configurations
DEFAULT_HOURLY_HORIZONS = [1, 2, 3, 6, 12, 18, 24, 48, 72, 96, 120]  
DEFAULT_HOURLY_LAGS = [1, 2, 3, 6, 12, 18, 24, 48, 96, 120, 144]  
DEFAULT_HOURLY_ROLLING = [3, 6, 12, 18, 24, 48, 72, 96, 120, 144]  

def prepare_features_hourly(df,
                            horizons=DEFAULT_HOURLY_HORIZONS,
                            lag_values=DEFAULT_HOURLY_LAGS,
                            rolling_windows=DEFAULT_HOURLY_ROLLING):
    """
    Prepare features for hourly weather forecasting.
    Similar to daily version but optimized for hourly data patterns.
    """
    print(f"Starting Feature Engineering: {len(df)} rows")
    print(f"Lags: {lag_values}, Rolling windows: {rolling_windows}, Horizons: {horizons}")
    
    df_work = df.copy()
    df_work['datetime'] = pd.to_datetime(df_work['datetime'])  
    df_work = df_work.sort_values('datetime').reset_index(drop=True)
    df_work['row_id'] = df_work.index

    # 1. Preprocessing
    text_columns = ['conditions', 'icon']  
    
    for col in text_columns:
        if col in df_work.columns:
            df_work[col] = df_work[col].fillna('').astype(str)
            
    df_work['preciptype'] = df_work['preciptype'].fillna('none')
    df_work['severerisk'] = df_work['severerisk'].fillna(0.0)

    # 2. Time Features
    df_work['day_of_year'] = df_work['datetime'].dt.dayofyear
    df_work['month'] = df_work['datetime'].dt.month
    df_work['year'] = df_work['datetime'].dt.year
    df_work['day_of_week'] = df_work['datetime'].dt.dayofweek
    df_work['is_weekend'] = (df_work['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for seasonal patterns
    df_work['sin_day_of_year'] = np.sin(2 * np.pi * df_work['day_of_year'] / 366)
    df_work['cos_day_of_year'] = np.cos(2 * np.pi * df_work['day_of_year'] / 366)
    df_work['sin_month'] = np.sin(2 * np.pi * df_work['month'] / 12)
    df_work['cos_month'] = np.cos(2 * np.pi * df_work['month'] / 12)

    # Ensure hour column exists
    if 'hour' not in df_work.columns:
        df_work['hour'] = df_work['datetime'].dt.hour
            
    # Cyclical encoding for 24-hour cycle (crucial for hourly data)
    df_work['sin_hour'] = np.sin(2 * np.pi * df_work['hour'] / 24)
    df_work['cos_hour'] = np.cos(2 * np.pi * df_work['hour'] / 24)
    
    # 4. Lag and Rolling Features
    base_numeric_cols = ['feelslike', 'humidity', 'dew', 'precip', 'precipprob',  
                         'precipcover', 'windgust', 'windspeed', 'winddir', 'sealevelpressure',  
                         'cloudcover', 'visibility', 'solarradiation', 'solarenergy',  
                         'uvindex', 'severerisk']
    base_numeric_cols = [col for col in base_numeric_cols if col in df_work.columns]
    
    # Create lag features
    lag_frames = []
    for lag in lag_values:
        lag_df = df_work[base_numeric_cols].shift(lag)
        lag_df.columns = [f'{col}_lag_{lag}' for col in base_numeric_cols]
        lag_frames.append(lag_df)
    
    # Add temperature lag features separately to avoid data leakage
    temp_lag_frames = []
    for lag in lag_values:
        temp_lag_df = pd.DataFrame(index=df_work.index)
        temp_lag_df[f'temp_lag_{lag}'] = df_work['temp'].shift(lag)
        temp_lag_frames.append(temp_lag_df)
    
    # Combine all lag frames
    all_lag_frames = lag_frames + temp_lag_frames

    # Create rolling features - SAFE VERSION (no data leakage)
    roll_frames = []
    # Correct implementation:
    for window in rolling_windows:
        roll_df = pd.DataFrame(index=df_work.index)
        # Apply shift FIRST, then rolling
        temp_shifted = df_work['temp'].shift(1)
        roll_df[f'temp_lag1_rollmean_{window}'] = temp_shifted.rolling(window, min_periods=1).mean()
        roll_df[f'temp_lag1_rollstd_{window}'] = temp_shifted.rolling(window, min_periods=1).std()
        # For other features, ensure they don't leak future info
        roll_df[f'humidity_rollmean_{window}'] = df_work['humidity'].shift(1).rolling(window, min_periods=1).mean()
        roll_df[f'precip_rollsum_{window}'] = df_work['precip'].shift(1).rolling(window, min_periods=1).sum()
        roll_df[f'windspeed_rollmean_{window}'] = df_work['windspeed'].shift(1).rolling(window, min_periods=1).mean()
        roll_frames.append(roll_df)

    # Concatenate all features
    df_work = pd.concat([df_work] + all_lag_frames + roll_frames, axis=1)

    # 5. Target Creation (Đã sửa đổi cho mục tiêu dự báo nhiệt độ trung bình ngày 5 ngày)
    target_cols = []
    
    # Tạo các target cho 5 ngày tiếp theo
    for day in range(1, 6):
        start_lag = -(24 * day)  # Bắt đầu từ t+1, t+25, t+49,...
        end_lag = -(24 * (day - 1) + 1) # Kết thúc ở t+24, t+48, t+72,...
        
        # Để dễ tính toán, tính Max/Min từ t+1 -> t+24, t+25 -> t+48, v.v.
        # Chúng ta cần sử dụng rolling window 24 giờ trên cột 'temp' đã được shift.
        
        # Shift cột 'temp' lên trên (shift âm) để các giá trị tương lai nằm cùng hàng.
        # Shift -1 sẽ đưa giá trị t+1 lên hàng t.
        # Để lấy cửa sổ [t+1, t+24], ta shift -24 và dùng rolling(window=24).
        
        # Tính Max và Min trong 24 giờ tiếp theo (window=24, dịch chuyển window 24h)
        # Ví dụ: Ngày 1 (t+1 đến t+24): shift(-24) rồi rolling(24)
        # shift_amount = 24 * day 
        # Cần phải dùng .rolling(24) trên một chuỗi đã được shift
        
        # Tạm thời tạo một cột shift để tính toán
        temp_shifted = df_work['temp'].shift(-1) # Giá trị t+1 nằm ở hàng t
        
        # Tạo cột chứa nhiệt độ của ngày hiện tại và 4 ngày sau đó, tổng cộng 5 ngày
        # Với day=1: t+1 đến t+24
        # Với day=2: t+25 đến t+48
        
        # Cắt chuỗi 'temp' và shift lên để window rolling có thể tính toán
        # Chuỗi temp_future[i] chứa temp[i + 1]
        
        # Max/Min trong cửa sổ 24 giờ:
        # Day 1: [t+1, t+24] -> Shift (-1) và Rolling(24)
        # Day 2: [t+25, t+48] -> Shift (-25) và Rolling(24)
        
        # Tính Max và Min cho 24 giờ tiếp theo (t+1 đến t+24)
        # Cột temp ở hàng i là temp tại thời điểm t.
        # temp.shift(-h) ở hàng i là temp tại thời điểm t+h.
        
        # Shift để giá trị cuối cùng của cửa sổ (t+24, t+48,...) nằm ở hàng hiện tại (i)
        
        # Cửa sổ [t+h_start, t+h_end]
        h_end = 24 * day
        h_start = 24 * (day - 1) + 1
        window = h_end - h_start + 1 # Luôn là 24
        
        # Ta cần cửa sổ 24 giờ BẮT ĐẦU từ t+h_start.
        # rolling(window).max().shift(-window + 1)
        # Ví dụ: [t+1, t+24]. rolling(24).max() ở i là max[i-23:i+1]. Ta muốn max[i+1:i+24]
        
        # Tạo chuỗi mới, shift lên, sau đó dùng rolling
        # Lấy cột 'temp' (temp_t). shift(-h_start) sẽ đưa temp_{t+h_start} lên hàng t
        # Cần một chuỗi mà rolling 24 sẽ tính trên [temp_{t+h_start}, ..., temp_{t+h_end}]
        
        # Tạo cửa sổ [t+1, t+24]
        temp_window_start = df_work['temp'].shift(-h_start)
        # Rolling(24) trên chuỗi này sẽ tính [temp_{t+h_start}, ..., temp_{t+h_start + 23}]
        # Tức là [temp_{t+1}, ..., temp_{t+24}] cho day 1.
        
        roll_max = temp_window_start.rolling(window=window).max()
        roll_min = temp_window_start.rolling(window=window).min()
        
        # Sau khi tính, ta cần shift kết quả rolling ngược lại.
        # .rolling().max() là một hàm window *right-aligned* (tính đến cuối window)
        # Giá trị ở hàng i của roll_max/min là Max/Min của window kết thúc ở hàng i.
        # Ta cần Max/Min của window [i+h_start, i+h_end] nằm ở hàng i.
        # Giá trị Max/Min của [t+1, t+24] nằm ở hàng có index là (t+24)-1.
        # Ta cần shift nó ngược lại (lên trên) 24*(day-1) + 1.
        
        # Shift lên (âm) để kết quả [t+h_start, t+h_end] nằm ở hàng t
        target_max_col = f'target_daily_max_day_{day}'
        target_min_col = f'target_daily_min_day_{day}'
        target_avg_col = f'target_avg_day_{day}'
        
        # Dùng shift để tạo cửa sổ: temp.shift(-h_start) đưa temp_{t+h_start} lên hàng t
        # rolling(window) tính max/min của cửa sổ [i, i+window-1] trên chuỗi shift này
        # Sau đó shift ngược lại: max/min của [t+h_start, t+h_end] ở hàng t-h_start.
        # Ta cần shift nó lên trên (shift âm) h_start-1.
        
        # Cách đơn giản nhất:
        df_work[target_max_col] = df_work['temp'].rolling(window=window).max().shift(-h_end)
        df_work[target_min_col] = df_work['temp'].rolling(window=window).min().shift(-h_end)

        # Tính trung bình (Max + Min) / 2
        df_work[target_avg_col] = (df_work[target_max_col] + df_work[target_min_col]) / 2.0
        
        target_cols.append(target_avg_col)
        # Bỏ 2 cột max/min nếu chỉ muốn target cuối cùng là trung bình
        # target_cols.extend([target_max_col, target_min_col])

    # 6. Cleanup and Slicing (Sử dụng max_horizon mới)
    cols_to_drop_final = ['sunrise', 'sunset', 'name', 'address', 'resolvedAddress',  
                          'latitude', 'longitude', 'source']
    cols_present_to_drop = [col for col in cols_to_drop_final if col in df_work.columns]
    if cols_present_to_drop:
        print(f"    - Dropping unnecessary columns: {cols_present_to_drop}")
        df_work = df_work.drop(columns=cols_present_to_drop)
    
    # Tính minimum history cần thiết
    min_history = max(
        max(lag_values, default=0),
        max(rolling_windows, default=0)
    )
    
    # Max horizon mới (ngày 5: t+120)
    max_forecast_horizon = 24 * 5 # 120 giờ
    
    # Calculate valid end index
    # Hàng cuối cùng có target (avg_day_5) là hàng len(df_work) - max_forecast_horizon - 1
    valid_end_index = len(df_work) - max_forecast_horizon
    
    if valid_end_index <= min_history:
        raise ValueError('Not enough data for this configuration. Reduce lags/rolling/horizons.')

    # Slice data to remove NaN rows
    df_model = df_work.iloc[min_history:valid_end_index].copy()
    
    print(f"Final data size for modeling: {len(df_model)} rows")

    # 7. Return Results
    # Cần trả về tất cả các cột target đã tạo (target_avg_day_1 đến target_avg_day_5)
    all_target_cols = [f'target_avg_day_{day}' for day in range(1, 6)]
    # Thêm cả các cột max/min vào df target để tham khảo
    
    # Lọc các cột target
    temp_target_cols = []
    for day in range(1, 6):
        temp_target_cols.extend([f'target_daily_max_day_{day}', f'target_daily_min_day_{day}', f'target_avg_day_{day}'])
        
    target_df = df_model[['row_id'] + [col for col in temp_target_cols if col in df_model.columns]].copy()
    feature_df = df_model.drop(columns=[col for col in temp_target_cols if col in df_model.columns] + ['row_id']).copy()
    
    # ⚠️ Loại bỏ các cột leakage
    to_remove_columns = ['temp']  # Current temperature
    
    columns_to_remove = [col for col in to_remove_columns if col in feature_df.columns]
    if columns_to_remove:
        feature_df = feature_df.drop(columns=columns_to_remove)
        print(f"    - Removed leakage columns: {columns_to_remove}")
    
    # Remove any columns that might contain current temperature info
    suspicious_columns = [col for col in feature_df.columns if 'temp' in col and 'lag' not in col and 'roll' not in col]
    if suspicious_columns:
        feature_df = feature_df.drop(columns=suspicious_columns)
        print(f"    - Removed suspicious temperature columns: {suspicious_columns}")
    
    print(f"    - Feature columns: {len(feature_df.columns)}")
    print(f"    - Target columns (Avg only): {len(all_target_cols)}")

    return feature_df, target_df, all_target_cols