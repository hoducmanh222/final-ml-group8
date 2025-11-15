"""
Vietnam Weather Forecasting App
================================
A production-ready Streamlit application for 5-day weather temperature forecasting.
Deployed on Hugging Face Spaces.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
from pathlib import Path
import warnings
import requests
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.config import *
    from src.preprocessing import identify_column_types, make_pipeline
    from src.model_training import compute_metrics
    from src.feature_engineering import prepare_supervised_dataset  # Import feature engineering
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure you're running from the project root directory")
    st.stop()

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Vietnam Weather Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-in;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .forecast-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .forecast-box:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    .weather-icon {
        font-size: 3rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .api-badge {
        display: inline-block;
        background: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained model from pickle file."""
    try:
        model_path = Path(__file__).parent.parent / "models" / "final_model.pkl"
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            st.info("Please train the model first: python scripts/train_models.py")
            return None
        
        # Use joblib for better cross-version compatibility
        import joblib
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================================
# API CONFIGURATION
# ============================================================
# Multiple API keys for rotation to prevent rate limiting
API_KEYS = [
    "H2VGPX5BLW8B7CM3KYWKFSSAV",
    "LHXYADC5P5YBQU6N78MB77J6N",
    "NKS2UDLBW3LZEC2UME4YR6FS9",
    "SKG826LA8K5SX9BERSB2MD5QL"
]
API_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
LOCATION = "ho chi minh"

# API key rotation index
if 'api_key_index' not in st.session_state:
    st.session_state.api_key_index = 0

def get_next_api_key():
    """Rotate to next API key."""
    st.session_state.api_key_index = (st.session_state.api_key_index + 1) % len(API_KEYS)
    return API_KEYS[st.session_state.api_key_index]

# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_api_weather_data(days_back: int = 90) -> Optional[pd.DataFrame]:
    """Fetch latest weather data from Visual Crossing API.
    
    Parameters:
    -----------
    days_back : int
        Number of days of historical data to fetch
    
    Returns:
    --------
    DataFrame with weather data or None if error
    """
    try:
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Build API URL
        url = f"{API_BASE_URL}/{LOCATION}/{start_date}/{end_date}"
        
        # Try API keys until one works
        last_error = None
        for attempt in range(len(API_KEYS)):
            current_key = API_KEYS[st.session_state.api_key_index]
            params = {
                'unitGroup': 'metric',
                'key': current_key,
                'contentType': 'json',
                'include': 'days'
            }
            
            try:
                # Make API request
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                break  # Success, exit loop
            except requests.exceptions.RequestException as e:
                last_error = e
                get_next_api_key()  # Try next key
                if attempt == len(API_KEYS) - 1:
                    raise last_error  # All keys failed
        
        data = response.json()
        
        # Extract daily data
        if 'days' not in data:
            st.error("No daily data found in API response")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data['days'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Fix list columns - convert to strings or take first element
        for col in df.columns:
            if df[col].dtype == object:
                # Check if column contains lists
                sample = df[col].iloc[0] if len(df) > 0 else None
                if isinstance(sample, list):
                    # Convert list to comma-separated string or take first element
                    if col == 'preciptype':
                        df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'none')
                    else:
                        df[col] = df[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x))
        
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing API data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_weather_data(use_api: bool = True):
    """Fetch weather data from API or local file.
    
    Parameters:
    -----------
    use_api : bool
        If True, fetch from API. Otherwise, use local file.
    """
    if use_api:
        df = fetch_api_weather_data(days_back=120)  # Fetch 4 months of data
        if df is not None:
            return df
        st.warning("Falling back to local data file...")
    
    # Fallback to local file
    try:
        if not Path(DATA_PATH_DAILY).exists():
            st.error(f"Data file not found: {DATA_PATH_DAILY}")
            return None
            
        df = pd.read_csv(DATA_PATH_DAILY)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================================
# FEATURE ENGINEERING FOR PREDICTION
# ============================================================
def prepare_features_for_prediction(df, model_config):
    """
    Prepare features from recent data for prediction using the same 
    feature engineering as training.
    
    Parameters:
    -----------
    df : DataFrame
        Historical weather data
    model_config : dict
        Model configuration with horizons, lag_values, etc.
        
    Returns:
    --------
    features : DataFrame
        Engineered features ready for prediction
    """
    try:
        # Make a copy to avoid modifying cached data
        df = df.copy()
        
        # Fix all list columns before processing
        for col in df.columns:
            if df[col].dtype == object:
                # Convert any lists to strings
                df[col] = df[col].apply(lambda x: 
                    x[0] if isinstance(x, list) and len(x) > 0 
                    else ('none' if isinstance(x, list) else x)
                )
        
        # Add missing columns that might not be in API response
        missing_cols = ['windspeedmax', 'windspeedmean', 'windspeedmin', 'sealevelpressure']
        for col in missing_cols:
            if col not in df.columns:
                if 'windspeed' in col:
                    # Derive from windspeed if available
                    if 'windspeed' in df.columns:
                        df[col] = df['windspeed']
                elif col == 'sealevelpressure':
                    # Use default pressure if not available
                    df[col] = 1013.25  # Standard atmospheric pressure
        
        # Use the same feature engineering as training
        feature_df, target_df, target_cols, _ = prepare_supervised_dataset(
            df,
            horizons=model_config['horizons'],
            lag_values=model_config['lag_values'],
            rolling_windows=model_config['rolling_windows'],
            sequence_length=model_config['sequence_length']
        )
        
        # Take the last row (most recent)
        features = feature_df.drop(columns=['datetime']).iloc[[-1]]
        
        # Final check: ensure no lists remain in features
        for col in features.columns:
            if features[col].dtype == object:
                features[col] = features[col].apply(lambda x: 
                    str(x) if not isinstance(x, (int, float)) else x
                )
        
        return features
        
    except Exception as e:
        st.error(f"Error in feature preparation: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown('<div class="main-header">🌤️ Vietnam Weather Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered 5-Day Temperature Prediction for Ho Chi Minh City <span class="api-badge">🔴 LIVE DATA</span></div>', unsafe_allow_html=True)
    
    # Sidebar with tabs
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/hoducmanh222/temp_holder/main/vietnam_flag.png", 
                width=100)
        st.title("⚙️ Settings")
        
        # Create tabs in sidebar
        tab1, tab2, tab3, tab4 = st.tabs(["ℹ️ Info", "⚙️ Data", "📊 Display", "📐 Charts"])
        
        with tab1:
            st.markdown("### 📊 Model Information")
            st.info("""
            **Model:** Ensemble (BayesianRidge + HGB)  
            **Forecast Horizon:** 5 days  
            **Training Data:** 2015-2025  
            **Accuracy:** RMSE < 0.8°C  
            **Data Source:** Visual Crossing API 🔴 LIVE
            """)
            
            st.markdown("### 📍 Location")
            st.write("**City:** Ho Chi Minh City, Vietnam")
            st.write("**Coordinates:** 10.8231° N, 106.6297° E")
            
            st.markdown("### 🔑 API Status")
            st.write(f"**Active Key:** #{st.session_state.api_key_index + 1} of {len(API_KEYS)}")
        
        with tab2:
            st.markdown("### 🌐 Data Source")
            use_api = st.checkbox("Use Live API Data", value=True)
            st.caption("⚡ Real-time data from Visual Crossing")
            
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
            
            if st.button("🔄 Rotate API Key"):
                get_next_api_key()
                st.cache_data.clear()
                st.success(f"Switched to API Key #{st.session_state.api_key_index + 1}")
                st.rerun()
        
        with tab3:
            st.markdown("### 📊 Display Options")
            show_forecast = st.checkbox("Show 5-Day Forecast", value=True)
            show_historical = st.checkbox("Show Historical Data", value=True)
            show_pred_vs_actual = st.checkbox("Show Predicted vs Actual", value=True)
            show_metrics = st.checkbox("Show Model Metrics", value=True)
        
        with tab4:
            st.markdown("### 📐 Chart Settings")
            chart_width = st.slider("Chart Width", min_value=8, max_value=16, value=12, step=1)
            chart_height = st.slider("Chart Height", min_value=4, max_value=10, value=6, step=1)
    
    # Load model and data
    with st.spinner("🔄 Loading model and fetching real-time weather data..."):
        model_data = load_model()
        df = fetch_weather_data(use_api=use_api)
    
    if model_data is None or df is None:
        st.error("Failed to load model or data. Please check configuration.")
        st.info("Make sure you have:")
        st.code("""
1. Trained the model: python scripts/train_models.py
2. Downloaded the data: python scripts/fetch_data.py
3. Running from project root: cd e:\\code\\final-ml
        """)
        return
    
    model = model_data['model']
    metrics = model_data.get('metrics', {})
    config = model_data.get('config', {})
    
    # Current weather
    latest = df.iloc[-1]
    latest_date = latest['datetime'].strftime("%B %d, %Y")
    
    st.markdown(f"## 📅 Current Weather - {latest_date}")
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%I:%M %p')}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        temp_change = latest['temp'] - df.iloc[-2]['temp']
        st.metric(
            "🌡️ Temperature",
            f"{latest['temp']:.1f}°C",
            f"{temp_change:+.1f}°C"
        )
    
    with col2:
        st.metric(
            "💧 Humidity",
            f"{latest['humidity']:.0f}%",
            f"{latest['humidity'] - df.iloc[-2]['humidity']:+.0f}%"
        )
    
    with col3:
        st.metric(
            "🌬️ Wind Speed",
            f"{latest['windspeed']:.1f} km/h"
        )
    
    with col4:
        st.metric(
            "☁️ Cloud Cover",
            f"{latest['cloudcover']:.0f}%"
        )
    
    with col5:
        if 'precip' in latest and latest['precip'] > 0:
            st.metric(
                "🌧️ Precipitation",
                f"{latest['precip']:.1f} mm"
            )
        else:
            st.metric(
                "☀️ Conditions",
                "Clear"
            )
    
    st.markdown("---")
    
    # Forecast
    if show_forecast:
        st.markdown("## 🔮 5-Day Forecast")
        
        with st.spinner("Generating forecast..."):
            try:
                # Prepare features using the same method as training
                features = prepare_features_for_prediction(df, config)
                
                if features is None:
                    st.error("Failed to prepare features for prediction")
                    return
                
                # Make prediction
                predictions = model.predict(features)[0]
                
                # Create forecast dates
                last_date = df['datetime'].iloc[-1]
                forecast_dates = [last_date + timedelta(days=i+1) for i in range(5)]
                
                # Display forecast
                forecast_cols = st.columns(5)
                
                for i, (date, temp) in enumerate(zip(forecast_dates, predictions)):
                    with forecast_cols[i]:
                        day_name = date.strftime("%a")
                        date_str = date.strftime("%b %d")
                        
                        # Determine icon and description based on temperature
                        if temp < 20:
                            icon = "🌤️"
                            condition = "Cool"
                        elif temp < 25:
                            icon = "☀️"
                            condition = "Pleasant"
                        elif temp < 30:
                            icon = "🌞"
                            condition = "Warm"
                        else:
                            icon = "🔥"
                            condition = "Hot"
                        
                        # Add gradient based on temperature
                        if temp < 25:
                            gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                        elif temp < 30:
                            gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
                        else:
                            gradient = "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)"
                        
                        st.markdown(f"""
                        <div class="forecast-box" style="background: {gradient};">
                            <div class="weather-icon">{icon}</div>
                            <p style="font-size: 1.3rem; font-weight: 600; margin: 0.5rem 0;">
                                {day_name}
                            </p>
                            <p style="font-size: 0.85rem; opacity: 0.9; margin: 0;">
                                {date_str}
                            </p>
                            <h2 style="margin: 0.8rem 0; font-size: 2rem;">
                                {temp:.1f}°C
                            </h2>
                            <p style="font-size: 0.9rem; opacity: 0.9; margin: 0;">
                                {condition}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Forecast chart
                st.markdown("### 📈 Temperature Trend")
                
                # Combine historical and forecast
                hist_dates = df['datetime'].tail(14).tolist()
                hist_temps = df['temp'].tail(14).tolist()
                
                # Create matplotlib figure with custom size
                fig, ax = plt.subplots(figsize=(chart_width, chart_height))
                
                # Historical data
                ax.plot(hist_dates, hist_temps, 
                       color='#1f77b4', linewidth=2, 
                       marker='o', markersize=6, 
                       label='Historical', zorder=2)
                
                # Forecast data
                ax.plot(forecast_dates, predictions, 
                       color='#ff7f0e', linewidth=2, 
                       linestyle='--', marker='*', markersize=10, 
                       label='Forecast', zorder=2)
                
                # Styling
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
                ax.set_title('14-Day Historical + 5-Day Forecast', fontsize=14, fontweight='bold', pad=20)
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Historical data
    if show_historical:
        st.markdown("## 📊 Historical Data")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=df['datetime'].max() - timedelta(days=90),
                min_value=df['datetime'].min(),
                max_value=df['datetime'].max()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=df['datetime'].max(),
                min_value=df['datetime'].min(),
                max_value=df['datetime'].max()
            )
        
        # Filter data
        mask = (df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))
        filtered_df = df.loc[mask]
        
        # Plot
        fig, ax = plt.subplots(figsize=(chart_width, chart_height))
        ax.plot(filtered_df['datetime'], filtered_df['temp'], 
               color='#1f77b4', linewidth=2)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Historical Temperature', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{filtered_df['temp'].mean():.1f}°C")
        with col2:
            st.metric("Maximum", f"{filtered_df['temp'].max():.1f}°C")
        with col3:
            st.metric("Minimum", f"{filtered_df['temp'].min():.1f}°C")
    
    # Predicted vs Actual Plot
    if show_pred_vs_actual and show_forecast:
        st.markdown("---")
        st.markdown("## 🎯 Model Validation: Predicted vs Actual")
        st.caption("Comparing model predictions against actual historical temperatures")
        
        try:
            # Use recent historical data for validation
            validation_days = 30
            validation_df = df.tail(validation_days).copy()
            
            # Generate predictions for validation period
            pred_temps = []
            actual_temps = validation_df['temp'].values
            dates = validation_df['datetime'].values
            
            # Make predictions for each day
            for i in range(len(validation_df) - 5):  # Need 5 days for forecast
                hist_subset = df.iloc[:len(df)-validation_days+i]
                features = prepare_features_for_prediction(hist_subset, config)
                if features is not None:
                    pred = model.predict(features)[0]
                    pred_temps.append(pred[0])  # Take 1-day ahead prediction
                else:
                    pred_temps.append(np.nan)
            
            # Align arrays
            pred_temps = np.array(pred_temps)
            actual_temps_aligned = actual_temps[:len(pred_temps)]
            dates_aligned = dates[:len(pred_temps)]
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(chart_width, chart_height))
            
            # Time series comparison
            ax1.plot(dates_aligned, actual_temps_aligned, 
                    color='#1f77b4', linewidth=2, marker='o', 
                    markersize=4, label='Actual', alpha=0.7)
            ax1.plot(dates_aligned, pred_temps, 
                    color='#ff7f0e', linewidth=2, marker='s', 
                    markersize=4, label='Predicted', alpha=0.7)
            ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
            ax1.set_title('Actual vs Predicted Temperature', fontsize=13, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Scatter plot
            ax2.scatter(actual_temps_aligned, pred_temps, 
                       alpha=0.6, s=50, color='#2ca02c')
            
            # Perfect prediction line
            min_temp = min(actual_temps_aligned.min(), pred_temps.min())
            max_temp = max(actual_temps_aligned.max(), pred_temps.max())
            ax2.plot([min_temp, max_temp], [min_temp, max_temp], 
                    'r--', linewidth=2, label='Perfect Prediction')
            
            ax2.set_xlabel('Actual Temperature (°C)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Predicted Temperature (°C)', fontsize=11, fontweight='bold')
            ax2.set_title('Prediction Accuracy', fontsize=13, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            
            # Add R² score
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(actual_temps_aligned, pred_temps)
            mae = mean_absolute_error(actual_temps_aligned, pred_temps)
            rmse = np.sqrt(mean_squared_error(actual_temps_aligned, pred_temps))
            
            ax2.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.2f}°C\nRMSE = {rmse:.2f}°C',
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Validation metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{r2:.3f}")
            with col2:
                st.metric("MAE", f"{mae:.2f}°C")
            with col3:
                st.metric("RMSE", f"{rmse:.2f}°C")
            with col4:
                accuracy = 100 * (1 - mae / actual_temps_aligned.mean())
                st.metric("Accuracy", f"{accuracy:.1f}%")
                
        except Exception as e:
            st.error(f"Error creating validation plot: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Model metrics
    if show_metrics and metrics:
        st.markdown("---")
        st.markdown("## 🎯 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"{metrics.get('RMSE_macro', 0):.4f}°C")
        with col2:
            st.metric("MAE", f"{metrics.get('MAE_macro', 0):.4f}°C")
        with col3:
            st.metric("R² Score", f"{metrics.get('R2_macro', 0):.4f}")
        with col4:
            st.metric("MAPE", f"{metrics.get('MAPE_macro', 0):.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 2rem 0;">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Built with ❤️ using Streamlit | Powered by Visual Crossing Weather API</p>
        <p style="font-size: 0.9rem;">© 2025 Weather Forecast App | Last updated: {}</p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">🌍 Ho Chi Minh City, Vietnam | 🔴 Real-time Data</p>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y at %I:%M %p")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()