"""
Vietnam Weather Forecasting App
================================
A production-ready Streamlit application for 5-day weather temperature forecasting.
Deployed on Hugging Face Spaces.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import cloudpickle as pickle  # Back to cloudpickle as requested
import sys
from pathlib import Path
import warnings

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
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .forecast-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_weather_data():
    """Fetch latest weather data."""
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
    st.markdown('<div class="sub-header">AI-Powered 5-Day Temperature Prediction for Hanoi</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/hoducmanh222/temp_holder/main/vietnam_flag.png", 
                width=100)
        st.title("⚙️ Settings")
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info("""
        **Model:** Ensemble (BayesianRidge + HGB)  
        **Forecast Horizon:** 5 days  
        **Training Data:** 2015-2025  
        **Accuracy:** RMSE < 0.8°C
        """)
        
        st.markdown("---")
        st.markdown("### 📍 Location")
        st.write("**City:** Hanoi, Vietnam")
        st.write("**Coordinates:** 21.0285° N, 105.8542° E")
        
        st.markdown("---")
        show_historical = st.checkbox("Show Historical Data", value=True)
        show_metrics = st.checkbox("Show Model Metrics", value=True)
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        model_data = load_model()
        df = fetch_weather_data()
    
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
    st.markdown("## 📅 Current Weather")
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🌡️ Temperature",
            f"{latest['temp']:.1f}°C",
            f"{latest['temp'] - df.iloc[-2]['temp']:.1f}°C"
        )
    
    with col2:
        st.metric(
            "💧 Humidity",
            f"{latest['humidity']:.0f}%"
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
    
    st.markdown("---")
    
    # Forecast
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
                    
                    # Determine icon based on temperature
                    if temp < 20:
                        icon = "🌤️"
                    elif temp < 28:
                        icon = "☀️"
                    else:
                        icon = "🌡️"
                    
                    st.markdown(f"""
                    <div class="forecast-box">
                        <h3 style="text-align: center; margin: 0;">{icon}</h3>
                        <p style="text-align: center; font-size: 1.2rem; margin: 0.5rem 0;">
                            {day_name}
                        </p>
                        <p style="text-align: center; font-size: 0.9rem; margin: 0;">
                            {date_str}
                        </p>
                        <h2 style="text-align: center; margin: 1rem 0;">
                            {temp:.1f}°C
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Forecast chart
            st.markdown("### 📈 Temperature Trend")
            
            # Combine historical and forecast
            hist_dates = df['datetime'].tail(14).tolist()
            hist_temps = df['temp'].tail(14).tolist()
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_temps,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=predictions,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=8, symbol='star')
            ))
            
            fig.update_layout(
                title="14-Day Historical + 5-Day Forecast",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
        fig = px.line(
            filtered_df,
            x='datetime',
            y='temp',
            title="Historical Temperature",
            labels={'datetime': 'Date', 'temp': 'Temperature (°C)'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{filtered_df['temp'].mean():.1f}°C")
        with col2:
            st.metric("Maximum", f"{filtered_df['temp'].max():.1f}°C")
        with col3:
            st.metric("Minimum", f"{filtered_df['temp'].min():.1f}°C")
    
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
    <div style="text-align: center; color: #888;">
        <p>Built with ❤️ using Streamlit | Data from Visual Crossing Weather API</p>
        <p>© 2025 Weather Forecast App | Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()