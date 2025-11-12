# 🚀 Deploying to Hugging Face Spaces

This guide explains how to deploy your Weather Forecasting Streamlit app to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Create account at https://huggingface.co/join
2. **Git**: Installed on your system
3. **Trained Model**: Run `python scripts/train_models.py` first

## 🎯 Step-by-Step Deployment

### Step 1: Create a Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in details:
   - **Owner**: Your username
   - **Space name**: `vietnam-weather-forecast` (or your choice)
   - **License**: MIT
   - **SDK**: Streamlit
   - **Hardware**: CPU (free tier)
4. Click **"Create Space"**

### Step 2: Clone the Space Repository

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cd <space-name>
```

### Step 3: Prepare Files

Copy the necessary files to your Space directory:

```bash
# From your project root (final-ml/)
# Copy Streamlit app
cp streamlit_app/app.py vietnam-weather-forecast/

# Copy source code
cp -r src/ vietnam-weather-forecast/

# Copy trained model
cp models/final_model.pkl vietnam-weather-forecast/models/

# Copy requirements
cp requirements.txt vietnam-weather-forecast/
```

### Step 4: Create Space Configuration

Create `vietnam-weather-forecast/README.md`:

```markdown
---
title: Vietnam Weather Forecast
emoji: 🌤️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.25.0
app_file: app.py
pinned: false
---

# Vietnam Weather Forecast

AI-Powered 5-Day Temperature Prediction for Hanoi, Vietnam.

## Features
- 5-day temperature forecasting
- Historical data visualization
- Real-time predictions
- Model performance metrics

## Model
- Ensemble (BayesianRidge + HistGradientBoosting)
- Trained on 10 years of data (2015-2025)
- RMSE < 0.8°C

Built with Streamlit and Scikit-learn.
```

### Step 5: Streamline Requirements

Create a lighter `requirements.txt` for Spaces:

```txt
# Core ML
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# Visualization
matplotlib==3.7.2
plotly==5.15.0

# Streamlit
streamlit==1.25.0

# Utilities
python-dotenv==1.0.0
```

### Step 6: Push to Hugging Face

```bash
cd vietnam-weather-forecast

# Add all files
git add .

# Commit
git commit -m "Initial deployment: Weather forecasting app"

# Push to Hugging Face
git push
```

### Step 7: Wait for Build

- Hugging Face will automatically build your Space
- Check the **Logs** tab for build progress
- Usually takes 2-5 minutes

### Step 8: Test Your App

- Once built, visit: `https://huggingface.co/spaces/<your-username>/<space-name>`
- Test all features
- Share the link!

## 🔧 Troubleshooting

### Issue: Module Not Found

**Solution**: Make sure `src/` directory is copied and `sys.path` is updated in `app.py`:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
```

### Issue: Model Not Found

**Solution**: Ensure `models/final_model.pkl` is in the Space directory:

```bash
mkdir -p models
cp /path/to/final_model.pkl models/
```

### Issue: Memory Error

**Solution**: 
1. Use a smaller model or reduce sample data
2. Request CPU upgrade (paid tier)
3. Optimize model size

### Issue: Slow Loading

**Solution**:
1. Add caching decorators:
```python
@st.cache_resource
def load_model():
    ...

@st.cache_data(ttl=3600)
def fetch_data():
    ...
```

2. Reduce data fetched
3. Pre-process data before deployment

## 🎨 Customization

### Change App Theme

Create `vietnam-weather-forecast/.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Add Secrets

For API keys or sensitive data:

1. Go to Space Settings
2. Click **"Repository secrets"**
3. Add secrets (e.g., `WEATHER_API_KEY`)
4. Access in code:

```python
import streamlit as st
api_key = st.secrets["WEATHER_API_KEY"]
```

## 📊 Monitoring

### View Usage Stats

- Hugging Face provides analytics
- Check **Settings** > **Analytics**
- Monitor:
  - Visitors
  - Session duration
  - Geographic distribution

### Update Model

When you retrain your model:

```bash
# 1. Copy new model
cp models/final_model.pkl vietnam-weather-forecast/models/

# 2. Commit and push
cd vietnam-weather-forecast
git add models/final_model.pkl
git commit -m "Update model with latest data"
git push
```

## 🌟 Best Practices

1. **Keep it Light**: Remove unnecessary dependencies
2. **Cache Everything**: Use `@st.cache_resource` and `@st.cache_data`
3. **Error Handling**: Add try-except blocks
4. **Loading States**: Use `st.spinner()` for long operations
5. **Mobile Friendly**: Test on mobile devices
6. **Documentation**: Update README with features

## 📝 Example Space Structure

```
vietnam-weather-forecast/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── README.md                 # Space description
├── src/                      # Source modules
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   └── model_training.py
├── models/                   # Trained models
│   └── final_model.pkl
└── .streamlit/               # Streamlit config (optional)
    └── config.toml
```

## 🔗 Resources

- **Hugging Face Docs**: https://huggingface.co/docs/hub/spaces-sdks-streamlit
- **Streamlit Docs**: https://docs.streamlit.io
- **Example Spaces**: https://huggingface.co/spaces/docs-demos

## 🎉 After Deployment

Share your Space:
- Twitter: "Check out my AI weather forecast app 🌤️"
- LinkedIn: Showcase your ML project
- GitHub: Link to your repository
- Resume: Add to your portfolio

## 💡 Next Steps

1. **Add More Features**:
   - Hourly forecasts
   - Multiple cities
   - Weather alerts
   - Export predictions

2. **Improve UI**:
   - Add images/icons
   - Better color scheme
   - Responsive design

3. **Enhance Model**:
   - Add more features
   - Try ensemble methods
   - Real-time updates

Good luck with your deployment! 🚀
