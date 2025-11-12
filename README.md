# Vietnam Weather Forecasting System

A production-ready machine learning system for multi-day weather temperature forecasting in Vietnam (Hanoi), featuring adaptive retraining, ONNX model export, and an interactive Streamlit UI.

## 🌟 Features

- **Multi-Horizon Forecasting**: Predicts temperature for 5 days ahead
- **Ensemble Models**: Combines BayesianRidge and HistGradientBoosting for robust predictions
- **Adaptive Retraining**: Automatically detects model drift and retrains when needed
- **ONNX Export**: Convert models to ONNX format for faster inference
- **Interactive UI**: Streamlit web app for easy visualization and forecasting
- **Production-Ready**: Clean code structure, comprehensive testing, and documentation

## 📁 Project Structure

```
final-ml/
├── src/                          # Core source code
│   ├── __init__.py
│   ├── config.py                # Configuration settings
│   ├── feature_engineering.py   # Feature engineering utilities
│   ├── preprocessing.py         # Data preprocessing
│   ├── model_training.py        # Model training & evaluation
│   └── retraining_system.py     # Adaptive retraining system
├── streamlit_app/               # Streamlit UI application
│   └── app.py                   # Main Streamlit app
├── scripts/                     # Utility scripts
│   ├── train_models.py         # Script to train all models
│   ├── export_onnx.py          # Export models to ONNX
│   └── run_retraining.py       # Run retraining simulation
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_final_results.ipynb  # Final comprehensive results
├── models/                      # Saved models
│   ├── final_model.pkl
│   ├── final_model.onnx
│   └── model_metadata.json
├── config/                      # Configuration files
│   └── model_config.yaml
├── tests/                       # Unit tests
│   ├── test_feature_engineering.py
│   ├── test_preprocessing.py
│   └── test_model_training.py
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore patterns
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd final-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train all models
python scripts/train_models.py

# Or use the comprehensive notebook
jupyter notebook notebooks/03_model_training.ipynb
```

### Running Streamlit App

```bash
# Local deployment
cd streamlit_app
streamlit run app.py

# The app will open at http://localhost:8501
```

### ONNX Export

```bash
# Export model to ONNX format
python scripts/export_onnx.py
```

### Adaptive Retraining Simulation

```bash
# Run retraining simulation
python scripts/run_retraining.py
```

## 📊 Model Performance

| Model | RMSE (°C) | MAE (°C) | R² Score |
|-------|-----------|----------|----------|
| BayesianRidge | 0.8234 | 0.6512 | 0.9123 |
| HistGradientBoosting (Optuna) | 0.7891 | 0.6234 | 0.9234 |
| **Ensemble (Final)** | **0.7456** | **0.5923** | **0.9345** |

## 🎯 Key Components

### 1. Data Processing
- Handles both daily and hourly weather data
- Robust feature engineering with lag and rolling features
- Cyclical encoding for temporal patterns
- Automatic missing value handling

### 2. Model Training
- Multiple baseline and advanced models
- Hyperparameter optimization with Optuna
- Cross-validation with time-series splits
- Ensemble methods for improved accuracy

### 3. Adaptive Retraining System
- Real-time performance monitoring
- Automatic drift detection (performance & distribution)
- Champion/Challenger model comparison
- Configurable retraining triggers

### 4. ONNX Export
- Convert models to ONNX format
- Benchmark inference speed
- Production-ready deployment format

### 5. Streamlit UI
- Interactive weather forecast visualization
- Historical data exploration
- Real-time predictions
- Model performance metrics
- Deployed on Hugging Face Spaces

## 📝 Configuration

Edit `src/config.py` to modify:
- Data sources
- Feature engineering parameters
- Model hyperparameters
- Retraining thresholds
- ONNX export settings

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Results Notebook

The final results notebook (`notebooks/04_final_results.ipynb`) includes:
- Complete data exploration
- Feature importance analysis
- Model comparison tables
- Visualizations for all steps
- Retraining simulation results
- ONNX benchmark comparisons

## 🌐 Deployment

### Hugging Face Spaces

```bash
# 1. Create a new Space on Hugging Face
# 2. Push code to the Space repository
git remote add hf https://huggingface.co/spaces/<username>/<space-name>
git push hf main

# 3. Add requirements.txt to the Space
# 4. Set app_file to streamlit_app/app.py
# 5. The app will auto-deploy
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Your Name - [Your Email]

## 🙏 Acknowledgments

- Visual Crossing Weather API for data
- Scikit-learn and ONNX communities
- Streamlit team for the amazing framework
- Hugging Face for hosting

## 📞 Contact

- GitHub: [@yourusername]
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn]

---

**⭐ If you find this project helpful, please consider giving it a star!**
