# 🍷 Wine Quality Predictor (Midnight Glass Edition)

This project implements a professional, self-contained **Wine Quality Prediction** system using a Random Forest Regressor. It features an elite dark-mode UI and a modular ML architecture.

## 🏗 Project Structure
- `app.py`: Streamlit frontend with Sidebar inputs and Analytics tabs.
- `model_training.py`: Independent script to fetch UCI data, train the model, and export metrics.
- `utils.py`: Helper functions for Plotly charts and model loading.
- `requirements.txt`: Project-specific dependencies.
- `data/`: Contains the raw dataset.

## 🚀 Usage Instructions

### 1. Model Training
To fetch the latest dataset and retrain the model:
```bash
python3 model_training.py
```

### 2. Launching the UI
Run the Streamlit application:
```bash
streamlit run app.py
```

## 📊 Features
- **Prediction Tab**: High-fidelity inference for physiochemical variables.
- **Evaluation Tab**: Visualizes Feature Importance and Model RMSE/R2.
- **Analytics Tab**: Deep-dive into the raw dataset structure.
