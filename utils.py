import joblib
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def load_model_and_metrics():
    """Loads the serialized model and performance metrics."""
    try:
        model = joblib.load("wine_quality_model.pkl")
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        return model, metrics
    except FileNotFoundError:
        return None, None

def predict_quality(model, input_data):
    """Runs inference on user input."""
    prediction = model.predict(input_data)
    return round(prediction[0], 2)

def generate_feature_importance_plot(metrics):
    """Generates a styled Plotly chart for feature importance."""
    fi = metrics.get('feature_importance', {})
    if not fi:
        return None
        
    df_fi = pd.DataFrame({
        'Feature': list(fi.keys()),
        'Importance': list(fi.values())
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(
        df_fi, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Random Forest Feature Importance'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title_font=dict(size=20, color='#38bdf8'),
        xaxis=dict(title="Importance Factor", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="")
    )
    fig.update_traces(marker_color='#8b5cf6')
    
    return fig

def get_grade_info(quality):
    """Returns stylistic parameters based on quality score."""
    if quality >= 7:
        return "🏆 Elite Reserve", "result-card-success"
    elif quality >= 5:
        return "🍷 Vintage Standard", "result-card-warning"
    else:
        return "🍂 Entry Blend", "result-card-danger"
