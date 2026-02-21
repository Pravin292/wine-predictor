import streamlit as st
import numpy as np
import pandas as pd
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wine Intelligence Pro",
    page_icon="🍷",
    layout="wide"
)

# ---------------- Fail-Safe Imports ----------------
try:
    from utils import load_model_and_metrics, predict_quality, generate_feature_importance_plot, get_grade_info
except Exception as e:
    st.error(f"⚠️ Critical Module Failure: {e}")
    st.stop()

# ---------------- CUSTOM CSS "Midnight Glass" ----------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: #e2e8f0; font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3 { color: #38bdf8 !important; font-weight: 700; }
.glass-panel { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); margin-bottom: 20px; }
.stNumberInput input, .stSelectbox select, .stTextInput input { background-color: rgba(15, 23, 42, 0.6) !important; color: #f8fafc !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; border-radius: 8px !important; }
.stNumberInput input:focus, .stSelectbox select:focus { border: 1px solid #38bdf8 !important; box-shadow: 0 0 10px rgba(56, 189, 248, 0.5) !important; }
.stButton>button { background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; border-radius: 8px; font-weight: 600; padding: 0.6em 1.5em; border: none; transition: all 0.3s ease; width: 100%; margin-top: 15px; }
.stButton>button:hover { background: linear-gradient(90deg, #60a5fa, #a78bfa); box-shadow: 0 0 15px rgba(139, 92, 246, 0.6); transform: translateY(-2px); }
.result-card-success { background: rgba(16, 185, 129, 0.15); border-left: 5px solid #10b981; padding: 20px; border-radius: 10px; margin-top: 20px; }
.result-card-warning { background: rgba(245, 158, 11, 0.15); border-left: 5px solid #f59e0b; padding: 20px; border-radius: 10px; margin-top: 20px; }
.result-card-danger { background: rgba(239, 68, 68, 0.15); border-left: 5px solid #ef4444; padding: 20px; border-radius: 10px; margin-top: 20px; }
.metric-text { font-size: 24px; font-weight: bold; color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD RESOURCES ----------------
try:
    model, metrics = load_model_and_metrics()
except Exception as e:
    st.error(f"Engine Boot Error: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

if model is None:
    st.warning("🍷 Model artifacts or metrics.json not found.")
    st.info("Ensure `python model_training.py` has been executed to generate binaries.")
    st.stop()

# ---------------- SIDEBAR: INPUTS ----------------
st.sidebar.markdown("### 🧪 Phsyiochemical Features")
st.sidebar.caption("Adjust the sliders to simulate a wine sample.")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.4)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 2.0, 0.7)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 16.0, 1.9)
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.6, 0.076)
free_sulfur_dioxide = st.sidebar.slider("Free S.D.", 1.0, 75.0, 11.0)
total_sulfur_dioxide = st.sidebar.slider("Total S.D.", 6.0, 290.0, 34.0)
density = st.sidebar.slider("Density", 0.99, 1.01, 0.9978)
pH = st.sidebar.slider("pH", 2.5, 4.5, 3.51)
sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.56)
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 9.4)

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("Analyze Composition")

# ---------------- HEADER ----------------
st.markdown("<h1>🍷 Wine Intelligence Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Advanced Random Forest Quality Assessor Dashboard</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- UI TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Explanation & Metrics", "📂 Raw Analytics"])

with tab1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Inference Engine")
    
    if analyze_btn:
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                                residual_sugar, chlorides, free_sulfur_dioxide,
                                total_sulfur_dioxide, density, pH,
                                sulphates, alcohol]])

        quality = predict_quality(model, input_data)
        grade, style_class = get_grade_info(quality)

        st.markdown(f"""
            <div class="{style_class}">
                <div class="metric-text">{grade} (Score: {quality})</div>
                <p style="color: #cbd5e1; margin-top: 5px;">Based on the physiochemical composition, this wine exhibits characteristics aligning with {grade.lower()} standards.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Please adjust parameters in the sidebar and click **Analyze Composition** to begin.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 Model DNA & Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R-Squared (Test)", metrics["R-squared (R2)"])
    col2.metric("Mean Absolute Error", metrics["Mean Absolute Error (MAE)"])
    col3.metric("Cross-Validation Score", metrics["Cross-Validation R2 (Mean)"])
    
    st.markdown("---")
    fig = generate_feature_importance_plot(metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📥 Underlying Dataset")
    try:
        df = pd.read_csv("data/winequality-red.csv")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Showing first 100 rows of the UCI Wine Quality Dataset (Red).")
    except Exception as e:
        st.error(f"Failed to load underlying training data: {e}")
    st.markdown('</div>', unsafe_allow_html=True)