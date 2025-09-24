import streamlit as st
import pandas as pd
import joblib
import os
import time
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go

# ----------------- Page Config -----------------
st.set_page_config(page_title="Insurance Prediction", layout="wide")

# ----------------- CSS: Gradient BG + Multiple Rockets + Glassmorphism -----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(270deg, #4facfe, #00f2fe, #43e97b, #38f9d7);
    background-size: 800% 800%;
    animation: gradientBG 25s ease infinite;
    overflow: hidden;
    position: relative;
    color: #fff;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Floating rockets */
.rocket {
    position: absolute;
    width: 60px;
    height: 60px;
    background: url('https://upload.wikimedia.org/wikipedia/commons/e/e4/Paper_rocket_icon.svg') no-repeat center/contain;
    opacity: 0.5;
}
.rocket:nth-child(1) { left: 20%; top: 100%; animation: float1 18s linear infinite; }
.rocket:nth-child(2) { left: 50%; top: 110%; animation: float2 22s linear infinite; }
.rocket:nth-child(3) { left: 80%; top: 120%; animation: float3 20s linear infinite; }

@keyframes float1 { 
    0%{transform: translateY(0) rotate(0);} 
    100%{transform: translateY(-130vh) rotate(20deg);} 
}
@keyframes float2 { 
    0%{transform: translateY(0) rotate(0);} 
    100%{transform: translateY(-140vh) rotate(-15deg);} 
}
@keyframes float3 { 
    0%{transform: translateY(0) rotate(0);} 
    100%{transform: translateY(-150vh) rotate(10deg);} 
}

/* Cards */
.input-card {
    background-color: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.result-box {
    background-color: rgba(255,255,255,0.2);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 1.2rem;
    margin-top: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}

/* Buttons */
div.stButton > button {
    background-color: #007bff;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1.5em;
    font-weight: 600;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #0056b3;
}
</style>

<!-- Multiple rockets -->
<div class="rocket"></div>
<div class="rocket"></div>
<div class="rocket"></div>
""", unsafe_allow_html=True)

# ----------------- Hero Section -----------------
st.markdown("<h1 style='text-align:center;'>🚀 Insurance Purchase Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict the likelihood of a customer buying insurance using Age & Salary.</p>", unsafe_allow_html=True)

# ----------------- Load Model & Scaler -----------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler not found. Train them first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ----------------- Layout: Form and Details -----------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("Enter Details")
    age = st.slider("Age", 18, 100, 30)
    salary = st.number_input("Estimated Salary", min_value=1000, max_value=99999999, value=50000, step=1000)
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            time.sleep(1)  # simulate loading
            input_data = pd.DataFrame([[age, salary]], columns=["Age", "EstimatedSalary"])
            scaled_input = scaler.transform(input_data)
            probability = model.predict_proba(scaled_input)[0][1] * 100

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Purchase Probability (%)"},
            gauge={'axis': {'range': [0,100]}, 'bar': {'color': "#007bff"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if probability >= 50:
            st.markdown(f'<div class="result-box">✅ Likely to purchase insurance.<br>Confidence: {probability:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box">❌ Unlikely to purchase insurance.<br>Confidence: {probability:.2f}%</div>', unsafe_allow_html=True)

        # Save prediction immediately after predicting
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"Age": age, "Salary": salary, "Probability": f"{probability:.2f}%"})

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("📊 About This Tool")
    st.write("""
    - Uses **Logistic Regression** trained on Age & Salary.  
    - Displays predictions as an animated **gauge chart**.
    - Built with **Streamlit**, **Scikit-Learn**, and **Plotly**.  
    """)

# ----------------- Prediction History -----------------
if "history" in st.session_state and st.session_state.history:
    st.subheader("📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    # 📥 CSV Download Button
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=hist_df.to_csv(index=False),
        file_name="predictions_history.csv",
        mime="text/csv"
    )

# ----------------- Footer -----------------
st.caption("Built by Selvakumar by using Streamlit, Scikit-Learn & Plotly.")
