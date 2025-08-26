# pages/1_Prediction.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_extras.colored_header import colored_header

# --- LOAD MODEL ---
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🧮 Insurance Purchase Prediction")
st.write("Enter Age and Estimated Salary to predict if the customer will purchase insurance.")

# --- USER INPUT FORM ---
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    salary = st.number_input("Estimated Salary", min_value=1000, max_value=200000, step=1000)

    submit_btn = st.form_submit_button("Predict")

# --- PREDICTION ---
if submit_btn:
    features = np.array([[age, salary]])

    # Get prediction (0 = No, 1 = Yes)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        result_text = "Likely to Purchase"
        colored_header(
            label=f"✅ {result_text}",
            description=f"Probability: {probability:.2f}",
            color_name="green-70"
        )
        st.balloons()
    else:
        result_text = "Not Likely to Purchase"
        colored_header(
            label=f"⚠️ {result_text}",
            description=f"Probability: {probability:.2f}",
            color_name="red-70"
        )
        st.snow()

    # --- Save to Leaderboard ---
    new_entry = {
        "Age": age,
        "Estimated Salary": salary,
        "Prediction": result_text,
        "Probability": round(probability, 2)
    }

    if "leaderboard" not in st.session_state:
        st.session_state.leaderboard = pd.DataFrame(
            columns=["Age", "Estimated Salary", "Prediction", "Probability"]
        )

    st.session_state.leaderboard = pd.concat(
        [st.session_state.leaderboard, pd.DataFrame([new_entry])],
        ignore_index=True
    )
