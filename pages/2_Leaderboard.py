# pages/2_Leaderboard.py
import streamlit as st
import pandas as pd

st.title("🏆 Leaderboard")
st.write("This page shows all the predictions made during this session.")

# --- Initialize Leaderboard in Session State ---
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = pd.DataFrame(columns=["Age", "Estimated Salary", "Prediction", "Probability"])

# --- Display Leaderboard ---
if not st.session_state.leaderboard.empty:
    st.dataframe(st.session_state.leaderboard, use_container_width=True)

    # Option to download leaderboard as CSV
    csv = st.session_state.leaderboard.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Leaderboard as CSV", data=csv, file_name="leaderboard.csv", mime="text/csv")
else:
    st.info("No predictions yet! Go to the Prediction page and make some predictions.")
