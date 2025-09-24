import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DATA_PATH = "data/Social_Network_Ads.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Hardcoded login
VALID_USERNAME = "Selva"
VALID_PASSWORD = "0412"

st.set_page_config(page_title="Insurance App", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def train_and_save_model(df):
    X = df[["Age", "EstimatedSalary"]]
    y = df["Purchased"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    df = load_data()
    train_and_save_model(df)
    return load_model()

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🔒 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()  # Updated
            else:
                st.error("Invalid credentials.")
    else:
        st.sidebar.success(f"Welcome, {VALID_USERNAME}!")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()  # Updated
login()

if st.session_state.get("logged_in"):
    st.title("🏠 Insurance Purchase Predictor")
    st.write("Use the sidebar to navigate between pages.")