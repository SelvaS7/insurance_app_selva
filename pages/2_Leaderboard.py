import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from auth import require_login
require_login()


MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "data/Social_Network_Ads.csv"

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    df = pd.read_csv(DATA_PATH)
    X = df[["Age", "EstimatedSalary"]]
    y = df["Purchased"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_scaled, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return model, scaler

st.title("🏆 Leaderboard - Top 10 Customers Most Likely to Buy")

df = pd.read_csv(DATA_PATH)
model, scaler = load_model()

X = scaler.transform(df[["Age", "EstimatedSalary"]])
probs = model.predict_proba(X)[:,1]
df["Purchase_Prob"] = probs
top10 = df.sort_values("Purchase_Prob", ascending=False).head(10)

st.dataframe(top10.reset_index(drop=True))

csv = top10.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download CSV", csv, "top10_customers.csv", "text/csv")
