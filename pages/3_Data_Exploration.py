# pages/3_Data_Exploration.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from auth import require_login
require_login()


# --- LOAD DATA ---
df = pd.read_csv("data/Social_Network_Ads.csv")

st.title("📊 Data Exploration")
st.write("Explore the dataset used for training the logistic regression model.")

# --- Show Dataset ---
st.subheader("Dataset Preview")
st.dataframe(df.head())

# --- Basic Statistics ---
st.subheader("Summary Statistics")
st.write(df[['Age', 'EstimatedSalary']].describe())

# --- Age Distribution ---
st.subheader("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Age'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# --- Salary Distribution ---
st.subheader("Estimated Salary Distribution")
fig, ax = plt.subplots()
sns.histplot(df['EstimatedSalary'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# --- Correlation Heatmap ---
st.subheader("Feature Correlation")
fig, ax = plt.subplots()
sns.heatmap(df[['Age','EstimatedSalary','Purchased']].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --- Decision Boundary Plot ---
st.subheader("Decision Boundary (Logistic Regression)")

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

# Create meshgrid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1000, X[:, 1].max() + 1000
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1000))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
ax.set_xlabel("Age")
ax.set_ylabel("Estimated Salary")
ax.set_title("Decision Boundary")

# Legend
handles, labels = scatter.legend_elements()
ax.legend(handles, ["Not Purchased", "Purchased"], title="Target")

st.pyplot(fig)
