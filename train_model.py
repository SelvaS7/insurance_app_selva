import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# ----------------- Load Dataset -----------------
df = pd.read_csv("Social_Network_Ads.csv")

# Use only Age and EstimatedSalary as features
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# ----------------- Split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------- Scale Features -----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------- Train Model -----------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ----------------- Save Model and Scaler -----------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully!")
