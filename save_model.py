import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("star_classification.csv")

# Clean
df = df[(df["u"] > 0) & (df["u"] < 30)]
df = df[(df["g"] > 0) & (df["g"] < 30)]
df = df[(df["r"] > 0) & (df["r"] < 30)]

# Features
df["g_r"] = df["g"] - df["r"]
df["u_g"] = df["u"] - df["g"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]
df["u_r"] = df["u"] - df["r"]

features = ["u","g","r","i","z","redshift","g_r","u_g","r_i","i_z","u_r"]

X = df[features]
y = df["class"]

# Train
model = RandomForestClassifier()
model.fit(X, y)

# Save
joblib.dump(model, "model.pkl")

print("Model saved ✅")