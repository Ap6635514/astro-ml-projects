import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ✅ Load + train model ONCE (cached)
def load_model():
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

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

model = load_model()

# ✅ Prediction
def predict(u, g, r, i, z, redshift):
    data = pd.DataFrame({
        "u":[u], "g":[g], "r":[r],
        "i":[i], "z":[z], "redshift":[redshift]
    })

    data["g_r"] = data["g"] - data["r"]
    data["u_g"] = data["u"] - data["g"]
    data["r_i"] = data["r"] - data["i"]
    data["i_z"] = data["i"] - data["z"]
    data["u_r"] = data["u"] - data["r"]

    prediction = model.predict(data)[0]

    if prediction == "STAR":
        return "⭐ STAR"
    elif prediction == "GALAXY":
        return "🌌 GALAXY"
    else:
        return "💫 QSO"

# UI
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="u"),
        gr.Number(label="g"),
        gr.Number(label="r"),
        gr.Number(label="i"),
        gr.Number(label="z"),
        gr.Number(label="redshift")
    ],
    outputs="text",
    title="🌌 Astronomy Classifier"
)

demo.launch()