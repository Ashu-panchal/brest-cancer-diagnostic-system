import gradio as gr
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("breast_cancer_model.h5")
scaler = joblib.load("breast_cancer_scaler.pkl")

features = [ 
    "Mean Radius","Mean Texture","Mean Perimeter","Mean Area","Mean Smoothness",
    "Mean Compactness","Mean Concavity","Mean Concave Points","Mean Symmetry","Mean Fractal Dimension",
    "Radius Error","Texture Error","Perimeter Error","Area Error","Smoothness Error",
    "Compactness Error","Concavity Error","Concave Points Error","Symmetry Error","Fractal Dimension Error",
    "Worst Radius","Worst Texture","Worst Perimeter","Worst Area","Worst Smoothness",
    "Worst Compactness","Worst Concavity","Worst Concave Points","Worst Symmetry","Worst Fractal Dimension"
]

def predict(*inputs):
    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)

    prob = float(model.predict(data, verbose=0)[0][0])

    if prob >= 0.5:
        result = "🔴 Malignant"
        confidence = prob * 100
    else:
        result = "🟢 Benign"
        confidence = (1 - prob) * 100

    return f"""
# {result}

### Confidence: **{confidence:.2f}%**
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Breast Cancer Prediction") as demo:

    gr.Markdown(
        """
# 🩺 Breast Cancer Prediction System
### Deep Learning + TensorFlow
"""
    )

    with gr.Row():
        with gr.Column():
            inputs = [
                gr.Number(label=f, value=0)
                for f in features
            ]

            btn = gr.Button("🔍 Predict", variant="primary")

        with gr.Column():
            output = gr.Markdown()

    btn.click(
        predict,
        inputs=inputs,
        outputs=output
    )

demo.launch()
