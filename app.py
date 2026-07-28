import gradio as gr
import numpy as np
import tensorflow as tf
import joblib
import os

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "breast_cancer_model.h5")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "breast_cancer_scaler.pkl")
)

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

css = """
.input-scroll{
    display:flex;
    overflow-x:auto;
    overflow-y:hidden;
    gap:15px;
    padding:10px;
}

.input-scroll > *{
    min-width:220px;
    flex-shrink:0;
}
"""

with gr.Blocks(css=css, title="Breast Cancer Prediction") as demo:

    gr.Markdown("""
# 🩺 Breast Cancer Prediction System
### Deep Learning + TensorFlow
""")

    with gr.Row():

        with gr.Column(scale=4):

            with gr.Row(elem_classes="input-scroll"):

                inputs = [
                    gr.Number(label=f, value=0)
                    for f in features
                ]

            btn = gr.Button(
                "🔍 Predict",
                variant="primary"
            )

        with gr.Column(scale=2):

            output = gr.Markdown("## Prediction")

    btn.click(
        predict,
        inputs=inputs,
        outputs=output
    )
    
port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
