import os
import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

MODELS_DICT = {
    "Model 1": "../best_model.keras",
    "Model 2": "saved_models/best_model_exp_01_.keras"
}

LOADED_MODELS = {}
IMAGE_SIZE = (32, 32)
CLASS_NAMES = ['Clear', 'Cloudy', 'Fog'] 


def get_model(model_name):
    model_path = MODELS_DICT[model_name]
    if model_path not in LOADED_MODELS:
        print(f"Loading model from: {model_path}...")
        LOADED_MODELS[model_path] = load_model(model_path)
    return LOADED_MODELS[model_path]


def predict_fog(image, model_name):
    if image is None or model_name is None:
        return None

    model = get_model(model_name)
    data = cv2.resize(image, IMAGE_SIZE)
    data = np.array(data, dtype='float32') / 255.0
    data = np.expand_dims(data, axis=0)
    
    predictions = model.predict(data)[0]
    
    result = {}
    
    for i in range(len(CLASS_NAMES)):
        result [CLASS_NAMES[i]] = float(predictions[i])
    return result



def shutdown():
    print("\n[INFO] Working was terminated by user")
    os._exit(0)


with gr.Blocks(title="The fog detection system") as demo:
    gr.Markdown("# The fog detection system")
    gr.Markdown("Please upload an image amd select a model for analysis")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload image")
            model_dropdown = gr.Dropdown(
                choices=list(MODELS_DICT.keys()), 
                value=list(MODELS_DICT.keys())[0], 
                label="Select a model"
            )
            btn_predict = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Analysis result")
    
    gr.Markdown("---")
    btn_stop = gr.Button(" Terminate the application", variant="stop")

    btn_predict.click(
        fn=predict_fog, 
        inputs=[input_image, model_dropdown], 
        outputs=output_label
    )
    
    btn_stop.click(fn=shutdown, inputs=None, outputs=None)


if __name__ == "__main__":
    demo.launch()