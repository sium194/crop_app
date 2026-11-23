import os
import gradio as gr
import joblib
import numpy as np
import onnxruntime as ort
from PIL import Image

svm_model = joblib.load("crop_disease_svm.pkl")
class_names = joblib.load("class_names.pkl")
session = ort.InferenceSession("efficientnet.onnx",
                               providers=["CPUExecutionProvider"])
IMG_SIZE = (224, 224)

def predict_crop_disease(image):
    if image is None:
        return "অনুগ্রহ করে একটি পাতার ছবি আপলোড করুন।"
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    features = session.run(None, {"input": arr})[0]
    pred = svm_model.predict(features)[0]
    try:
        label = class_names[int(pred)]
    except (ValueError, TypeError):
        label = str(pred)
    return f"Predicted Class: {label}"

demo = gr.Interface(
    fn=predict_crop_disease,
    inputs=gr.Image(type="pil", label="ফসলের পাতার ছবি আপলোড করুন"),
    outputs=gr.Textbox(label="ফলাফল"),
    title="Crop Disease Detection",
    description="ধান, আলু ও ভুট্টার পাতার রোগ শনাক্তকরণ (EfficientNetB0 + SVM)",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)))
