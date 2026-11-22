import gradio as gr
import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load saved SVM model and class names
svm_model = joblib.load("crop_disease_svm.pkl")
class_names = joblib.load("class_names.pkl")

# Load EfficientNetB0 feature extractor
IMG_SIZE = (224, 224)
feature_extractor = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)


def predict_crop_disease(image):
    if image is None:
        return "অনুগ্রহ করে একটি পাতার ছবি আপলোড করুন।"

    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    features = feature_extractor.predict(arr, verbose=0)
    prediction = svm_model.predict(features)[0]

    return f"Predicted Class: {prediction}"


demo = gr.Interface(
    fn=predict_crop_disease,
    inputs=gr.Image(type="pil", label="ফসলের পাতার ছবি আপলোড করুন"),
    outputs=gr.Textbox(label="ফলাফল"),
    title="Crop Disease Detection",
    description="EfficientNetB0 + SVM ব্যবহার করে ফসলের পাতার রোগ শনাক্তকরণ অ্যাপ।",
)

if __name__ == "__main__":
    demo.launch()