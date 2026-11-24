import os
import subprocess
import sys
import urllib.request
import joblib
import numpy as np
import onnxruntime as ort
from PIL import Image
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GDRIVE_FILE_ID = "1Aioeu070wGZz1opgAzXshTWL3rqK58_Y"
SVM_MODEL_PATH = os.path.join(BASE_DIR, "crop_disease_svm.pkl")


def _looks_like_html(file_path: str) -> bool:
  try:
    with open(file_path, "rb") as f:
      head = f.read(512).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")
  except OSError:
    return True


def ensure_svm_model_downloaded(dest_path: str, file_id: str) -> None:
  if (
      os.path.exists(dest_path)
      and os.path.getsize(dest_path) > 10 * 1024 * 1024
      and not _looks_like_html(dest_path)
  ):
    return

  cmd = [
      sys.executable,
      "-m",
      "gdown",
      "--fuzzy",
      f"https://drive.google.com/uc?id={file_id}",
      "-O",
      dest_path,
  ]
  subprocess.run(cmd, check=False)

  if not os.path.exists(dest_path) or _looks_like_html(dest_path):
    download_url = (
        f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    )
    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with (
        urllib.request.urlopen(req, timeout=180) as response,
        open(dest_path, "wb") as out_file,
    ):
      while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
          break
        out_file.write(chunk)

  if _looks_like_html(dest_path):
    raise RuntimeError(
        "Google Drive থেকে HTML পেজ ডাউনলোড হয়েছে; ফাইলের শেয়ারিং 'Anyone"
        " with the link' (Viewer) আছে কি না যাচাই করুন।"
    )


ensure_svm_model_downloaded(SVM_MODEL_PATH, GDRIVE_FILE_ID)

ort_session = ort.InferenceSession(
    os.path.join(BASE_DIR, "efficientnet.onnx"),
    providers=["CPUExecutionProvider"],
)
input_name = ort_session.get_inputs()[0].name
svm_pipeline = joblib.load(SVM_MODEL_PATH)
class_names = joblib.load(os.path.join(BASE_DIR, "class_names.pkl"))


def predict_crop(img: Image.Image) -> dict:
  if img is None:
    return {}
  try:
    resized = img.convert("RGB").resize((224, 224))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    inp = np.expand_dims(arr, axis=0)
    features = ort_session.run(None, {input_name: inp})[0]
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
      features = np.expand_dims(features, axis=0)
    probs = svm_pipeline.predict_proba(features)[0]
    return {
        str(class_names[i]): float(probs[i]) for i in range(len(class_names))
    }
  except Exception as e:
    return {"Error": 0.0, str(e)[:120]: 1.0}


demo = gr.Interface(
    fn=predict_crop,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Crop Disease Detection",
    description="EfficientNetB0 (ONNX) + StandardScaler + RBF SVM",
)

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 7860))
  demo.launch(server_name="0.0.0.0", server_port=port)
