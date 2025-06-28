import gradio as gr
import cv2
import numpy as np
import pyttsx3
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# Firebase setup
cred = credentials.Certificate(r"C:\Users\sneha\Downloads\slm-1-f370b-firebase-adminsdk-fbsvc-196ee42d05.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load YOLO
yolo_model = YOLO("yolov8n.pt")

# Constants
DESCRIPTIVE_CLASSES = ["chair", "sofa", "tv", "bed", "potted plant", "dining table"]
OBSTACLE_CLASSES = ["person", "chair", "sofa", "bed", "dining table"]
activity_log = []

# Log to Firebase
def log_to_firebase(entry):
    doc_ref = db.collection("activity_logs").document()
    doc_ref.set({
        "timestamp": firestore.SERVER_TIMESTAMP,
        "entry": entry
    })

# Frame processing
def process_frame(image, mode):
    if image is None:
        return None, "No image provided.", ""

    image = image.convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    try:
        results = yolo_model(frame)[0]
    except Exception as e:
        return image, f"YOLO error: {str(e)}", ""

    labels = [yolo_model.names[int(cls)] for cls in results.boxes.cls]
    label_counts = {label: labels.count(label) for label in set(labels)}

    # Create description
    if mode == "Describe Room":
        filtered = {k: v for k, v in label_counts.items() if k in DESCRIPTIVE_CLASSES}
        if filtered:
            desc = ", ".join([f"{v} {k}(s)" for k, v in filtered.items()])
            output_text = f"Room contains: {desc}."
        else:
            output_text = "No describable furniture detected."
    elif mode == "Obstacle Warning":
        filtered = {k: v for k, v in label_counts.items() if k in OBSTACLE_CLASSES}
        if filtered:
            desc = ", ".join([f"{v} {k}(s)" for k, v in filtered.items()])
            output_text = f"Warning: Obstacles ahead - {desc}."
        else:
            output_text = "No obstacles detected."
    else:
        output_text = "Unknown mode."

    # Text to speech
    try:
        engine.say(output_text)
        engine.runAndWait()
    except:
        output_text += " (TTS failed)"

    # Log and return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {output_text}"
    activity_log.append(log_entry)
    log_to_firebase(log_entry)

    return image, output_text, "\n".join(activity_log[-10:])  # last 10

# Gradio UI
demo = gr.Interface(
    fn=process_frame,
    inputs=[
        gr.Image(label="Upload or Webcam Frame", type="pil", sources=["upload", "webcam"]),
        gr.Radio(["Describe Room", "Obstacle Warning"], label="Mode")
    ],
    outputs=[
        gr.Image(label="Processed Image"),
        gr.Textbox(label="Voice Output"),
        gr.Textbox(label="Activity Log (Recent)")
    ],
    title="ViZionary - Wearable Vision Aid (Prototype)",
    description="A prototype for a Smart Location Module (SLM) wearable device that describes rooms and warns about obstacles using AI and voice.",
)

demo.launch(share=False)
