import gradio as gr
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
from PIL import Image

# Initialize the TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# List of common objects to describe
DESCRIPTIVE_CLASSES = ["chair", "sofa", "tv", "bed", "potted plant", "dining table"]
OBSTACLE_CLASSES = ["person", "chair", "sofa", "bed", "dining table"]


def process_frame(image, mode):
    if image is None:
        return None, "No image provided."

    # Convert to RGB to ensure compatibility
    image = image.convert("RGB")

    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run YOLOv8 object detection
    results = yolo_model(frame)[0]

    # Extract labels
    labels = [yolo_model.names[int(cls)] for cls in results.boxes.cls]
    label_counts = {label: labels.count(label) for label in set(labels)}

    # Construct description
    if mode == "Describe Room":
        filtered = {k: v for k, v in label_counts.items() if k in DESCRIPTIVE_CLASSES}
        if filtered:
            description = ", ".join([f"{v} {k}(s)" for k, v in filtered.items()])
            output_text = f"Room contains: {description}."
        else:
            output_text = "No describable furniture detected."

    elif mode == "Obstacle Warning":
        filtered = {k: v for k, v in label_counts.items() if k in OBSTACLE_CLASSES}
        if filtered:
            description = ", ".join([f"{v} {k}(s)" for k, v in filtered.items()])
            output_text = f"Warning: Obstacles ahead - {description}."
        else:
            output_text = "No obstacles detected."
    else:
        output_text = "Unknown mode."

    # Speak the result
    engine.say(output_text)
    engine.runAndWait()

    # Return image and spoken text
    return image, output_text


# Gradio interface
demo = gr.Interface(
    fn=process_frame,
    inputs=[
        gr.Image(label="Upload Webcam Snapshot or Live Feed Frame", type="pil", sources=["upload", "webcam"]),
        gr.Radio(["Describe Room", "Obstacle Warning"], label="Mode")
    ],
    outputs=[
        gr.Image(label="Processed Image"),
        gr.Textbox(label="Detected Voice Output")
    ],
    title="ViZionary",
    description="Upload a webcam frame to detect surroundings and get voice assistance.",
    theme="default"
)

demo.launch(share=False)
