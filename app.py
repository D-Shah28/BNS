import streamlit as st
import torch
import cv2
import os
import json
import numpy as np
import geopy.distance
import requests
from gtts import gTTS
import tempfile

# Load YOLO model (Streamlit-compatible version)
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)
    return model

model = load_model()

# Load university locations from GeoJSON
with open("university_map.geojson", "r", encoding="utf-8") as file:
    locations = json.load(file)

# Get user location (simulated for Streamlit)
def get_current_location():
    return (28.7041, 77.1025)  # Default: New Delhi, India

# Function to calculate distance
def calculate_distance(coord1, coord2):
    return geopy.distance.geodesic(coord1, coord2).meters

# Text-to-Speech (TTS) function
def speak(text):
    with tempfile.NamedTemporaryFile(delete=True) as temp_audio:
        tts = gTTS(text=text, lang="en")
        tts.save(temp_audio.name + ".mp3")
        os.system(f"mpg321 {temp_audio.name}.mp3")  # Works on Streamlit Cloud

# Detect objects using YOLO model
def detect_objects():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        detections = results.pandas().xyxy[0]
        for _, row in detections.iterrows():
            label = row['name']
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        st.image(frame, channels="BGR")
    cap.release()

# Main Streamlit App
st.title("Blind Navigation System")

# Get destination from user
destination = st.text_input("Where do you want to go?")
if st.button("Start Navigation"):
    found = False
    for feature in locations["features"]:
        if destination.lower() in feature["properties"]["name"].lower():
            dest_coords = feature["geometry"]["coordinates"][::-1]
            found = True
            break
    if not found:
        st.error("Location not found.")
    else:
        current_coords = get_current_location()
        distance = calculate_distance(current_coords, dest_coords)
        st.success(f"Navigating to {destination} ({distance:.2f} meters away)")
        speak(f"Navigating to {destination}. Distance is {int(distance)} meters.")

# Start Object Detection
if st.button("Start Object Detection"):
    detect_objects()

# Stop Navigation
if st.button("Stop Navigation"):
    st.warning("Navigation Stopped.")
    speak("Navigation stopped.")
