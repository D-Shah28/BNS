import streamlit as st
import cv2
import torch
import pyttsx3
import speech_recognition as sr
import json
import numpy as np
import geocoder
import difflib
import time
import multiprocessing
from geopy.distance import geodesic
import os
import signal

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty("rate", 160)

# Load YOLO model (Optimized for Render Deployment)
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "yolov5m", force_reload=True)

model = load_model()

# Load university locations from GeoJSON
with open("university_map.geojson", "r", encoding="utf-8") as file:
    university_map = json.load(file)

locations = {}
for feature in university_map["features"]:
    properties = feature.get("properties", {})
    name = properties.get("name", "").strip().lower()
    if name:
        locations[name] = tuple(reversed(feature["geometry"]["coordinates"]))

# Function to get current GPS location
def get_current_location():
    g = geocoder.ip("me")
    return g.latlng if g.latlng else None

# Function to stop all processes
def stop_all_processes(navigation_proc, object_detection_proc, voice_command_proc):
    for proc in [navigation_proc, object_detection_proc, voice_command_proc]:
        if proc and proc.is_alive():
            proc.terminate()

# Function to recognize voice commands
def listen_for_commands(command_queue):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            recognizer.adjust_for_ambient_noise(source)
            try:
                print("Listening for commands...")
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                if "stop navigation" in command:
                    command_queue.put("STOP")
                elif "start navigation" in command:
                    command_queue.put("START")
                else:
                    command_queue.put(command)
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass

# Function for real-time GPS navigation
def navigation_process(target_location, command_queue):
    last_alert_time = 0  
    while True:
        if not command_queue.empty():
            command = command_queue.get()
            if command == "STOP":
                return

        current_location = get_current_location()
        if not current_location:
            continue

        distance = geodesic(current_location, target_location).meters
        if distance < 3:
            return

        if time.time() - last_alert_time > 5:
            direction = "Move Forward" if distance > 50 else "Slight Left" if distance > 30 else "Slight Right" if distance > 10 else "Stop"
            last_alert_time = time.time()
        time.sleep(2)

# Function for real-time object detection
def object_detection_process(command_queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    last_object_alert = 0  
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        detected_objects = [(model.names[int(cls)], max(10, int((x2 - x1) / 2))) for x1, y1, x2, y2, conf, cls in results.xyxy[0] if conf > 0.3]

        if detected_objects and time.time() - last_object_alert > 3:
            last_object_alert = time.time()

        if not command_queue.empty():
            command = command_queue.get()
            if command == "STOP":
                cap.release()
                return
        time.sleep(1)

# Function to get destination from user
def get_destination():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Where do you want to go?")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            st.error("⚠️ Could not understand destination.")
        except sr.RequestError:
            st.error("⚠️ Speech recognition error.")
    return None

# Function to start navigation and object detection
def start_navigation():
    destination = get_destination()
    if not destination:
        return

    matched_location = difflib.get_close_matches(destination.lower(), locations.keys(), n=1, cutoff=0.3)
    if not matched_location:
        return

    target_location = locations[matched_location[0]]
    
    command_queue = multiprocessing.Queue()
    navigation_proc = multiprocessing.Process(target=navigation_process, args=(target_location, command_queue))
    object_detection_proc = multiprocessing.Process(target=object_detection_process, args=(command_queue,))
    voice_command_proc = multiprocessing.Process(target=listen_for_commands, args=(command_queue,))

    navigation_proc.start()
    object_detection_proc.start()
    voice_command_proc.start()

    navigation_proc.join()
    object_detection_proc.join()
    voice_command_proc.terminate()

# Main function
def main():
    st.title("Blind Navigation System")
    start_navigation()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
