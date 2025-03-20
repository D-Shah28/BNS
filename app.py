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

# Load YOLO model (Upgraded to YOLOv5m for better accuracy)
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

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
    print("Stopping all processes...")
    for proc in [navigation_proc, object_detection_proc, voice_command_proc]:
        if proc and proc.is_alive():
            proc.terminate()
    print("All processes stopped successfully.")

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
                print(f"Recognized command: {command}")

                if "stop navigation" in command:
                    command_queue.put("STOP")
                elif "start navigation" in command:
                    command_queue.put("START")
                else:
                    command_queue.put(command)

            except sr.UnknownValueError:
                print("Could not understand voice.")
            except sr.RequestError:
                print("Error with speech recognition.")

# Function for real-time GPS navigation
def navigation_process(target_location, command_queue):
    last_alert_time = 0  

    while True:
        if not command_queue.empty():
            command = command_queue.get()
            if command == "STOP":
                print("Navigation stopped.")
                engine.say("Navigation stopped.")
                engine.runAndWait()
                return

        current_location = get_current_location()
        if not current_location:
            print("GPS Error: Unable to fetch location.")
            continue

        distance = geodesic(current_location, target_location).meters

        if distance < 3:
            print("You have reached your destination.")
            engine.say("You have arrived at your destination.")
            engine.runAndWait()
            return

        if time.time() - last_alert_time > 5:
            if distance > 50:
                direction = "Move Forward"
            elif distance > 30:
                direction = "Slight Left"
            elif distance > 10:
                direction = "Slight Right"
            else:
                direction = "Stop and check surroundings"

            print(f"Distance: {distance:.2f}m | {direction}")
            engine.say(f"{direction}, {int(distance)} meters remaining.")
            engine.runAndWait()
            last_alert_time = time.time()

        time.sleep(2)

# Function for real-time object detection
def object_detection_process(command_queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error accessing webcam.")
        return

    last_object_alert = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        detected_objects = []

        for result in results.xyxy[0]:
            try:
                x1, y1, x2, y2, conf, cls = result[:6]
                if conf < 0.3:
                    continue

                label = model.names[int(cls)]
                object_distance = max(10, int((x2 - x1) / 2))  
                detected_objects.append((label, object_distance))

            except Exception as e:
                print(f"Detection Error: {e}")

        if detected_objects and time.time() - last_object_alert > 3:
            objects_text = ", ".join([f"{obj[0]} {obj[1]} cm ahead" for obj in detected_objects])
            print(f"Caution: {objects_text}")
            engine.say(f"Caution: {objects_text}")
            engine.runAndWait()
            last_object_alert = time.time()

        if not command_queue.empty():
            command = command_queue.get()
            if command == "STOP":
                print("Stopping object detection.")
                cap.release()
                return

        time.sleep(1)

# Function to get destination from user
def get_destination():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Where do you want to go?")
        engine.say("Where do you want to go?")
        engine.runAndWait()
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5)
            destination = recognizer.recognize_google(audio).lower()
            st.success(f"✅ Recognized: {destination}")
            return destination
        except sr.UnknownValueError:
            st.error("⚠️ Could not understand destination.")
        except sr.RequestError:
            st.error("⚠️ Speech recognition error.")

    return None

# Function to start the navigation and object detection processes
def start_navigation():
    destination = get_destination()
    if not destination:
        return

    matched_location = difflib.get_close_matches(destination.lower(), locations.keys(), n=1, cutoff=0.3)
    if not matched_location:
        st.error("Invalid destination. Please try again.")
        engine.say("Invalid destination. Please try again.")
        engine.runAndWait()
        return

    target_location = locations[matched_location[0]]
    
    st.success(f"📍 Navigating to {matched_location[0]}")
    engine.say(f"Navigating to {matched_location[0]}")
    engine.runAndWait()

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

# Ensure multiprocessing works correctly on Windows
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
