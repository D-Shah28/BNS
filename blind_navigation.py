import cv2
import torch
import pyttsx3
import speech_recognition as sr
from geopy.distance import geodesic
import json
import time
import difflib
import math
import numpy as np
import geocoder

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)

# Load YOLO model for object detection
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Load university locations from GeoJSON
with open("university_map.geojson", "r", encoding="utf-8") as file:
    university_map = json.load(file)

# Extract locations into a dictionary
locations = {}
for feature in university_map["features"]:
    name = feature["properties"].get("name", "").strip().lower()
    coordinates = feature["geometry"]["coordinates"]
    if name:
        locations[name] = tuple(reversed(coordinates))  # Convert (lat, lon)

# Function to get real-time GPS location
def get_current_location():
    """Fetches real-time GPS location using geocoder."""
    try:
        g = geocoder.ip('me')  # Get location from IP
        if g.latlng:
            return tuple(g.latlng)  # Returns (latitude, longitude)
    except Exception as e:
        print(f"⚠️ GPS Error: {e}")
    return None  # GPS failed

# Function to calculate bearing angle
def calculate_bearing(start, end):
    lat1, lon1 = math.radians(start[0]), math.radians(start[1])
    lat2, lon2 = math.radians(end[0]), math.radians(end[1])

    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing = math.degrees(math.atan2(x, y))

    return (bearing + 360) % 360  # Normalize

# Function to determine movement direction
def get_direction(previous_location, current_location, destination):
    """Determines movement direction based on GPS updates."""
    if not previous_location or not current_location:
        return "Calculating route..."

    bearing_to_dest = calculate_bearing(current_location, destination)
    movement_bearing = calculate_bearing(previous_location, current_location)
    bearing_difference = (bearing_to_dest - movement_bearing + 360) % 360

    if 0 <= bearing_difference < 30 or bearing_difference > 330:
        return "Move Forward"
    elif 30 <= bearing_difference < 150:
        return "Turn Right"
    elif 150 <= bearing_difference < 210:
        return "Move Backward"
    elif 210 <= bearing_difference < 330:
        return "Turn Left"
    else:
        return "Recalculating..."

# Predefined real-world widths of objects (in cm)
KNOWN_WIDTHS = {
    "person": 40,  
    "chair": 50,
    "car": 180,
    "bottle": 7,
    "cup": 8
}

FOCAL_LENGTH = 615  # Adjust based on testing

# Function to estimate distance based on object width
def estimate_distance(object_width_pixels, real_width_cm):
    return (FOCAL_LENGTH * real_width_cm) / object_width_pixels if object_width_pixels > 0 else -1  

# Function to detect objects
def detect_objects(frame, last_warning_time):
    if frame is None:
        return [], frame, last_warning_time

    results = model(frame)
    detected_objects = []
    current_time = time.time()

    for result in results.xyxy[0]:  
        x1, y1, x2, y2, confidence, class_id = result.tolist()
        label = model.names[int(class_id)]
        object_width_pixels = x2 - x1
        real_width_cm = KNOWN_WIDTHS.get(label, 30)
        distance = estimate_distance(object_width_pixels, real_width_cm)

        if confidence > 0.5:
            detected_objects.append(f"{label} {int(distance)} cm ahead")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(distance)} cm", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if current_time - last_warning_time > 5 and detected_objects:
        alert_message = "Caution! " + ", ".join(detected_objects)
        print(alert_message)
        engine.say(alert_message)
        engine.runAndWait()
        last_warning_time = current_time

    return detected_objects, frame, last_warning_time

# Function to listen for stop command
def listen_for_stop_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            print("Listening for 'Stop Navigation' command...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            return "stop navigation" in recognizer.recognize_google(audio).lower()
        except:
            return False

# Core navigation function
def blind_navigation(destination):
    if destination not in locations:
        engine.say("Invalid destination. Please try again.")
        engine.runAndWait()
        return

    target_location = locations[destination]
    print(f"✅ Navigating to {destination} → 📍 {target_location}")
    engine.say(f"Navigating to {destination}")
    engine.runAndWait()

    cap = cv2.VideoCapture(0)  # Open camera
    last_warning_time = time.time()
    last_location = get_current_location()
    
    while True:
        current_location = get_current_location()
        if not current_location:
            continue  

        distance = geodesic(current_location, target_location).meters

        if distance < 3:
            engine.say(f"You have reached {destination}.")
            engine.runAndWait()
            break

        direction = get_direction(last_location, current_location, target_location)

        print(f"📍 Distance: {distance:.2f}m | {direction}")
        engine.say(f"{direction}, {int(distance)} meters remaining.")
        engine.runAndWait()
        last_location = current_location

        ret, frame = cap.read()
        if ret:
            detected_objects, frame, last_warning_time = detect_objects(frame, last_warning_time)
            cv2.imshow("Blind Navigation", frame)

        if listen_for_stop_command():
            print("🛑 Stop command detected. Exiting navigation...")
            engine.say("Navigation stopped.")
            engine.runAndWait()
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break    

    cap.release()
    cv2.destroyAllWindows()

# Function to capture voice command
def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        engine.say("Where do you want to go?")
        engine.runAndWait()
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            return recognizer.recognize_google(audio).lower()
        except:
            return None

if __name__ == "__main__":
    destination = get_voice_command()
    if destination:
        matched_location = difflib.get_close_matches(destination.lower(), locations.keys(), n=1, cutoff=0.5)
        if matched_location:
            blind_navigation(matched_location[0])
        else:
            engine.say("Destination not recognized. Try again.")
            engine.runAndWait()
