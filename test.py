import speech_recognition as sr

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something...")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    try:
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
        text = recognizer.recognize_google(audio)
        print("Recognized:", text)
    except Exception as e:
        print("Error:", e)
