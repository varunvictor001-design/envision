"""  import cv2
import numpy as np
import serial
import time
from keras.models import load_model

# ======== CONFIGURATION ========
ESP32_PORT = "COM3"        # Change to your ESP32 port
BAUD_RATE = 115200
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
# ===============================

# Load the Keras model and labels
model = load_model(MODEL_PATH, compile=False)
class_names = open(LABELS_PATH, "r").readlines()
 
# Open the serial connection to ESP32
try:
    ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
    time.sleep(1)  # Wait for ESP32 reset
    print(f"[INFO] Connected to ESP32 on {ESP32_PORT}")
except serial.SerialException as e:
    print(f"[ERROR] Could not open port {ESP32_PORT}: {e}")
    exit()

# Start webcam
camera = cv2.VideoCapture(0)

while True:
    
    ret, frame = camera.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Keep a copy for display
    display_frame = frame.copy()

    # Resize frame for model
    model_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    model_frame = np.asarray(model_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    model_frame = (model_frame / 127.5) - 1

    # Predict
    prediction = model.predict(model_frame)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Show in Python terminal
    print(f"[PC] Class: {class_name} | Confidence: {confidence_score*100:.2f}%")

    # Send to ESP32
    ser.write(f"{class_name}\n".encode())

    # Read ESP32 response
    if ser.in_waiting > 0:
        esp_msg = ser.readline().decode(errors="ignore").strip()
        if esp_msg:
            print(f"[ESP32] {esp_msg}")

    # Display webcam
    cv2.putText(display_frame, f"{class_name} ({confidence_score*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Stream", display_frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()
ser.close() """

""" import cv2

# Replace with your phone’s IP address shown in the app
url = "http://192.168.29.66:8080/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open stream")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # ... run your AI here ...
    cv2.imshow("Phone Stream", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows() """
""" 
import cv2
import numpy as np
import serial
import time
from keras.models import load_model

# ======== CONFIGURATION ========
ESP32_PORT = "COM3"        # Change to your ESP32 port
BAUD_RATE = 115200
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
# ===============================

# Load the Keras model and labels
model = load_model(MODEL_PATH, compile=False)
class_names = open(LABELS_PATH, "r").readlines()
 
# Open the serial connection to ESP32
try:
    ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
    time.sleep(1)  # Wait for ESP32 reset
    print(f"[INFO] Connected to ESP32 on {ESP32_PORT}")
except serial.SerialException as e:
    print(f"[ERROR] Could not open port {ESP32_PORT}: {e}")
    exit()

# Start webcam
url = "http://192.168.20.80:8080/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open stream")
    exit(1)
while True:
    time.sleep(10)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Keep a copy for display
    display_frame = frame.copy()

    # Resize frame for model
    model_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    model_frame = np.asarray(model_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    model_frame = (model_frame / 127.5) - 1

    # Predict
    prediction = model.predict(model_frame)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Show in Python terminal
    print(f"[PC] Class: {class_name} | Confidence: {confidence_score*100:.2f}%")

    # Send to ESP32
    ser.write(f"{class_name}\n".encode())

    # Read ESP32 response
    if ser.in_waiting > 0:
        esp_msg = ser.readline().decode(errors="ignore").strip()
        if esp_msg:
            print(f"[ESP32] {esp_msg}")

    # Display webcam
    cv2.putText(display_frame, f"{class_name} ({confidence_score*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Stream", display_frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
ser.close() """

"""
from keras.models import load_model
import cv2
import numpy as np
import serial
from twilio.rest import Client
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------
# 🔑 Twilio Credentials
ACCOUNT_SID = "YOUR ACCOUNT SID"
AUTH_TOKEN = "YOUR AUTH TOKEN"
TWILIO_NUMBER = "+17652663033"
MY_NUMBER = "+91 9790755971"

client = Client(ACCOUNT_SID, AUTH_TOKEN)
# ----------------------------

np.set_printoptions(suppress=True)

# ======== CONFIGURATION ========
ESP32_PORT = "COM17"
BAUD_RATE = 115200
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
CONF_THRESHOLD = 0.8
STAY_THRESHOLD = 15   # seconds → sine wave
ALERT_EXTRA = 5       # extra seconds after sine wave → SMS
# ===============================

# Frequency mapping per animal
animal_frequencies = {
    "elephant": 800,
    "tiger": 1000,
    "lion": 1200,
    "leopard": 1500,
    "cheetah": 2000
}

# Load the Keras model and labels
model = load_model(MODEL_PATH, compile=False)
class_names = open(LABELS_PATH, "r").readlines()

# Open the serial connection to ESP32
try:
    ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
    time.sleep(1)  # Wait for ESP32 reset
    print(f"[INFO] Connected to ESP32 on {ESP32_PORT}")
except serial.SerialException as e:
    print(f"[ERROR] Could not open port {ESP32_PORT}: {e}")
    exit()

camera = cv2.VideoCapture(0)

# To prevent spamming SMS
last_alert_time = 0
alert_cooldown = 30  # seconds

# To track continuous detection
last_class = None
class_start_time = 0
sine_wave_triggered = False  # Did we already play sine wave?

import sounddevice as sd

# ==== Function: Show sine wave for frequency (graph + sound) ====
def show_sine_wave(freq):
    duration = 5        # seconds per run
    sample_rate = 44100 # CD-quality sound

    # Generate sine wave for sound
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y_audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Generate smaller sample for plotting (to avoid heavy graph)
    t_plot = np.linspace(0, duration, 1000)
    y_plot = np.sin(2 * np.pi * freq * t_plot)

    for i in range(2):  # repeat twice
        # 🎵 Play sound
        print(f"[SOUND] Playing {freq} Hz ({i+1}/2)")
        sd.play(y_audio, samplerate=sample_rate, blocking=False)

        # 📊 Plot graph
        plt.figure(figsize=(8,4))
        plt.plot(t_plot, y_plot, color='blue', linewidth=2)
        plt.title(f"Sine Wave @ {freq} Hz (Run {i+1}/2)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.ylim(-1.2, 1.2)

        plt.show(block=False)
        plt.pause(duration)     # keep visible for 5 sec
        plt.close()

        sd.stop()               # stop sound
        time.sleep(1)           # 1-sec gap before next run

# ===========================
while True:
    time.sleep(1)
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame from camera")
        break

    display_frame = frame.copy()

    model_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    model_frame = np.asarray(model_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    model_frame = (model_frame / 127.5) - 1

    prediction = model.predict(model_frame)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    print(f"[PC] Class: {class_name} | Confidence: {confidence_score*100:.2f}%")

    # Send to ESP32
    ser.write(f"{class_name}\n".encode())

    # Read ESP32 response
    if ser.in_waiting > 0:
        esp_msg = ser.readline().decode(errors="ignore").strip()
        if esp_msg:
            print(f"[ESP32] {esp_msg}")

    cv2.putText(display_frame, f"{class_name} ({confidence_score*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam Stream", display_frame)
    cv2.moveWindow("Webcam Stream", 100, 100)

    # ===============================
    # 🐘 Detect if animal stays long
    # ===============================
    if confidence_score > CONF_THRESHOLD:
        if class_name == last_class:
            elapsed = time.time() - class_start_time

            # Trigger sine wave after STAY_THRESHOLD
            if elapsed > STAY_THRESHOLD and not sine_wave_triggered:
                freq = animal_frequencies.get(class_name, None)
                if freq:
                    print(f"[JUMPSCARE] {class_name} stayed >{STAY_THRESHOLD}s → Playing {freq} Hz sine wave!")
                    show_sine_wave(freq)
                    sine_wave_triggered = True  # remember we already did it

            # After sine wave + extra time → send SMS
            if sine_wave_triggered and elapsed > (STAY_THRESHOLD + ALERT_EXTRA):
                if (time.time() - last_alert_time) > alert_cooldown:
                    message = client.messages.create(
                        body=f"⚠ ALERT: {class_name} detected for over {STAY_THRESHOLD+ALERT_EXTRA}s with {confidence_score*100:.1f}% confidence!",
                        to=MY_NUMBER,
                        messaging_service_sid='MGc2e86eb42660c5811f4dc9ddba804207'
                    )
                    print("SMS sent! SID:", message.sid)
                    last_alert_time = time.time()
                    # Reset after sending
                    sine_wave_triggered = False
                    class_start_time = time.time()

        else:
            last_class = class_name
            class_start_time = time.time()
            sine_wave_triggered = False  # reset for new animal

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

camera.release()
cv2.destroyAllWindows()
 """

import cv2
import numpy as np
import serial
import time
import sounddevice as sd
import matplotlib.pyplot as plt
from keras.models import load_model
from twilio.rest import Client

# ----------------------------
# 🔑 Twilio Credentials (replace with env vars in production!)
ACCOUNT_SID = "YOUR ACCOUNT SID"
AUTH_TOKEN = "YOUR AUTH TOKEN"
TWILIO_NUMBER = "+17652663033"
MY_NUMBER = "+91 9790755971"
client = Client(ACCOUNT_SID, AUTH_TOKEN)
# ----------------------------

# ======== CONFIGURATION ========
ESP32_PORT = "COM17"       # your ESP32 COM port
BAUD_RATE = 115200
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
CONF_THRESHOLD = 0.8
STAY_THRESHOLD = 5        # seconds before sine wave
ALERT_EXTRA = 5            # seconds after sine wave → SMS
ALERT_COOLDOWN = 30        # seconds between SMS
# ===============================

# Animal → frequency mapping
animal_frequencies = {
    "elephant": 800,
    "tiger": 1000,
    "lion": 1200,
    "leopard": 1500,
    "cheetah": 2000
}

# Load AI model + labels
model = load_model(MODEL_PATH, compile=False)
class_names = open(LABELS_PATH, "r").readlines()

# Serial connection to ESP32
try:
    ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
    time.sleep(1)
    print(f"[INFO] Connected to ESP32 on {ESP32_PORT}")
except serial.SerialException as e:
    print(f"[ERROR] Could not open port {ESP32_PORT}: {e}")
    exit()

# Camera (set to 0 for webcam, or replace with IP stream)
CAMERA_URL = 0 # change to "http://<phone-ip>:8080/video" for IP webcam
cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    print("[ERROR] Failed to open camera stream")
    exit()

# Tracking states
last_alert_time = 0
last_class = None
class_start_time = 0
sine_wave_triggered = False

# ==== Function: Show sine wave (graph + sound) ====
def show_sine_wave(freq, duration=5):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y_audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Smaller sample for graph
    t_plot = np.linspace(0, duration, 1000)
    y_plot = np.sin(2 * np.pi * freq * t_plot)

    for i in range(2):  # repeat twice
        print(f"[SOUND] Playing {freq} Hz ({i+1}/2)")
        sd.play(y_audio, samplerate=sample_rate, blocking=False)

        plt.figure(figsize=(8, 4))
        plt.plot(t_plot, y_plot, color='blue', linewidth=2)
        plt.title(f"Sine Wave @ {freq} Hz (Run {i+1}/2)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.ylim(-1.2, 1.2)

        plt.show(block=False)
        plt.pause(duration)
        plt.close()

        sd.stop()
        time.sleep(1)

# ===========================
# 🔴 MAIN LOOP
# ===========================
while True:
    time.sleep(1)
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera")
        break

    display_frame = frame.copy()

    model_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    model_frame = np.asarray(model_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    model_frame = (model_frame / 127.5) - 1

    prediction = model.predict(model_frame)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    print(f"[PC] Class: {class_name} | Confidence: {confidence_score*100:.2f}%")

    # Send to ESP32
    ser.write(f"{class_name}\n".encode())

    # Read ESP32 response
    if ser.in_waiting > 0:
        esp_msg = ser.readline().decode(errors="ignore").strip()
        if esp_msg:
            print(f"[ESP32] {esp_msg}")

    cv2.putText(display_frame, f"{class_name} ({confidence_score*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam Stream", display_frame)
    cv2.moveWindow("Webcam Stream", 100, 100)

    # ===============================
    # 🐘 Detect if animal stays long
    # ===============================
    if confidence_score > CONF_THRESHOLD:
        if class_name == last_class:
            elapsed = time.time() - class_start_time

            # Trigger sine wave after STAY_THRESHOLD
            if elapsed > STAY_THRESHOLD and not sine_wave_triggered:
                freq = animal_frequencies.get(class_name, None)
                if freq:
                    print(f"[JUMPSCARE] {class_name} stayed >{STAY_THRESHOLD}s → Playing {freq} Hz sine wave!")
                    show_sine_wave(freq)
                    sine_wave_triggered = True  # remember we already did it

            # After sine wave + extra time → send SMS
            if sine_wave_triggered and elapsed > (STAY_THRESHOLD + ALERT_EXTRA):
                if (time.time() - last_alert_time) > ALERT_COOLDOWN:
                    message = client.messages.create(
                        body=f"⚠ ALERT: {class_name} detected for over {STAY_THRESHOLD+ALERT_EXTRA}s with {confidence_score*100:.1f}% confidence!",
                        to=MY_NUMBER,
                        messaging_service_sid='MGc2e86eb42660c5811f4dc9ddba804207'
                    )
                    print("SMS sent! SID:", message.sid)
                    last_alert_time = time.time()
                    # Reset after sending
                    sine_wave_triggered = False
                    class_start_time = time.time()

        else:
            last_class = class_name
            class_start_time = time.time()
            sine_wave_triggered = False  # reset for new animal

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()
