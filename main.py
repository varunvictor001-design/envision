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
camera = cv2.VideoCapture(1)

while True:
    time.sleep(10)
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
ser.close()