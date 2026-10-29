"""
SCRIPT 3 - OBJECT DETECTION AND POSITION IN CM
Run this script after calibration.
It detects a colored object in the image and gives its position (x, y) in cm
relative to the center of the work area.
"""

import cv2
import numpy as np
from pathlib import Path

# PARAMETERS
BASE_DIR = Path(__file__).resolve().parent
CAMERA_INDEX = 0
PARAMS_FILE = BASE_DIR / "camera_params.npz"
CAMERA_HEIGHT_CM = 40.0    # ← height of the camera above the table in cm (to be measured)

# Color of the object to detect in HSV
# Red by default — change these values if your object is a different color
# Useful tool to find the values: https://github.com/harishtpj/ColorPicker
HSV_LOW  = np.array([0,   120, 70])   # lower bound
HSV_HIGH = np.array([10, 255, 255])   # upper bound
# For blue:  HSV_LOW=[100,120,70]  HSV_HIGH=[130,255,255]
# For green: HSV_LOW=[40,50,50]    HSV_HIGH=[80,255,255]



# Load calibration parameters
try:
    data = np.load(PARAMS_FILE)
    K    = data["K"]
    dist = data["dist"]
    print("Camera calibration loaded.")
except FileNotFoundError:
    print(f"'{PARAMS_FILE}' not found. Run 2_calibration.py first")
    exit()

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Detection running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: unable to read the camera")
        break

    # Distortion correction
    frame = cv2.undistort(frame, K, dist)

    # Convert to HSV and build the color mask
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)

    # Morphological cleanup (erosion then dilation)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 300:
            M = cv2.moments(c)
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])

            # Convert pixels → cm in the camera frame (top-down view)
            Z = CAMERA_HEIGHT_CM
            X_cm = (u - cx) * Z / fx
            Y_cm = (v - cy) * Z / fy

            # Display
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.circle(frame, (u, v), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"Object: X={X_cm:.1f}cm  Y={Y_cm:.1f}cm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, "Q = quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Object detection - SO101", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection finished.")
