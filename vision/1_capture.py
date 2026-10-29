"""
SCRIPT 1 - CAPTURE CALIBRATION PHOTOS
"""


import cv2
import os
import numpy as np
from pathlib import Path

# PARAMETERS
BASE_DIR = Path(__file__).resolve().parent
CAMERA_INDEX = 0          # iPhone index (adjust if needed)
SAVE_DIR = BASE_DIR / "photos_calibration"
CHESSBOARD_SIZE = (9, 6)  # internal corners: 9 columns, 6 rows
TARGET_PHOTOS = 30


os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".png")])
print(f"Camera opened. Photos already saved: {count}/{TARGET_PHOTOS}")
print("Commands: S = save | Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: unable to read the camera")
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Look for the chessboard in the live image
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        # Refine detection to sub-pixel accuracy
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        status = "CHESSBOARD DETECTED - press S to save"
        color = (0, 255, 0)
    else:
        status = "Chessboard not detected - show the full board"
        color = (0, 0, 255)

    # Display info
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display, f"Photos: {count}/{TARGET_PHOTOS}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Progress bar
    progress = int((count / TARGET_PHOTOS) * frame.shape[1])
    cv2.rectangle(display, (0, frame.shape[0]-20), (progress, frame.shape[0]), (0, 255, 0), -1)

    cv2.imshow("Calibration capture - iPhone", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        if found:
            path = os.path.join(SAVE_DIR, f"photo_{count:03d}.png")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  Saved: {path} ({count}/{TARGET_PHOTOS})")
            if count >= TARGET_PHOTOS:
                print("Target reached! You can quit with Q.")
        else:
            print("  Chessboard not detected, move it and try again.")

cap.release()
cv2.destroyAllWindows()
print(f"\nCapture finished: {count} photos saved in '{SAVE_DIR}/'")
