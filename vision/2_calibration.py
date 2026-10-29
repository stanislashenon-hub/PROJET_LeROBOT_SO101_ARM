"""
SCRIPT 2 - CAMERA CALIBRATION
After running script 1: computes the camera's optical parameters and saves them to camera_params.npz
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

# PARAMETERS
BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / "photos_calibration"
CHESSBOARD_SIZE = (9, 6)       # internal corners: 9 columns, 6 rows
CASE_SIZE_MM = 25.0            # ← CHANGE THIS VALUE to your actual measurement in mm
OUTPUT_FILE = BASE_DIR / "camera_params.npz"


photos = sorted(glob.glob(os.path.join(PHOTOS_DIR, "*.png")))
print(f"Photos found: {len(photos)}")

if len(photos) < 10:
    print("Not enough photos (minimum 10). Run 1_capture.py first")
    exit()

# Theoretical 3D points of the chessboard (in the board's own frame)
# E.g.: (0,0,0), (25,0,0), (50,0,0) ... in mm
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= CASE_SIZE_MM

objpoints = []  # real 3D points
imgpoints = []  # 2D points in the image
valid_photos = []

print("\nAnalyzing photos...")
for path in photos:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners)
        valid_photos.append(path)
        print(f"  ✓ {os.path.basename(path)}")
    else:
        print(f"  ✗ {os.path.basename(path)} — chessboard not detected, skipped")

print(f"\nValid photos: {len(valid_photos)}/{len(photos)}")

if len(valid_photos) < 10:
    print("Not enough valid photos. Redo the capture with script 1.")
    exit()

# Calibration
print("\nCalibrating...")
h, w = cv2.imread(valid_photos[0]).shape[:2]
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

print(f"\nReprojection error: {ret:.4f} px  (good if < 1.0)")
print(f"\nIntrinsic matrix K:")
print(f"  Focal x (fx): {K[0,0]:.1f} px")
print(f"  Focal y (fy): {K[1,1]:.1f} px")
print(f"  Optical center: ({K[0,2]:.1f}, {K[1,2]:.1f}) px")
print(f"\nDistortion coefficients:")
print(f"  {dist.ravel()}")

# Save
np.savez(OUTPUT_FILE, K=K, dist=dist, img_size=(w, h), case_size_mm=CASE_SIZE_MM)
print(f"\nParameters saved to '{OUTPUT_FILE}'")

# Visualize the distortion on a sample image
sample = cv2.imread(valid_photos[0])
undistorted = cv2.undistort(sample, K, dist)
cv2.imshow("Original", sample)
cv2.imshow("Corrected (undistort)", undistorted)
print("\nDisplaying: original vs corrected. Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
