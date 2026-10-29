"""
SCRIPT 4 - HAND-EYE CALIBRATION
Automatically links the iPhone camera frame to the SO-101 robot frame.
The robot moves to several known positions while the camera observes
a colored marker on the gripper.

BEFORE RUNNING:
- Camera calibration must be done (camera_params.npz must exist)
- The robot must be connected on /dev/ttyACM1
- Place a colored object (red or other) ON the robot's gripper
- The iPhone must see the entire work area
"""

import cv2
import numpy as np
import time
from pathlib import Path
from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig as SO101FollowerConfig

# PARAMETERS
BASE_DIR = Path(__file__).resolve().parent
CAMERA_INDEX    = 0
PARAMS_FILE     = BASE_DIR / "camera_params.npz"
OUTPUT_FILE     = BASE_DIR / "handeye_params.npz"
ROBOT_PORT      = "/dev/ttyACM4"
CAMERA_HEIGHT_CM = 30.0   # iPhone height above the table in cm

# Color of the marker on the gripper (red by default)
HSV_LOW  = np.array([0, 100, 150])
HSV_HIGH = np.array([15, 255, 255])

# Known robot positions for calibration (angles in degrees)
# The robot will move to each of these positions, and we detect the gripper in the image
CALIBRATION_POSES = [
    # Safe positions based on the robot's actual rest position
    # measured rest: shoulder_pan=-1.6, shoulder_lift=-99.7, elbow_flex=99.0, wrist_flex=27.9
    # Keep the arm folded (shoulder_lift low, elbow_flex high) and vary shoulder_pan
    {'shoulder_pan':   0, 'shoulder_lift': -90, 'elbow_flex': 90, 'wrist_flex': 30, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan':  25, 'shoulder_lift': -90, 'elbow_flex': 90, 'wrist_flex': 30, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan': -25, 'shoulder_lift': -90, 'elbow_flex': 90, 'wrist_flex': 30, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan':  15, 'shoulder_lift': -80, 'elbow_flex': 85, 'wrist_flex': 25, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan': -15, 'shoulder_lift': -80, 'elbow_flex': 85, 'wrist_flex': 25, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan':   0, 'shoulder_lift': -85, 'elbow_flex': 88, 'wrist_flex': 28, 'wrist_roll': 0, 'gripper': 10},
]


# SO-101 forward kinematics (from the official URDF)
def Rx(deg):
    r = np.deg2rad(deg)
    return np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])

def Ry(deg):
    r = np.deg2rad(deg)
    return np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])

def Rz(deg):
    r = np.deg2rad(deg)
    return np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])

def make_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def forward_kinematics(pose):
    """Returns the gripper position (x,y,z) in meters in the robot frame."""
    T_w1 = make_T(Rz(180) @ Rx(180) @ Rz(pose['shoulder_pan']),
                  [0.0388353, 0.0, 0.0624])
    T_12 = make_T(Rx(90) @ Ry(pose['shoulder_lift']),
                  [0.0, -0.0452, 0.0])
    T_23 = make_T(Ry(pose['elbow_flex']),
                  [0.1013, 0.0, 0.0])
    T_34 = make_T(Rx(-90) @ Ry(pose['wrist_flex']),
                  [0.094, 0.0, 0.0])
    T_45 = make_T(Rx(90) @ Rz(pose['wrist_roll']),
                  [0.0, -0.0426, 0.0])
    T_5t = make_T(np.eye(3), [0.0, 0.0, 0.065])

    T = T_w1 @ T_12 @ T_23 @ T_34 @ T_45 @ T_5t
    return T[:3, 3]  # x, y, z position in meters

def detect_marker(frame, K, dist):
    """Detects the colored marker and returns its position (u,v) in pixels."""
    frame_ud = cv2.undistort(frame, K, dist)
    hsv = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, frame_ud
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 300:
        return None, frame_ud
    M = cv2.moments(c)
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    cv2.circle(frame_ud, (u,v), 8, (0,255,0), -1)
    cv2.putText(frame_ud, f"Marker ({u},{v})", (u+10, v-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return (u, v), frame_ud

# ── Load camera calibration ──
try:
    data = np.load(PARAMS_FILE)
    K    = data["K"]
    dist = data["dist"]
    print("Camera calibration loaded.")
except FileNotFoundError:
    print(f"'{PARAMS_FILE}' not found. Run 2_calibration.py first")
    exit()

fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]

# Robot connection
print(f"Connecting to the robot on {ROBOT_PORT}...")
config = SO101FollowerConfig(port=ROBOT_PORT)
robot  = SO101Follower(config)
robot.connect()
print("Robot connected.")

# Camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

robot_positions_m = []   # gripper positions in the robot frame (meters)
camera_positions_cm = [] # gripper positions in the camera frame (cm)

print(f"\nStarting hand-eye calibration: {len(CALIBRATION_POSES)} positions")
print("Press SPACE to validate each position, Q to cancel.\n")

for i, pose in enumerate(CALIBRATION_POSES):
    print(f"Position {i+1}/{len(CALIBRATION_POSES)} — moving the robot...")

    # Send the target pose to the robot
    action = {f"{k}.pos": v for k, v in pose.items()}
    robot.send_action(action)
    time.sleep(2.0)  # wait for the robot to settle

    # Gripper position via forward kinematics
    pos_robot = forward_kinematics(pose)  # in meters
    print(f"  Robot FK: x={pos_robot[0]*100:.1f}cm  y={pos_robot[1]*100:.1f}cm  z={pos_robot[2]*100:.1f}cm")

    # Capture and detect the marker
    detected = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        uv, display = detect_marker(frame, K, dist)

        status = f"Pos {i+1}/{len(CALIBRATION_POSES)} — "
        status += "MARKER DETECTED — SPACE to validate" if uv else "Marker not detected"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if uv else (0,0,255), 2)
        cv2.imshow("Hand-Eye Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and uv:
            # Convert pixels → cm in the camera frame
            u, v_px = uv
            Z = CAMERA_HEIGHT_CM
            X_cam = (u  - cx) * Z / fx
            Y_cam = (v_px - cy) * Z / fy
            camera_positions_cm.append([X_cam, Y_cam, Z])
            robot_positions_m.append(pos_robot)
            print(f"  Cam      : X={X_cam:.1f}cm  Y={Y_cam:.1f}cm  Z={Z:.1f}cm  ✓")
            detected = True
            break
        elif key == ord('q'):
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            robot.disconnect()
            exit()

cv2.destroyAllWindows()
cap.release()

# Compute the camera → robot transformation
print("\nComputing the hand-eye transformation...")

cam_pts  = np.array(camera_positions_cm)   # N x 3, in cm
rob_pts  = np.array(robot_positions_m) * 100  # N x 3, in cm

# Least-squares resolution: rob = R @ cam + t
# We use the SVD method to find R and t
cam_mean = cam_pts.mean(axis=0)
rob_mean = rob_pts.mean(axis=0)

cam_c = cam_pts - cam_mean
rob_c = rob_pts - rob_mean

H = cam_c.T @ rob_c
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T

# Correction if det(R) = -1 (reflection)
if np.linalg.det(R) < 0:
    Vt[-1,:] *= -1
    R = Vt.T @ U.T

t = rob_mean - R @ cam_mean

print(f"\nComputed transformation:")
print(f"  Rotation R:\n{R}")
print(f"  Translation t: {t} cm")

# Verification: reprojection error
errors = []
for i in range(len(cam_pts)):
    pred = R @ cam_pts[i] + t
    err  = np.linalg.norm(pred - rob_pts[i])
    errors.append(err)
    print(f"  Point {i+1}: error = {err:.2f} cm")

print(f"\nAverage error: {np.mean(errors):.2f} cm")

# Save
np.savez(OUTPUT_FILE, R=R, t=t)
print(f"\nHand-eye parameters saved to '{OUTPUT_FILE}'")

robot.disconnect()
