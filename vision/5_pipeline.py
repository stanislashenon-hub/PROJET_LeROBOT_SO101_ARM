"""
SCRIPT 5 - FULL PICK & PLACE PIPELINE
Detects the object (white cylinder) with the iPhone, computes its position in the
robot frame, runs inverse kinematics, and commands the SO-101.

PREREQUISITES:
- camera_params.npz  (script 2)
- handeye_params.npz (script 4)
- Robot connected on /dev/ttyACM0
- iPhone active via Droidcam
"""

import cv2
import numpy as np
import time
from pathlib import Path
from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig as SO101FollowerConfig

# PARAMETERS
BASE_DIR          = Path(__file__).resolve().parent
CAMERA_INDEX      = 0
CAMERA_PARAMS     = BASE_DIR / "camera_params.npz"
HANDEYE_PARAMS    = BASE_DIR / "handeye_params.npz"
ROBOT_PORT        = "/dev/ttyACM0"
CAMERA_HEIGHT_CM  = 30.0    # iPhone height above the table

# Object color: white cylinder
# White in HSV: low saturation, high value
HSV_LOW  = np.array([150, 100, 150])
HSV_HIGH = np.array([180, 255, 255])

# Table height as seen by the robot (gripper Z when grasping)
# Adjust to your setup: gripper height above the table in cm
Z_PICK_CM = 1.6   # gripper 2cm above the table to grasp
Z_LIFT_CM = 15.0  # safety height for movements

# SO-101 segment lengths (from the official URDF)
L1 = 6.24   # base → shoulder height (cm)
L2 = 10.13  # upper arm (cm)
L3 = 9.40   # forearm (cm)
L4 = 6.50   # tool/gripper (cm)

# SO-101 inverse kinematics
def inverse_kinematics(x_cm, y_cm, z_cm):
    """
    Computes the servo angles to reach (x, y, z) in cm
    in the robot frame (base at the center).
    Returns a dict of angles in degrees, or None if out of reach.

    Robot frame: x = forward, y = left, z = up
    """
    # Base rotation angle (shoulder_pan)
    shoulder_pan = np.degrees(np.arctan2(y_cm, x_cm))

    # Horizontal distance from the base
    r = np.sqrt(x_cm**2 + y_cm**2)

    # Height to reach from the shoulder
    z_eff = z_cm - L1

    # Total distance shoulder → target
    d = np.sqrt(r**2 + z_eff**2)

    # Reach check
    if d > (L2 + L3) or d < abs(L2 - L3):
        print(f"  Out of reach: distance={d:.1f}cm, max={L2+L3:.1f}cm")
        return None

    # Law of cosines for elbow_flex
    cos_elbow = (L2**2 + L3**2 - d**2) / (2 * L2 * L3)
    cos_elbow = np.clip(cos_elbow, -1, 1)
    elbow_flex = np.degrees(np.arccos(cos_elbow))

    # Shoulder angle
    alpha = np.degrees(np.arctan2(z_eff, r))
    cos_beta = (L2**2 + d**2 - L3**2) / (2 * L2 * d)
    cos_beta = np.clip(cos_beta, -1, 1)
    beta = np.degrees(np.arccos(cos_beta))
    shoulder_lift = -(alpha + beta)

    # Wrist: keep the gripper pointing down
    wrist_flex = -(shoulder_lift + elbow_flex - 90)

    return {
        'shoulder_pan':  shoulder_pan,
        'shoulder_lift': shoulder_lift,
        'elbow_flex':    elbow_flex,
        'wrist_flex':    wrist_flex,
        'wrist_roll':    0.0,
        'gripper':       10,   # open
    }

def send_pose(robot, pose, gripper_val=None):
    """Sends a target position to the robot."""
    if gripper_val is not None:
        pose = dict(pose)
        pose['gripper'] = gripper_val
    action = {f"{k}.pos": v for k, v in pose.items()}
    robot.send_action(action)

def detect_object(frame, K, dist):
    """Detects the white cylinder and returns (u, v) in pixels."""
    frame_ud = cv2.undistort(frame, K, dist)
    hsv = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, frame_ud, mask

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 400:
        return None, frame_ud, mask

    M = cv2.moments(c)
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    cv2.drawContours(frame_ud, [c], -1, (0,255,0), 2)
    cv2.circle(frame_ud, (u,v), 6, (0,255,0), -1)
    cv2.putText(frame_ud, "Object detected", (u+10, v-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return (u, v), frame_ud, mask

# Load parameters
try:
    cam_data = np.load(CAMERA_PARAMS)
    K    = cam_data["K"]
    dist = cam_data["dist"]
    print("Camera calibration loaded.")
except FileNotFoundError:
    print(f"'{CAMERA_PARAMS}' not found. Run 2_calibration.py first"); exit()

try:
    he_data = np.load(HANDEYE_PARAMS)
    R_he = he_data["R"]
    t_he = he_data["t"]
    print("Hand-eye calibration loaded.")
except FileNotFoundError:
    print(f"'{HANDEYE_PARAMS}' not found. Run 4_handeye.py first"); exit()

fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]

# Robot connection
print(f"Connecting to the robot on {ROBOT_PORT}...")
config = SO101FollowerConfig(port=ROBOT_PORT)
robot  = SO101Follower(config)
robot.connect()
print("Robot connected.\n")

# Home position
HOME = {'shoulder_pan': 0, 'shoulder_lift': -90, 'elbow_flex': 90,
        'wrist_flex': 0, 'wrist_roll': 0, 'gripper': 10}
send_pose(robot, HOME)
time.sleep(2.0)

# Camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Detection running. Press SPACE to start the pick, Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    uv, display, mask = detect_object(frame, K, dist)

    if uv:
        u, v_px = uv

        # pixels → camera frame (cm)
        Z = CAMERA_HEIGHT_CM
        X_cam = (u    - cx) * Z / fx
        Y_cam = (v_px - cy) * Z / fy

        # camera frame → robot frame (cm)
        pos_cam   = np.array([X_cam, Y_cam, Z])
        pos_robot = R_he @ pos_cam + t_he

        x_r, y_r = pos_robot[0], pos_robot[1]

        pos_text = f"Robot: X={x_r:.1f}cm  Y={y_r:.1f}cm"
        cv2.putText(display, pos_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(display, "SPACE = pick & place", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

    cv2.putText(display, "Q = quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.imshow("Pick & Place SO-101", display)
    cv2.imshow("Object mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' ') and uv:
        print(f"\n--- PICK & PLACE ---")
        print(f"Object detected: X={x_r:.1f}cm  Y={y_r:.1f}cm")

        # 1. Inverse kinematics for the pick position
        pose_pick = inverse_kinematics(x_r, y_r, Z_PICK_CM)
        if pose_pick is None:
            print("Object out of the robot's reach!")
            continue

        # 2. Intermediate lift position (avoids scraping the table)
        pose_above = dict(pose_pick)
        pose_above_ik = inverse_kinematics(x_r, y_r, Z_LIFT_CM)
        if pose_above_ik is None:
            print("Lift position out of reach!")
            continue

        print("1. Opening gripper...")
        send_pose(robot, HOME, gripper_val=10)
        time.sleep(1.0)

        print("2. Descending toward the object...")
        send_pose(robot, pose_above_ik, gripper_val=10)
        time.sleep(2.0)
        send_pose(robot, pose_pick, gripper_val=10)
        time.sleep(2.0)

        print("3. Closing gripper (grasp)...")
        send_pose(robot, pose_pick, gripper_val=50)
        time.sleep(1.5)

        print("4. Lifting the object...")
        send_pose(robot, pose_above_ik, gripper_val=50)
        time.sleep(2.0)

        print("5. Returning home with the object...")
        send_pose(robot, HOME, gripper_val=50)
        time.sleep(2.0)

        print("6. Dropping the object...")
        send_pose(robot, HOME, gripper_val=10)
        time.sleep(1.0)

        print("Pick & place finished!\n")

cap.release()
cv2.destroyAllWindows()
send_pose(robot, HOME)
time.sleep(1.0)
robot.disconnect()
print("Robot disconnected.")
