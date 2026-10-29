"""
AI ORCHESTRATION - VISION -> DECISION -> PICK & PLACE

Ties the workstreams together:
  1. VISION    : HSV detection (same method as vision/3_detection.py) watches the table.
  2. DECISION  : if the object is recognized (stable detection), the YES branch of the
                 synoptic diagram lights up; otherwise the NO branch stays lit and the
                 arm does nothing.
  3. EXECUTION : on YES, run the pick & place with one of three executors:
                   --executor act   trained ACT policy on the real SO-101 (default)
                   --executor ik    geometric IK grasp sequence (from vision/5_pipeline.py)
                   --executor dry   no hardware, simulated pick (test the interface)

Interface: a single OpenCV window = live camera feed + synoptic diagram
(VISION -> RECOGNIZED? -> YES: PICK & PLACE -> BIN / NO: WAIT) with the active
branch highlighted.

Keys:
  SPACE  confirm the pick when the YES branch is lit
  A      toggle automatic mode (pick launches by itself after stable detection)
  ESC    abort a running episode
  Q      quit

PREREQUISITES:
  - vision/camera_params.npz  (script 2)            [recommended, warns if missing]
  - vision/handeye_params.npz (script 4)            [required for --executor ik]
  - a trained ACT checkpoint                        [required for --executor act]
  - robot connected on /dev/ttyACM0                 [act and ik executors]
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR   = Path(__file__).resolve().parent.parent
VISION_DIR = BASE_DIR / "vision"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

HOME = {'shoulder_pan': 0, 'shoulder_lift': -90, 'elbow_flex': 90,
        'wrist_flex': 0, 'wrist_roll': 0, 'gripper': 10}

# Same presets as vision/3_detection.py
HSV_PRESETS = {
    "red":   (np.array([0,   120, 70]),  np.array([10,  255, 255])),
    "blue":  (np.array([100, 120, 70]),  np.array([130, 255, 255])),
    "green": (np.array([40,  50,  50]),  np.array([80,  255, 255])),
}

# Colors (BGR)
C_BG       = (30, 30, 30)
C_INACTIVE = (90, 90, 90)
C_YES      = (80, 220, 80)
C_NO       = (60, 60, 230)
C_WAIT     = (60, 180, 255)
C_TEXT     = (230, 230, 230)


# ----------------------------------------------------------------------------
# Vision
# ----------------------------------------------------------------------------
def detect_object(frame, hsv_low, hsv_high, min_area, K=None, dist=None):
    """Returns ((u, v), area, annotated_frame) or (None, 0, annotated_frame)."""
    if K is not None:
        frame = cv2.undistort(frame, K, dist)

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0, frame

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, area, frame

    M = cv2.moments(c)
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    cv2.drawContours(frame, [c], -1, C_YES, 2)
    cv2.circle(frame, (u, v), 6, C_YES, -1)
    return (u, v), area, frame


def pixel_to_robot(u, v, K, camera_height_cm, R_he, t_he):
    """pixels -> camera frame (cm) -> robot frame (cm). Needs hand-eye calibration."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = camera_height_cm
    x_cam = (u - cx) * Z / fx
    y_cam = (v - cy) * Z / fy
    pos_robot = R_he @ np.array([x_cam, y_cam, Z]) + t_he
    return float(pos_robot[0]), float(pos_robot[1])


# ----------------------------------------------------------------------------
# Synoptic diagram
# ----------------------------------------------------------------------------
def _box(panel, x, y, w, h, label, color, filled=False, blink=False):
    if blink and int(time.time() * 2) % 2 == 0:
        color = C_INACTIVE
    if filled:
        cv2.rectangle(panel, (x, y), (x + w, y + h), tuple(int(c * 0.35) for c in color), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), color, 2)
    for i, line in enumerate(label.split("\n")):
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        tx = x + (w - size[0]) // 2
        ty = y + h // 2 + (i - (label.count("\n")) / 2) * 22 + 6
        cv2.putText(panel, line, (tx, int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _arrow(panel, p1, p2, color, label=None):
    cv2.arrowedLine(panel, p1, p2, color, 2, tipLength=0.06)
    if label:
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        cv2.putText(panel, label, (mx + 8, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_synoptic(state, auto_mode, status_text, executor_name):
    """
    state: SCAN | GO_WAIT | PICKING | DONE
    Layout (640x480):
        [ VISION ] -> [ RECOGNIZED ? ]
                        YES |     | NO
              [ PICK & PLACE ]  [ WAIT ]
                    |
                 [ BIN ]
    """
    panel = np.full((480, 640, 3), C_BG, dtype=np.uint8)
    cv2.putText(panel, "AI ORCHESTRATION - SO-101", (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 2)

    recognized = state in ("GO_WAIT", "PICKING", "DONE")

    c_vision = C_WAIT if state == "SCAN" else C_YES
    c_decide = C_YES if recognized else C_WAIT
    c_yes    = C_YES if recognized else C_INACTIVE
    c_no     = C_NO if state == "SCAN" else C_INACTIVE
    c_pick   = C_YES if state in ("PICKING", "DONE") else (C_YES if state == "GO_WAIT" else C_INACTIVE)
    c_bin    = C_YES if state == "DONE" else C_INACTIVE

    _box(panel, 30, 70, 130, 60, "VISION", c_vision, filled=True)
    _arrow(panel, (160, 100), (230, 100), c_vision)
    _box(panel, 230, 70, 190, 60, "OBJECT\nRECOGNIZED ?", c_decide, filled=True)

    # YES branch (down-left)
    _arrow(panel, (280, 130), (200, 210), c_yes, "YES")
    _box(panel, 90, 210, 220, 70, "PICK & PLACE\n(" + executor_name + ")", c_pick,
         filled=state in ("GO_WAIT", "PICKING"), blink=(state == "GO_WAIT"))
    _arrow(panel, (200, 280), (200, 340), C_YES if state == "DONE" else c_pick)
    _box(panel, 130, 340, 140, 60, "BIN", c_bin, filled=(state == "DONE"))

    # NO branch (down-right)
    _arrow(panel, (380, 130), (460, 210), c_no, "NO")
    _box(panel, 390, 210, 160, 70, "WAIT\n(no pick)", c_no, filled=(state == "SCAN"))

    # Status bar
    cv2.rectangle(panel, (0, 430), (640, 480), (50, 50, 50), -1)
    cv2.putText(panel, status_text, (10, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1)
    mode = "AUTO" if auto_mode else "MANUAL (SPACE = confirm)"
    cv2.putText(panel, f"Mode: {mode}   A=auto  Q=quit", (10, 472),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
    return panel


def show_ui(frame, panel, window="AI Orchestration SO-101"):
    cam = cv2.resize(frame, (640, 480))
    cv2.imshow(window, np.hstack([cam, panel]))


# ----------------------------------------------------------------------------
# Executors
# ----------------------------------------------------------------------------
def find_robot_port():
    """Auto-detect the SO-101 serial adapter.
    Excludes genuine-Arduino VIDs (the rl_grip piezo sensor)."""
    import serial.tools.list_ports
    ARDUINO_VIDS = {0x2341, 0x2A03}
    candidates = [p for p in serial.tools.list_ports.comports()
                  if ("ttyACM" in p.device or "ttyUSB" in p.device)
                  and p.vid not in ARDUINO_VIDS]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if "ttyACM" in p.device else 1, p.device))
    if len(candidates) > 1:
        print("Several serial ports found:")
        for c in candidates:
            print(f"  {c.device}  ({c.description})")
        print(f"Using {candidates[0].device} — if that is the leader arm or "
              f"another device, pass --robot-port explicitly.")
    return candidates[0].device


def connect_robot(port, robot_id):
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower
        from lerobot.robots.so_follower.config_so_follower import (
            SOFollowerRobotConfig as SO101FollowerConfig,
        )
    # The id selects the calibration file: it MUST be the one used during
    # lerobot-record, so observations/actions live in the same normalized
    # space (raw 0-4095 servo ticks -> calibrated units) as the training data.
    print(f"Connecting to the robot on {port} (id={robot_id})...")
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    robot.connect()
    print("Robot connected.")
    return robot


def send_pose(robot, pose, gripper_val=None):
    if gripper_val is not None:
        pose = dict(pose)
        pose['gripper'] = gripper_val
    robot.send_action({f"{k}.pos": v for k, v in pose.items()})


def load_act_policy(path, device):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    print(f"Loading ACT policy from {path}...")
    policy = ACTPolicy.from_pretrained(str(path))
    policy = policy.to(device)
    policy.eval()
    print("ACT policy loaded.")
    return policy


def run_act_episode(robot, policy, cap, K, dist, device, seconds, fps,
                    ui_state):
    """Run the ACT policy on the real robot, updating the UI at each step.
    Returns True if completed, False if aborted (ESC)."""
    import torch
    policy.reset()
    dt = 1.0 / fps
    t_end = time.time() + seconds
    step = 0

    while time.time() < t_end:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Camera error during episode."); return False
        # No undistortion here: lerobot-record fed RAW frames during training,
        # so inference must see raw frames too.

        obs_robot = robot.get_observation()
        state = torch.tensor(
            [[float(obs_robot[f"{n}.pos"]) for n in JOINT_NAMES]],
            dtype=torch.float32, device=device)
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = (torch.from_numpy(img).float().permute(2, 0, 1)
                 .unsqueeze(0).to(device) / 255.0)

        with torch.no_grad():
            action = policy.select_action({
                "observation.state": state,
                "observation.images.front": img_t,
            })
        action = action.squeeze(0).cpu().numpy()
        robot.send_action({f"{n}.pos": float(v)
                           for n, v in zip(JOINT_NAMES, action)})
        step += 1

        remaining = int(t_end - time.time())
        panel = draw_synoptic("PICKING", ui_state["auto"],
                              f"ACT running... step {step}, {remaining}s left (ESC = abort)",
                              ui_state["executor"])
        show_ui(frame, panel)
        if (cv2.waitKey(1) & 0xFF) == 27:
            print("Episode aborted by user.")
            return False

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)
    return True


# --- IK executor (adapted from vision/5_pipeline.py) -------------------------
L1, L2, L3 = 6.24, 10.13, 9.40   # SO-101 segment lengths (cm)


def inverse_kinematics(x_cm, y_cm, z_cm):
    shoulder_pan = np.degrees(np.arctan2(y_cm, x_cm))
    r = np.sqrt(x_cm**2 + y_cm**2)
    z_eff = z_cm - L1
    d = np.sqrt(r**2 + z_eff**2)
    if d > (L2 + L3) or d < abs(L2 - L3):
        print(f"  Out of reach: distance={d:.1f}cm, max={L2+L3:.1f}cm")
        return None
    cos_elbow = np.clip((L2**2 + L3**2 - d**2) / (2 * L2 * L3), -1, 1)
    elbow_flex = np.degrees(np.arccos(cos_elbow))
    alpha = np.degrees(np.arctan2(z_eff, r))
    cos_beta = np.clip((L2**2 + d**2 - L3**2) / (2 * L2 * d), -1, 1)
    beta = np.degrees(np.arccos(cos_beta))
    shoulder_lift = -(alpha + beta)
    wrist_flex = -(shoulder_lift + elbow_flex - 90)
    return {'shoulder_pan': shoulder_pan, 'shoulder_lift': shoulder_lift,
            'elbow_flex': elbow_flex, 'wrist_flex': wrist_flex,
            'wrist_roll': 0.0, 'gripper': 10}


def run_ik_pick(robot, x_r, y_r, z_pick, z_lift):
    """Scripted grasp sequence. Returns True if completed."""
    pose_pick = inverse_kinematics(x_r, y_r, z_pick)
    pose_lift = inverse_kinematics(x_r, y_r, z_lift)
    if pose_pick is None or pose_lift is None:
        print("Object out of the robot's reach!")
        return False

    steps = [
        ("Opening gripper",   HOME,      10, 1.0),
        ("Descending",        pose_lift, 10, 2.0),
        ("Reaching object",   pose_pick, 10, 2.0),
        ("Closing gripper",   pose_pick, 50, 1.5),
        ("Lifting",           pose_lift, 50, 2.0),
        ("Moving to bin",     HOME,      50, 2.0),
        ("Dropping in bin",   HOME,      10, 1.0),
    ]
    for name, pose, grip, wait in steps:
        print(f"  {name}...")
        send_pose(robot, pose, gripper_val=grip)
        time.sleep(wait)
    return True


def run_dry_pick(cap, K, dist, ui_state, seconds=5):
    """No hardware: animate the PICKING state for a few seconds."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        ret, frame = cap.read()
        if not ret:
            return False
        if K is not None:
            frame = cv2.undistort(frame, K, dist)
        panel = draw_synoptic("PICKING", ui_state["auto"],
                              f"DRY RUN... {int(t_end - time.time())}s left",
                              ui_state["executor"])
        show_ui(frame, panel)
        if (cv2.waitKey(30) & 0xFF) == 27:
            return False
    return True


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="AI orchestration: vision -> decision -> pick & place")
    p.add_argument("--executor", choices=["act", "ik", "dry"], default="act")
    p.add_argument("--policy-path", default="/home/leo28/models/act_so101",
                   help="ACT checkpoint (executor=act)")
    p.add_argument("--robot-port", default="auto",
                   help="serial port of the follower arm, or 'auto' to detect it")
    p.add_argument("--robot-id", default="my_follower_arm",
                   help="robot id used during lerobot-record (selects the calibration file)")
    p.add_argument("--camera-index", default="0",
                   help="camera index (0, 2, ...) or device path (/dev/video2). "
                        "For executor=act, use the SAME camera the policy was trained with.")
    p.add_argument("--color", choices=list(HSV_PRESETS), default="red",
                   help="HSV preset of the object to recognize")
    p.add_argument("--min-area", type=int, default=400,
                   help="minimum contour area (px) to consider the object recognized")
    p.add_argument("--stable-frames", type=int, default=15,
                   help="consecutive detection frames required before YES")
    p.add_argument("--episode-seconds", type=int, default=30,
                   help="duration of an ACT episode (executor=act)")
    p.add_argument("--fps", type=int, default=15, help="ACT control frequency")
    p.add_argument("--camera-height", type=float, default=40.0,
                   help="camera height above the table in cm (executor=ik)")
    p.add_argument("--z-pick", type=float, default=1.6)
    p.add_argument("--z-lift", type=float, default=15.0)
    p.add_argument("--no-undistort", action="store_true",
                   help="skip undistortion in the HSV detection (use when the camera "
                        "is not the one calibrated in camera_params.npz)")
    p.add_argument("--auto", action="store_true",
                   help="start in automatic mode (no confirmation)")
    args = p.parse_args()

    hsv_low, hsv_high = HSV_PRESETS[args.color]

    # --- calibration files ---
    K = dist = None
    try:
        cam_data = np.load(VISION_DIR / "camera_params.npz")
        K, dist = cam_data["K"], cam_data["dist"]
        print("Camera calibration loaded.")
    except FileNotFoundError:
        print("WARNING: vision/camera_params.npz not found - no distortion correction.")
    if args.no_undistort:
        K = dist = None
        print("Undistortion disabled (--no-undistort).")

    R_he = t_he = None
    if args.executor == "ik":
        try:
            he_data = np.load(VISION_DIR / "handeye_params.npz")
            R_he, t_he = he_data["R"], he_data["t"]
            print("Hand-eye calibration loaded.")
        except FileNotFoundError:
            print("ERROR: executor=ik requires vision/handeye_params.npz (run 4_handeye.py).")
            return
        if K is None:
            print("ERROR: executor=ik requires vision/camera_params.npz (run 2_calibration.py).")
            return

    # --- hardware / model (lazy where possible) ---
    device = "cpu"
    policy = None
    if args.executor == "act":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            print("ERROR: PyTorch is required for executor=act."); return
        policy = load_act_policy(args.policy_path, device)

    robot = None
    if args.executor in ("act", "ik"):
        port = args.robot_port
        if port == "auto":
            port = find_robot_port()
            if port is None:
                print("ERROR: no robot serial port detected. Plug the follower arm "
                      "(USB) or pass --robot-port /dev/ttyACMx.")
                return
            print(f"Auto-detected robot port: {port}")
        robot = connect_robot(port, args.robot_id)
        send_pose(robot, HOME)
        time.sleep(2.0)

    cam_src = int(args.camera_index) if args.camera_index.isdigit() else args.camera_index
    cap = cv2.VideoCapture(cam_src)
    # 640x480 = the resolution the ACT dataset was recorded at (lerobot-record).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {cam_src}."); return

    print("\nOrchestration running. SPACE=confirm  A=auto  Q=quit\n")

    state = "SCAN"
    stable_count = 0
    done_until = 0.0
    ui_state = {"auto": args.auto, "executor": args.executor.upper()}
    last_uv = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error."); break

            uv, area, frame = detect_object(frame, hsv_low, hsv_high,
                                            args.min_area, K, dist)

            # --- state machine ---
            if state == "SCAN":
                if uv:
                    stable_count += 1
                    last_uv = uv
                    if stable_count >= args.stable_frames:
                        state = "GO_WAIT"
                else:
                    stable_count = 0
                status = (f"Scanning for {args.color} object... "
                          f"({stable_count}/{args.stable_frames} stable frames)")

            elif state == "GO_WAIT":
                if not uv:
                    stable_count = 0
                    state = "SCAN"
                    status = "Object lost - back to scan."
                else:
                    last_uv = uv
                    status = ("Object RECOGNIZED -> "
                              + ("auto pick..." if ui_state["auto"]
                                 else "press SPACE to pick"))

            elif state == "DONE":
                status = "Pick & place finished. Cooling down..."
                if time.time() > done_until:
                    state = "SCAN"
                    stable_count = 0

            panel = draw_synoptic(state, ui_state["auto"], status,
                                  ui_state["executor"])
            show_ui(frame, panel)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('a'):
                ui_state["auto"] = not ui_state["auto"]

            # --- launch the pick ---
            launch = (state == "GO_WAIT"
                      and (ui_state["auto"] or key == ord(' ')))
            if launch:
                print("\n--- DECISION: YES -> PICK & PLACE ---")
                ok = False
                if args.executor == "act":
                    ok = run_act_episode(robot, policy, cap, K, dist, device,
                                         args.episode_seconds, args.fps, ui_state)
                    send_pose(robot, HOME)
                    time.sleep(1.5)
                elif args.executor == "ik":
                    u, v = last_uv
                    x_r, y_r = pixel_to_robot(u, v, K, args.camera_height,
                                              R_he, t_he)
                    print(f"Object in robot frame: X={x_r:.1f}cm Y={y_r:.1f}cm")
                    ok = run_ik_pick(robot, x_r, y_r, args.z_pick, args.z_lift)
                else:
                    ok = run_dry_pick(cap, K, dist, ui_state)

                print("Pick & place finished." if ok else "Pick aborted/failed.")
                state = "DONE"
                done_until = time.time() + 3.0
                stable_count = 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if robot is not None:
            send_pose(robot, HOME)
            time.sleep(1.0)
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()
