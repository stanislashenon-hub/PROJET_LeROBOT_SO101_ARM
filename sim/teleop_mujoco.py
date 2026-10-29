"""
════════════════════════════════════════════════════════════
Real-time teleoperation: physical SO101 leader → MuJoCo

Optimizations:
  • sync_read      : reads the 6 servos in a single batch (~3-5x faster)
  • reader thread   : USB reads never block the viewer
  • light filtering : reduced ALPHA for less inertia

Usage:
  python teleop_mujoco.py                    # auto-detect
  python teleop_mujoco.py --port /dev/ttyACM1
  python teleop_mujoco.py --scene path.xml
  python teleop_mujoco.py --simulate         # without a leader
════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# Configs
BASE_DIR = Path(__file__).resolve().parent
SCENE_CANDIDATES = [
    BASE_DIR / "scene_pick_place.xml",
    Path.cwd() / "scene_pick_place.xml",
]
SCENE_FALLBACK = BASE_DIR / "scene.xml"

LEADER_PORT = "/dev/ttyACM2"
CALIB_PATH  = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_leader_arm.json"

FREQ_HZ      = 100        # more responsive viewer
DT           = 1.0 / FREQ_HZ
ALPHA_FILTER = 0.15       # less inertia

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper",
]


# Mujoco
def load_mujoco(scene_path):
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data  = mujoco.MjData(model)
    return model, data


def set_joint_positions(model, data, actions):
    actions = np.atleast_1d(np.array(actions, dtype=float)).flatten()
    for i, name in enumerate(JOINT_NAMES):
        if i >= len(actions):
            break
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            data.ctrl[aid] = actions[i]


# LeRobot
def _load_lerobot():
    try:
        import serial.tools.list_ports
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration

        motors = {name: Motor(i + 1, "sts3215", MotorNormMode.DEGREES)
                  for i, name in enumerate(JOINT_NAMES)}
        return serial, FeetechMotorsBus, motors, MotorCalibration, True
    except ImportError as e:
        print(f"[WARN] LeRobot not available: {e}")
        return None, None, None, None, False


def detect_leader_port(serial_module, preferred=LEADER_PORT):
    ports = list(serial_module.tools.list_ports.comports())
    devices = [p.device for p in ports]
    if preferred and preferred in devices:
        print(f"[Leader] Preferred port found: {preferred}")
        return preferred
    for p in ports:
        if "ttyACM" in p.device or "ttyUSB" in p.device:
            print(f"[Leader] Port detected: {p.device}")
            return p.device
    return None


# Leader arm with reader thread
class LeaderArm:
    def __init__(self, port, simulate=False):
        serial_mod, FeetechMotorsBus, motors, MotorCalibration, lerobot_ok = _load_lerobot()

        self.simulate = simulate or not lerobot_ok or port is None
        self._bus = None
        self._last = np.zeros(len(JOINT_NAMES))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._use_sync_read = False
        self._warned = False

        if self.simulate:
            print("[Leader] STATIC mode — no leader, arm at rest.")
            return

        try:
            self._bus = FeetechMotorsBus(port=port, motors=motors)
            self._bus.connect()

            if CALIB_PATH.exists():
                with open(CALIB_PATH) as f:
                    raw = json.load(f)
                calib = {
                    name: MotorCalibration(
                        id=int(p["id"]),
                        drive_mode=int(p["drive_mode"]),
                        homing_offset=int(p["homing_offset"]),
                        range_min=int(p["range_min"]),
                        range_max=int(p["range_max"]),
                    )
                    for name, p in raw.items()
                }
                self._bus.calibration = calib
                if hasattr(self._bus, "write_calibration"):
                    self._bus.write_calibration(calib)
                print(f"[Leader] Calibration loaded and written ({CALIB_PATH.name})")
            else:
                print(f"[WARN] Calibration not found: {CALIB_PATH}")

            try:
                self._bus.disable_torque()
                print("[Leader] Torque disabled.")
            except Exception as e:
                print(f"[Leader] (info) disable_torque unavailable: {e}")

            self._use_sync_read = hasattr(self._bus, "sync_read")
            mode = "sync_read (batched)" if self._use_sync_read else "sequential read"
            print(f"[Leader] Read mode: {mode}")

            print(f"[Leader] Connected on {port}")
            print("[Leader] Move the arm — the MuJoCo twin will follow!\n")

            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

        except Exception as e:
            print(f"[Leader] Connection error: {e}")
            self.simulate = True
            self._bus = None

    def _read_once(self):
        try:
            if self._use_sync_read:
                vals = self._bus.sync_read("Present_Position")
                angles_deg = np.array([float(vals[n]) for n in JOINT_NAMES])
            else:
                angles_deg = np.array([
                    float(self._bus.read("Present_Position", n)) for n in JOINT_NAMES
                ])
            return np.deg2rad(angles_deg)
        except Exception as e:
            if not self._warned:
                print(f"[Leader] Read error: {e}")
                self._warned = True
            return None

    def _reader_loop(self):
        filt = None
        while not self._stop_event.is_set():
            angles = self._read_once()
            if angles is not None:
                if filt is None:
                    filt = angles
                else:
                    filt = ALPHA_FILTER * filt + (1 - ALPHA_FILTER) * angles
                with self._lock:
                    self._last = filt

    def read_positions_rad(self):
        with self._lock:
            return self._last.copy()

    def disconnect(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._bus is not None:
            try:
                self._bus.disconnect()
                print("[Leader] Disconnected.")
            except Exception:
                pass


# Main loop
def run(scene_path, leader):
    print(f"[MuJoCo] Loading scene: {scene_path}")
    model, data = load_mujoco(scene_path)
    print(f"[MuJoCo] Model: {model.nq} DOF, {model.nu} actuators\n")

    for name in JOINT_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        status = "✓" if aid >= 0 else "✗ NOT FOUND"
        print(f"  Actuator '{name}' → id={aid}  {status}")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.monotonic()
            set_joint_positions(model, data, leader.read_positions_rad())
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, DT - (time.monotonic() - t0)))


# Entry point
def find_scene(scene_arg):
    if scene_arg:
        p = Path(scene_arg).expanduser()
        return p if p.exists() else None
    for p in SCENE_CANDIDATES:
        if p.exists():
            print(f"[Scene] Found: {p}")
            return p
    if SCENE_FALLBACK.exists():
        print(f"[WARN] scene_pick_place.xml not found → falling back to scene.xml")
        return SCENE_FALLBACK
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--scene", default=None, help="Path to a MuJoCo .xml file")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    scene_path = find_scene(args.scene)
    if scene_path is None:
        print("[ERROR] No MuJoCo scene found.")
        return

    port = None
    if not args.simulate:
        serial_mod, _, _, _, lerobot_ok = _load_lerobot()
        if lerobot_ok:
            port = args.port or detect_leader_port(serial_mod)
        if port is None:
            print("[WARN] Leader not found.")
            rep = input("  Continue in static mode? [y/n]: ").strip().lower()
            if rep != "y":
                return
            args.simulate = True

    leader = LeaderArm(port=port, simulate=args.simulate)
    try:
        run(scene_path, leader)
    except KeyboardInterrupt:
        print("\n[Info] Stop requested.")
    finally:
        leader.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
