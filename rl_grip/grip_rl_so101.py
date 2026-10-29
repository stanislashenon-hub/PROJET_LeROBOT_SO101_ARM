from __future__ import annotations
import argparse, json, os, sys, time, threading, datetime
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError :
    print("[WARN] pyserial missing — pip install pyserial")

try:
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    LEROBOT_OK = True
except ImportError as _lerobot_err:
    LEROBOT_OK = False
    print(f"[WARN] LeRobot import failed : {_lerobot_err}")
    print("       Rerun : conda activate lerobot && cd ~/lerobot && pip install -e .")

class CFG:
    ROBOT_PORT  = "/dev/ttyACM4"
    ROBOT_ID    = "my_follower_arm"
    ROBOT_TYPE  = "so101_follower"
    PORTS_ROBOT = {"/dev/ttyACM4"}
    ARDUINO_BAUD = 9600

    GRIPPER_IDX        = 5
    GRIPPER_OPEN_DEG   = 40.0
    GRIPPER_PHASE1_DEG = 15.0
    GRIPPER_LEVELS_DEG = [15.0, 12.5, 10.0, 7.5, 5.0]

    HOME_DEG = [0.0, -45.0, 90.0, 0.0, 0.0, GRIPPER_OPEN_DEG]

    HOLD_TIME_S        = 2.0
    TIMEOUT_CONTACT_S = 4.0
    RETURN_HOME_S     = 3.0
    RESET_TIME_S       = 8.0

    EMERGENCY_PEAK_MV   = 4800.0
    CONTACT_THRESHOLD_MV = 12.0

    N_ACTIONS    = 5
    N_BINS_PEAK   = 6
    N_BINS_GRAD  = 6
    ALPHA        = 0.3
    GAMMA        = 0.0
    EPSILON_INIT = 1.0
    EPSILON_MIN  = 0.15
    EPSILON_DECAY= 0.96
    PEAK_MAX_MV   = 400.0
    GRAD_MAX     = 100.0

    OBJECTS = {0: "plastic_cup", 1: "cardboard_box",
               2: "wood_block",  3: "metal_block"}
    N_TRIALS_PER_OBJECT = 50
    #REWARD APPLIED
    R_SUCCESS    =  1.0
    R_TOO_STRONG = -0.5
    R_FAILED     = -1.0

STIFFNESSES = {0: 0.15, 1: 0.35, 2: 0.65, 3: 0.92}

def find_arduino_port() -> str | None:
    if not SERIAL_OK:
        return None

    ARDUINO_VIDS = {0x2341, 0x2a03, 0x1A86, 0x0403, 0x10C4}
    candidates = []

    for port in serial.tools.list_ports.comports():
        if port.device in CFG.PORTS_ROBOT:
            continue
        if port.vid in ARDUINO_VIDS:
            candidates.insert(0, port.device)
        elif "ttyACM" in port.device or "ttyUSB" in port.device:
            candidates.append(port.device)

    if candidates:
        print(f"[Arduino] Detected : {candidates[0]}")
        return candidates[0]

    print("[Arduino] No port found")
    return None

class GripSensor:

    def __init__(self, port: str | None, simulate: bool = False):
        self.simulate    = simulate or not SERIAL_OK or port is None
        self._lock       = threading.Lock()
        self._peaks: list[float] = []
        self._running    = True
        self._emergency_cb = None

        if not self.simulate:
            try:
                self._ser = serial.Serial(port, CFG.ARDUINO_BAUD, timeout=2)
                self._thread = threading.Thread(target=self._read_loop, daemon=True)
                self._thread.start()
                time.sleep(1.5)
                print(f"[Sensor] Connected on {port} @ {CFG.ARDUINO_BAUD} baud")
            except serial.SerialException as e:
                print(f"[Sensor] Port error : {e} → simulation")
                self.simulate = True
        else:
            print("[Sensor] Simulation mode")

    def set_emergency_callback(self, cb): self._emergency_cb = cb

    def _read_loop(self):
        while self._running:
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("IMPACT_MV:"):
                    mv = float(line.split(":")[1])
                    with self._lock:
                        self._peaks.append(mv)
                    if mv >= CFG.EMERGENCY_PEAK_MV and self._emergency_cb:
                        self._emergency_cb(mv)
            except Exception:
                pass

    def reset(self):
        with self._lock:
            self._peaks.clear()

    def capture(self, timeout_s: float, object_stiffness: float = 0.5) -> dict:
        if self.simulate:
            time.sleep(0.3)
            return self._simulate(object_stiffness)
        self.reset()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            time.sleep(0.02)
            with self._lock:
                peaks = list(self._peaks)
            if len(peaks) >= 3 and max(peaks) > CFG.CONTACT_THRESHOLD_MV:
                return self._extract(peaks)
        return {"peak_mv": 0.0, "gradient_norm": 0.0, "duration_ms": 0.0}

    def _extract(self, peaks: list[float]) -> dict:
        peak_mv   = max(peaks)
        diffs    = [abs(peaks[i+1]-peaks[i]) for i in range(len(peaks)-1)]
        grad     = max(diffs) if diffs else 0.0
        grad_n   = round(min(grad / 0.8, CFG.GRAD_MAX), 2)
        thresh   = peak_mv * 0.1
        duration_ms = sum(1 for v in peaks if v > thresh) * 5.0
        return {"peak_mv": round(peak_mv,2), "gradient_norm": grad_n,
                "duration_ms": round(duration_ms,1)}

    def _simulate(self, stiffness: float) -> dict:
        n = lambda s: np.random.normal(0, s)
        return {"peak_mv":       round(float(np.clip(stiffness*340+n(35), 10, 400)), 2),
                "gradient_norm":round(float(np.clip(stiffness*78 +n(12),  1, 100)), 2),
                "duration_ms":  round(float(np.clip((1-stiffness)*110+n(18), 5, 200)), 1)}

    def close(self):
        self._running = False
        if not self.simulate:
            try: self._ser.close()
            except Exception: pass

class RobotInterface:

    def __init__(self, simulate: bool = False):
        self.simulate = simulate or not LEROBOT_OK
        self._robot   = None
        self._state   = list(CFG.HOME_DEG)

        if not self.simulate:
            try:
                from lerobot.robots.so_follower.so_follower import SOFollower
                from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
                config = SOFollowerRobotConfig(port=CFG.ROBOT_PORT, id=CFG.ROBOT_ID)
                self._robot = SOFollower(config)
                self._robot.connect()
                print(f"[Robot] Connected — {CFG.ROBOT_PORT} / {CFG.ROBOT_ID}")
            except Exception as e:
                print(f"[Robot] Connection error : {e}")
                print("[Robot] → simulation mode")
                self.simulate = True
        else:
            print("[Robot] Simulation mode — commands displayed without execution")

    def home(self):
        self._send_joints(CFG.HOME_DEG, label="HOME")
        time.sleep(CFG.RETURN_HOME_S)

    def open_gripper(self):
        self._set_gripper(CFG.GRIPPER_OPEN_DEG, label="Open (100°)")

    def close_phase1(self):
        self._set_gripper(CFG.GRIPPER_PHASE1_DEG, label="Phase 1 (85°)")

    def close_phase2(self, action: int):
        deg = CFG.GRIPPER_LEVELS_DEG[action]
        self._set_gripper(deg, label=f"Phase 2 (A{action}={deg:.0f}°)")

    JOINTS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
              'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']

    def _set_gripper(self, deg: float, label: str = ""):
        if self.simulate:
            print(f"  [sim] {label}")
            self._state[CFG.GRIPPER_IDX] = deg
            time.sleep(0.3)
            return
        try:
            obs    = self._robot.get_observation()
            action = dict(obs)
            action['gripper.pos'] = deg
            self._robot.send_action(action)
            print(f"  [Robot] {label}")
        except Exception as e:
            print(f"  [Robot] Error _set_gripper : {e}")

    def _send_joints(self, angles: list[float], label: str = ""):
        if self.simulate:
            print(f"  [sim] Joints → {[f'{a:.0f}°' for a in angles]}  ({label})")
            self._state = list(angles)
            return
        try:
            action = dict(zip(self.JOINTS, angles))
            self._robot.send_action(action)
            print(f"  [Robot] {label}")
        except Exception as e:
            print(f"  [Robot] Error _send_joints : {e}")

    def disconnect(self):
        if self._robot is not None:
            try: self._robot.disconnect()
            except Exception: pass

class QTable:

    def __init__(self):
        self.q       = np.full((CFG.N_BINS_PEAK, CFG.N_BINS_GRAD, CFG.N_ACTIONS), 0.1)
        self.epsilon = CFG.EPSILON_INIT
        self.updates = 0

    def _bin(self, f: dict) -> tuple[int, int]:
        bp = min(int(f["peak_mv"]/CFG.PEAK_MAX_MV*CFG.N_BINS_PEAK),  CFG.N_BINS_PEAK-1)
        bg = min(int(f["gradient_norm"]/CFG.GRAD_MAX*CFG.N_BINS_GRAD), CFG.N_BINS_GRAD-1)
        return bp, bg

    def choose(self, f: dict) -> tuple[int, str]:
        if np.random.random() < self.epsilon:
            return np.random.randint(CFG.N_ACTIONS), "exploration"
        bp, bg = self._bin(f)
        return int(np.argmax(self.q[bp, bg])), "exploitation"

    def update(self, f: dict, action: int, reward: float):
        bp, bg = self._bin(f)
        self.q[bp,bg,action] += CFG.ALPHA*(reward + CFG.GAMMA*np.max(self.q[bp,bg]) - self.q[bp,bg,action])
        self.updates += 1
        if reward > 0:
            self.epsilon = max(CFG.EPSILON_MIN, self.epsilon*CFG.EPSILON_DECAY)

    def save(self, path: str):
        np.save(path, self.q)
        with open(path.replace(".npy","_meta.json"),"w") as f:
            json.dump({"epsilon":self.epsilon,"updates":self.updates}, f)

    def load(self, path: str):
        self.q = np.load(path)
        mp = path.replace(".npy","_meta.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            self.epsilon = m.get("epsilon", CFG.EPSILON_INIT)
            self.updates = m.get("updates", 0)
        print(f"[QTable] Loaded — ε={self.epsilon:.3f}  updates={self.updates}")

    def display(self) -> str:
        best  = np.argmax(self.q, axis=2)
        lines = [f"  ε={self.epsilon:.3f}  updates={self.updates}",
                 "  " + "  ".join(f"G{j}" for j in range(CFG.N_BINS_GRAD))]
        for i in range(CFG.N_BINS_PEAK):
            row = "  ".join(f"A{best[i,j]}({CFG.GRIPPER_LEVELS_DEG[best[i,j]]:.0f}°)"
                            for j in range(CFG.N_BINS_GRAD))
            lines.append(f"  P{i}: {row}")
        return "\n".join(lines)

@dataclass
class Transition:
    trial_id:  int;  object_id: int;  object_name: str
    features:  dict; action:   int;  degrees:    float
    mode:      str;  reward:   float; label_src: str
    epsilon:   float; timestamp: str
    def to_json(self): return json.dumps(asdict(self), ensure_ascii=False)

def ask_label() -> tuple[float, str]:
    KEYS = {"o":(CFG.R_SUCCESS,"success (+1.0)"),
            "f":(CFG.R_TOO_STRONG,"too strong (-0.5)"),
            "r":(CFG.R_FAILED,"failed (-1.0)")}
    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │  Result ?                                │")
    print("  │  [o] success    [f] too strong   [r] failed │")
    print("  │  [q] stop cleanly                        │")
    print("  └─────────────────────────────────────────┘")
    print("  > ", end="", flush=True)
    while True:
        try:
            c = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt
        if c == "q": raise KeyboardInterrupt
        if c in KEYS:
            r, desc = KEYS[c]
            print(f"  → {desc}")
            return r, "human"
        print("  [o/f/r/q] : ", end="", flush=True)

class Session:
    SEP = "═"*58

    def __init__(self, robot, sensor, model, out_dir):
        self.robot    = robot
        self.sensor   = sensor
        self.model    = model
        self.out_dir  = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.transitions: list[Transition] = []
        self.global_trial = 0
        self.counters = {i:0 for i in CFG.OBJECTS}
        self.successes = {i:0 for i in CFG.OBJECTS}
        sensor.set_emergency_callback(self._emergency)

    def _emergency(self, mv):
        print(f"\n  !! EMERGENCY peak={mv:.0f}mV — releasing !!")
        self.robot.open_gripper()

    def run_trial(self, object_id: int) -> bool:
        name      = CFG.OBJECTS[object_id]
        stiffness = STIFFNESSES[object_id]
        self.global_trial += 1
        n_obj    = self.counters[object_id] + 1

        print(f"\n{self.SEP}")
        print(f"  Trial {self.global_trial:3d}/200  │  {name}  ({n_obj}/50)  │  ε={self.model.epsilon:.3f}")
        print(self.SEP)

        print("  [0] Return HOME...")
        self.robot.home()
        self.robot.open_gripper()

        print(f"\n  Place object : {name.upper()}")
        print("  Press [Enter] when object is in position... ", end="", flush=True)
        try:
            input()
        except KeyboardInterrupt:
            return False

        print("  [1] Phase 1 — exploratory close (15°)...")
        self.sensor.reset()
        self.robot.close_phase1()
        features = self.sensor.capture(CFG.TIMEOUT_CONTACT_S, stiffness)
        print(f"      peak={features['peak_mv']:.1f}mV  grad={features['gradient_norm']:.2f}  dur={features['duration_ms']:.0f}ms")

        if features["peak_mv"] < CFG.CONTACT_THRESHOLD_MV:
            print("  [!] No contact — trial ignored")
            self.robot.open_gripper()
            return True

        action, mode = self.model.choose(features)
        degrees = CFG.GRIPPER_LEVELS_DEG[action]
        print(f"  [2] Decision : A{action} = {degrees:.0f}°  [{mode}]")

        print(f"  [3] Phase 2 — tighten to {degrees:.0f}°...")
        self.robot.close_phase2(action)
        print(f"      Hold {CFG.HOLD_TIME_S:.0f}s...")
        time.sleep(CFG.HOLD_TIME_S)

        try:
            reward, label_src = ask_label()
        except KeyboardInterrupt:
            self.robot.open_gripper()
            self.robot.home()
            return False

        self.robot.open_gripper()

        self.model.update(features, action, reward)

        t = Transition(
            trial_id=self.global_trial, object_id=object_id, object_name=name,
            features=features, action=action, degrees=degrees, mode=mode,
            reward=reward, label_src=label_src, epsilon=self.model.epsilon,
            timestamp=datetime.datetime.now().isoformat(),
        )
        self.transitions.append(t)
        with open(self.out_dir/"transitions.jsonl","a") as f:
            f.write(t.to_json()+"\n")
        self.model.save(str(self.out_dir/"qtable.npy"))

        self.counters[object_id] += 1
        if reward > 0: self.successes[object_id] += 1

        col = "OK" if reward>0 else ("STRONG" if reward>-1 else "FAILED")
        c, r = self.counters[object_id], self.successes[object_id]
        print(f"  [{col}]  r={reward:+.1f}  {name}: {r}/{c} ({r/c*100:.0f}%)")

        if self.global_trial < 200:
            print(f"\n  Replace object — {CFG.RESET_TIME_S:.0f}s...", end="", flush=True)
            time.sleep(CFG.RESET_TIME_S)
            print(" ready.")
        return True

    def run(self):
        self._save_config()
        print(f"\n{self.SEP}")
        print("  RL SESSION — SO101  |  4 objects × 50 = 200 trials")
        print(f"  Output : {self.out_dir}")
        print(self.SEP)

        agenda, per = [], {i:0 for i in CFG.OBJECTS}
        while any(per[i] < CFG.N_TRIALS_PER_OBJECT for i in CFG.OBJECTS):
            candidates = [i for i in CFG.OBJECTS if per[i] < CFG.N_TRIALS_PER_OBJECT]
            np.random.shuffle(candidates)
            for oid in candidates:
                agenda.append(oid); per[oid] += 1

        try:
            for oid in agenda:
                if not self.run_trial(oid):
                    print("\n  Session stopped cleanly.")
                    break
        except KeyboardInterrupt:
            print("\n  Ctrl+C — saving...")
        finally:
            self._finalize()

    def _save_config(self):
        cfg = {k:v for k,v in CFG.__dict__.items() if not k.startswith("_")}
        cfg["timestamp"] = datetime.datetime.now().isoformat()
        with open(self.out_dir/"config.json","w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)

    def _finalize(self):
        self.model.save(str(self.out_dir/"qtable.npy"))
        if not self.transitions: return
        rewards = [t.reward for t in self.transitions]
        lines = [self.SEP, "SESSION REPORT",
                 f"Trials       : {len(self.transitions)}",
                 f"Success rate : {sum(1 for r in rewards if r>0)/len(rewards)*100:.1f}%",
                 f"Avg reward   : {np.mean(rewards):.3f}", "", "By object :"]
        for i,name in CFG.OBJECTS.items():
            c,r = self.counters[i], self.successes[i]
            lines.append(f"  {name:<22} {r:2d}/{c:2d}  ({r/c*100:.0f}%)" if c else f"  {name:<22}  0 trial")
        lines += ["", "Final Q-table :", self.model.display(), self.SEP]
        report = "\n".join(lines)
        with open(self.out_dir/"report.txt","w") as f: f.write(report)
        print("\n"+report)
        print(f"\n  Data → {self.out_dir}")

def main():
    parser = argparse.ArgumentParser(description="RL two-phase gripper learning — SO101")
    parser.add_argument("--simulate",  action="store_true", help="Full simulation")
    parser.add_argument("--resume", default=None, metavar="DIR")
    parser.add_argument("--output",    default="experiments")
    args = parser.parse_args()

    out_dir = (Path(args.resume) if args.resume
               else Path(args.output) / datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S"))

    arduino_port = None
    if not args.simulate:
        arduino_port = find_arduino_port()
        if arduino_port is None:
            rep = input("  Arduino not found. Continue in simulation ? [y/n] : ").strip().lower()
            if rep == "y": args.simulate = True
            else: sys.exit(0)

    sensor = GripSensor(port=arduino_port, simulate=args.simulate)
    robot  = RobotInterface(simulate=args.simulate)
    model = QTable()

    if args.resume and (out_dir/"qtable.npy").exists():
        model.load(str(out_dir/"qtable.npy"))

    session = Session(robot, sensor, model, out_dir)
    try:
        session.run()
    finally:
        robot.disconnect()
        sensor.close()
        print("Done.")

if __name__ == "__main__":
    main()