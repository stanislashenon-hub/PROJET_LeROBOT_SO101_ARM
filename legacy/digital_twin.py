import time
import rerun as rr
import numpy as np
import pinocchio as pin
from pathlib import Path
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

REPO_ROOT   = Path(__file__).resolve().parent.parent
URDF_PATH   = REPO_ROOT / "sim" / "so101_new_calib.urdf"
PACKAGE_DIR = REPO_ROOT / "sim"
PORT        = "/dev/ttyACM1"

# STS3215 encoder resolution: 4096 ticks = 360°
TICKS_PER_REV  = 4096
CENTER_TICK    = 2048
MAX_DEGREES    = 180.0   # max range allowed per joint (±180°)

MOTORS = {
    "shoulder_pan":  Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "shoulder_lift": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "elbow_flex":    Motor(id=3, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "wrist_flex":    Motor(id=4, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "wrist_roll":    Motor(id=5, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "gripper":       Motor(id=6, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
}

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Sign per joint: +1 = normal direction, -1 = flip if the visual moves the wrong way
# Adjust these values if a joint appears mirrored in rerun
JOINT_SIGNS = {
    "shoulder_pan":  1,
    "shoulder_lift": 1,
    "elbow_flex":    1,
    "wrist_flex":    1,
    "wrist_roll":    1,
    "gripper":       1,
}

# Offset in degrees per joint if the motor's "neutral" position != 2048
# Leave at 0 to start, adjust if a joint shows a visible bias
JOINT_OFFSETS_DEG = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 0.0,
    "elbow_flex":    0.0,
    "wrist_flex":    0.0,
    "wrist_roll":    0.0,
    "gripper":       0.0,
}

def ticks_to_rad(ticks, name):
    """Convert raw encoder ticks (0-4095) to radians,
    centered on 2048, with per-joint sign and offset."""
    degrees = (ticks - CENTER_TICK) * 360.0 / TICKS_PER_REV
    degrees += JOINT_OFFSETS_DEG[name]
    degrees *= JOINT_SIGNS[name]
    # Clamp to avoid outlier values
    degrees = np.clip(degrees, -MAX_DEGREES, MAX_DEGREES)
    return np.deg2rad(degrees)

def read_all_positions(bus):
    positions = {}
    for name in JOINT_NAMES:
        try:
            val = bus.read("Present_Position", name, normalize=False)
            # Safety: out-of-range value → keep the center
            if not (0 <= val <= 4095):
                print(f"[WARN] {name}: tick out of range ({val}), replaced with 2048")
                val = CENTER_TICK
            positions[name] = val
        except Exception as e:
            print(f"Error reading {name}: {e}")
            positions[name] = CENTER_TICK
    return positions

def main():
    print("Loading URDF model + meshes...")
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(URDF_PATH),
        package_dirs=[str(PACKAGE_DIR)]
    )
    data        = model.createData()
    visual_data = pin.GeometryData(visual_model)
    print(f"Model loaded: {model.nq} degrees of freedom")
    print(f"Visual parts: {len(visual_model.geometryObjects)}")

    # Neutral position (all joints at 0 rad) for initialization
    q_neutral = pin.neutral(model)
    print(f"URDF neutral position: {np.round(np.rad2deg(q_neutral), 1)} °")

    print(f"Connecting to bus on {PORT}...")
    bus = FeetechMotorsBus(port=PORT, motors=MOTORS)
    bus.connect()
    print("Bus connected!")

    # Initialize rerun
    rr.init("so101_digital_twin", spawn=True)
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Spatial3DView(origin="/", contents="/**"),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Load the STL meshes once (static=True → sent only once)
    print("Sending 3D meshes to rerun...")
    missing = 0
    for geom in visual_model.geometryObjects:
        mesh_path = PACKAGE_DIR / geom.meshPath
        if mesh_path.exists():
            rr.log(
                f"world/robot/{geom.name}",
                rr.Asset3D(path=str(mesh_path)),
                static=True,
            )
        else:
            print(f"[WARN] Mesh not found: {mesh_path}")
            missing += 1
    print(f"Meshes sent! ({missing} missing)")

    print("Digital twin started — move the arm!")
    print("Ctrl+C to stop\n")

    try:
        while True:
            positions = read_all_positions(bus)

            # Build the configuration vector q
            # Start from the URDF neutral position to respect the model's offsets
            q = q_neutral.copy()
            for i, name in enumerate(JOINT_NAMES):
                if i < model.nq:
                    q[i] = ticks_to_rad(positions[name], name)

            # Optional debug: print the angles every 2 seconds
            if int(time.time() * 10) % 20 == 0:
                angles_deg = [round(np.rad2deg(q[i]), 1) for i in range(model.nq)]
                print(f"q (°): {dict(zip(JOINT_NAMES, angles_deg))}")

            # Forward kinematics
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pin.updateGeometryPlacements(model, data, visual_model, visual_data)

            # Send each part's transform to rerun
            for geom_id, geom in enumerate(visual_model.geometryObjects):
                T = visual_data.oMg[geom_id]
                rr.log(
                    f"world/robot/{geom.name}",
                    rr.Transform3D(
                        translation=T.translation,
                        mat3x3=T.rotation,
                    )
                )

            time.sleep(0.05)  # 20 Hz

    except KeyboardInterrupt:
        print("\nStopping digital twin")
        bus.disconnect()
        print("Disconnected cleanly")

if __name__ == "__main__":
    main()
