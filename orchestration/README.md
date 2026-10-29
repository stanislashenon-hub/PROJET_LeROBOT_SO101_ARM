# AI Orchestration — vision → decision → pick & place

`orchestrator.py` ties the workstreams together: the external camera watches the
table (HSV detection, same method as `vision/3_detection.py`); when the target
object is **recognized**, the YES branch of a live synoptic diagram lights up and
the pick & place is launched — otherwise the NO branch stays lit and the arm does
nothing.

```
[ VISION ] ──► [ OBJECT RECOGNIZED ? ]
                    │YES         │NO
                    ▼            ▼
             [ PICK & PLACE ] [ WAIT ]
                    │
                    ▼
                 [ BIN ]
```

The interface is a single OpenCV window: live camera feed on the left, synoptic
diagram on the right, with the active branch highlighted (green = YES/pick,
red = NO/wait).

## Executors

| Flag | What runs on YES | Requirements |
|---|---|---|
| `--executor act` (default) | trained ACT policy on the real SO-101 | ACT checkpoint, robot on `/dev/ttyACM0`, PyTorch |
| `--executor ik` | geometric IK grasp sequence (from `vision/5_pipeline.py`) | `camera_params.npz` + `handeye_params.npz`, robot |
| `--executor dry` | simulated pick (nothing moves) | camera only — use this to test the interface |

## Usage

```bash
# Test the interface without any hardware
python orchestration/orchestrator.py --executor dry --color red

# Real run with the trained ACT policy (use the SAME camera as during training!)
python orchestration/orchestrator.py --executor act \
    --policy-path /home/leo28/models/act_so101 \
    --camera-index /dev/video2 --no-undistort --color red

# The checkpoint can also come straight from the Hugging Face hub:
#   --policy-path LEO2862/act_so101_pick_place

# Geometric IK fallback
python orchestration/orchestrator.py --executor ik --camera-height 40
```

Keys: `SPACE` confirm the pick when YES is lit · `A` toggle automatic mode ·
`ESC` abort a running episode · `Q` quit.

Useful options: `--color {red,blue,green}`, `--min-area` (px), `--stable-frames`
(consecutive detections required before YES), `--episode-seconds` (ACT episode
duration), `--auto` (start in automatic mode).

The follower arm's serial port is **auto-detected** by default (`--robot-port auto`).
If several serial devices are plugged (leader arm, Arduino piezo sensor), the script
lists them and picks the first `ttyACM` — pass `--robot-port /dev/ttyACMx` to override.
The calibration is selected with `--robot-id` (default `my_follower_arm`, the id used
during `lerobot-record`).
