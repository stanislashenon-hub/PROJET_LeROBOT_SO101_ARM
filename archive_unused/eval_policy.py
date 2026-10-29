from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.robots.factory import make_robot
import torch

# Load the policy
print("Loading the policy...")
policy = ACTPolicy.from_pretrained("LEO2862/act_so101_pick_place")
policy.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.to(device)

# Connect the robot
print("Connecting to the robot...")
robot = make_robot(
    robot_type="so101_follower",
    robot_kwargs={
        "port": "/dev/ttyACM1",
        "cameras": {
            "front": {
                "type": "opencv",
                "index_or_path": "/dev/video2",
                "width": 640,
                "height": 480,
                "fps": 30
            }
        }
    }
)

robot.connect()

# Evaluation loop
num_episodes = 3
for episode in range(num_episodes):
    print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

    # Reset
    robot.teleop_to_starting_configuration()
    obs = robot.capture_observation()

    # Rollout
    for step in range(200):  # Max 200 steps per episode
        # Prepare the observation for the policy
        obs_dict = {
            "observation.images.front": obs["observation.images.front"],
            "observation.state": obs["observation.state"]
        }

        # Inference
        with torch.no_grad():
            action = policy.select_action(obs_dict)

        # Execute the action
        obs, reward, done, info = robot.send_action(action)

        if done:
            print(f"Episode finished at step {step}")
            break

    print(f"Success: {info.get('success', False)}")

robot.disconnect()
print("\nEvaluation finished!")
