import yaml
import os

def load_config(config_path="box_pose_estimation/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
data_dir = os.path.abspath(config["data_dir"])
print(f"Data directory: {data_dir}")
