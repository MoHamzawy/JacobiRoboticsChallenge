import yaml
import os
import numpy as np
import open3d as o3d

def load_config(config_path="box_pose_estimation/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    data_dir = os.path.abspath(config["data_dir"])
    depth = np.load(os.path.join(data_dir, config["depth_file"]))
    color = np.load(os.path.join(data_dir, config["color_file"]))
    intrinsics = np.load(os.path.join(data_dir, config["intrinsics_file"]))
    return depth, color, intrinsics

def create_rgbd_image(depth, color):
    if depth.dtype != np.uint16:
        depth_mm = (depth * 1000).astype(np.uint16)
    else:
        depth_mm = depth

    if color.dtype != np.uint8:
        if color.max() <= 1.0:
            color_uint8 = (color * 255).astype(np.uint8)
        else:
            color_uint8 = color.astype(np.uint8)
    else:
        color_uint8 = color

    if color_uint8.ndim == 2:
        color_uint8 = np.stack([color_uint8] * 3, axis=-1)

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_uint8),
        o3d.geometry.Image(depth_mm),
        convert_rgb_to_intensity=False
    )


def get_intrinsics_o3d(intrinsics, width, height):
    return o3d.camera.PinholeCameraIntrinsic(width, height, 
                                             fx=intrinsics[0, 0], fy=intrinsics[1, 1], 
                                             cx=intrinsics[0, 2], cy=intrinsics[1, 2])

def visualize_point_cloud(depth, color, intrinsics):
    rgbd = create_rgbd_image(depth, color)
    height, width = depth.shape
    intrinsic = get_intrinsics_o3d(intrinsics, width, height)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    config = load_config()
    depth, color, intrinsics = load_data(config)
    visualize_point_cloud(depth, color, intrinsics)
