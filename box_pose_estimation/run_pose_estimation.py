import os
import copy
import yaml
import numpy as np
import open3d as o3d


def load_config(path="box_pose_estimation/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(cfg):
    base_dir = os.path.abspath(cfg["data_dir"])
    depth = np.load(os.path.join(base_dir, cfg["depth_file"]))
    color = np.load(os.path.join(base_dir, cfg["color_file"]))
    intrinsics = np.load(os.path.join(base_dir, cfg["intrinsics_file"]))
    extrinsics = np.load(os.path.join(base_dir, cfg["extrinsics_file"]))
    return depth, color, intrinsics, extrinsics


def create_point_cloud(depth, color, intrinsics):
    """Creates an Open3D point cloud from depth and color images."""
    if depth.dtype != np.uint16:
        depth = (depth * 1000).astype(np.uint16)

    if color.dtype != np.uint8:
        color = (color * 255).astype(np.uint8) if color.max() <= 1 else color.astype(np.uint8)

    if color.ndim == 2:
        color = np.stack([color] * 3, axis=-1)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        convert_rgb_to_intensity=False
    )

    h, w = depth.shape
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def cluster_and_colorize(pcd, eps=0.02, min_points=20):
    """Applies DBSCAN clustering and assigns random colors to each cluster."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")

    cluster_colors = np.random.rand(max_label + 2, 3)
    cluster_colors[-1] = [0, 0, 0]  # noise points (-1 label) in black
    pcd.colors = o3d.utility.Vector3dVector(cluster_colors[labels + 1])

    return pcd, labels


def create_clipping_plane(z_height, size=1.0, color=(0.8, 0.1, 0.1), alpha=0.3):
    """Creates a flat box to visualize a clipping plane at a certain height."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.001)
    mesh.paint_uniform_color(color)
    mesh.translate([-size / 2, -size / 2, z_height])
    mesh.compute_vertex_normals()
    return mesh


def filter_by_world_height(pcd, extrinsics, z_min=0.1):
    """Transforms point cloud to world coordinates and filters points above z_min."""
    transformed = copy.deepcopy(pcd)
    transformed.transform(extrinsics)
    points = np.asarray(transformed.points)
    mask = points[:, 2] > z_min
    return transformed.select_by_index(np.where(mask)[0])


if __name__ == "__main__":
    cfg = load_config()
    depth, color, K, extr = load_data(cfg)

    extr[:3, 3] /= 1000.0  # convert mm to meters
    point_cloud = create_point_cloud(depth, color, K)
    filtered_cloud = filter_by_world_height(point_cloud, extr, z_min=-0.85)
    downsampled = filtered_cloud.voxel_down_sample(voxel_size=0.005)

    clustered_cloud, labels = cluster_and_colorize(downsampled)

    o3d.visualization.draw_geometries([clustered_cloud])
