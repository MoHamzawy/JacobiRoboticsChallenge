import os
import copy
import yaml
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import cv2

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
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")

    cluster_colors = np.random.rand(max_label + 2, 3)
    cluster_colors[-1] = [0, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(cluster_colors[labels + 1])

    return pcd, labels


def fit_rectangle_on_plane(points):
    mean = points.mean(axis=0)
    centered = points - mean

    # PCA to define 2D projection frame
    pca = PCA(n_components=2)
    pca.fit(centered)
    plane_axes = pca.components_  # shape (2, 3)
    z_axis = np.cross(plane_axes[0], plane_axes[1])
    z_axis /= np.linalg.norm(z_axis)

    # Project points to 2D
    points_2d = centered @ plane_axes.T

    # Fit rotated rectangle
    rect = cv2.minAreaRect(points_2d.astype(np.float32))
    box_2d = cv2.boxPoints(rect)
    center_2d = np.mean(box_2d, axis=0)

    # Compute 2D X/Y directions
    x_dir_2d = box_2d[1] - box_2d[0]
    x_dir_2d /= np.linalg.norm(x_dir_2d)
    y_dir_2d = np.array([-x_dir_2d[1], x_dir_2d[0]])

    # Backproject X/Y to 3D
    x_axis = x_dir_2d[0] * plane_axes[0] + x_dir_2d[1] * plane_axes[1]
    y_axis = y_dir_2d[0] * plane_axes[0] + y_dir_2d[1] * plane_axes[1]
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Pose
    R = np.column_stack([x_axis, y_axis, z_axis])
    center_3d = center_2d[0] * plane_axes[0] + center_2d[1] * plane_axes[1] + mean

    # Size
    size = np.array(rect[1])
    return center_3d, R, size

def construct_pose_matrix(center, angle, z_height):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [center[0], center[1], z_height]
    return T

def create_oriented_box(pose, size=(0.3, 0.28, 0.2), color=(0, 1, 0)):
    box = o3d.geometry.TriangleMesh.create_box(*size)
    box.paint_uniform_color(color)
    box.translate(-np.array(size) / 2)
    box.transform(pose)
    return box


def filter_by_world_height(pcd, extrinsics, z_min=0.1):
    transformed = copy.deepcopy(pcd)
    transformed.transform(extrinsics)
    points = np.asarray(transformed.points)
    mask = points[:, 2] > z_min
    return transformed.select_by_index(np.where(mask)[0])


if __name__ == "__main__":
    cfg = load_config()
    depth, color, K, extr = load_data(cfg)
    extr[:3, 3] /= 1000.0

    point_cloud = create_point_cloud(depth, color, K)
    filtered_cloud = filter_by_world_height(point_cloud, extr, z_min=-0.85)
    downsampled = filtered_cloud.voxel_down_sample(voxel_size=0.005)

    clustered_cloud, labels = cluster_and_colorize(downsampled)

    geometries = [clustered_cloud]

    max_label = labels.max()
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        cluster = clustered_cloud.select_by_index(indices)
        points = np.asarray(cluster.points)
        if len(points) < 4:
            continue

        center, rotation, size = fit_rectangle_on_plane(points)
        z_height = points[:, 2].mean()

        # Build 4x4 pose matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = center

        box = create_oriented_box(T, size=np.append(size, 0.01))  # add thin height
        geometries.append(box)


    o3d.visualization.draw_geometries(geometries)
