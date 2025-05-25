import yaml
import os
import numpy as np
import open3d as o3d
import copy
import cv2
from sklearn.decomposition import PCA

def load_config_file(path="box_pose_estimation/config.yaml"):
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

def transform_point_cloud(pcd, extrinsics):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(extrinsics)
    return pcd_copy


def crop_point_cloud_by_roi(pcd, min_bound, max_bound):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return pcd.crop(bbox)

def filter_point_cloud_full(pcd, cfg):
    # Step 1: Apply min Z filter
    z_min = cfg.get("z_min_filter", -0.75)
    points = np.asarray(pcd.points)
    mask = points[:, 2] > z_min
    pcd = pcd.select_by_index(np.where(mask)[0])

    # Step 2: Estimate normals and keep only horizontal surfaces
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    normals = np.asarray(pcd.normals)
    horizontal_mask = np.abs(normals[:, 2]) > cfg.get("normal_z_threshold", 0.5)
    pcd = pcd.select_by_index(np.where(horizontal_mask)[0])

    # Step 3: Denoise
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    # Step 4: Cluster and filter small clusters
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
    max_label = labels.max()
    clusters = []

    min_cluster_size = cfg.get("min_cluster_size", 100)

    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        if len(indices) >= min_cluster_size:
            clusters.append(pcd.select_by_index(indices))

    if not clusters:
        print("No valid clusters found after filtering.")
        return o3d.geometry.PointCloud()

    merged = clusters[0]
    for cluster in clusters[1:]:
        merged += cluster

    return merged

def estimate_box_pose_from_cluster(pcd_cluster):
    # Step 1: Fit plane using RANSAC
    plane_model, inliers = pcd_cluster.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=1000)
    pcd_plane = pcd_cluster.select_by_index(inliers)

    # Step 2: Center and apply PCA for 2D projection
    points = np.asarray(pcd_plane.points)
    mean = points.mean(axis=0)
    centered = points - mean

    pca = PCA(n_components=2)
    pca.fit(centered)
    axes_2d = pca.components_
    z_axis = np.cross(axes_2d[0], axes_2d[1])
    z_axis /= np.linalg.norm(z_axis)

    points_2d = centered @ axes_2d.T

    # Step 3: Fit 2D bounding box
    rect = cv2.minAreaRect(points_2d.astype(np.float32))
    box_2d = cv2.boxPoints(rect)
    center_2d = np.mean(box_2d, axis=0)

    # Step 4: Compute 2D orientation
    x_dir_2d = box_2d[1] - box_2d[0]
    x_dir_2d /= np.linalg.norm(x_dir_2d)
    y_dir_2d = np.array([-x_dir_2d[1], x_dir_2d[0]])

    x_axis = x_dir_2d[0] * axes_2d[0] + x_dir_2d[1] * axes_2d[1]
    y_axis = y_dir_2d[0] * axes_2d[0] + y_dir_2d[1] * axes_2d[1]

    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Step 5: Reconstruct 3D rotation and center
    R = np.column_stack([x_axis, y_axis, z_axis])
    center_3d = center_2d[0] * axes_2d[0] + center_2d[1] * axes_2d[1] + mean
    size = np.array(rect[1])  # width, height

    return center_3d, R, size

def apply_filtering_strategy(pcd_world, cfg):
    strategy = cfg.get("filter_strategy", "roi")
    if strategy == "roi":
        print("Filtering strategy: ROI")
        min_bound = np.array(cfg.get("roi_min_bound", [-1.5, -1.3, -0.85]))
        max_bound = np.array(cfg.get("roi_max_bound", [1.5, 1.3, 0]))
        return crop_point_cloud_by_roi(pcd_world, min_bound, max_bound)
    elif strategy == "full":
        print("Filtering strategy: FULL")
        return filter_point_cloud_full(pcd_world, cfg)
    else:
        raise ValueError(f"Unknown filtering strategy: {strategy}")


def estimate_and_visualize_boxes(pcd_filtered, cfg):
    geometries = [pcd_filtered]
    labels = np.array(pcd_filtered.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
    max_label = labels.max()

    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        if len(indices) < cfg.get("min_cluster_size", 100):
            continue

        cluster = pcd_filtered.select_by_index(indices)
        center, rotation, size = estimate_box_pose_from_cluster(cluster)

        # Create oriented box
        box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=0.01)
        box.paint_uniform_color([0, 1, 0])
        box.translate(-np.array([size[0] / 2, size[1] / 2, 0]))
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = center
        box.transform(T)
        geometries.append(box)

        # Add aligned coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.rotate(rotation, center=np.zeros(3))
        frame.translate(center)
        geometries.append(frame)

    return geometries

if __name__ == "__main__":
    cfg = load_config_file()
    depth, color, intrinsics, extrinsics = load_data(cfg)
    extrinsics[:3, 3] /= 1000.0

    pcd = create_point_cloud(depth, color, intrinsics)
    pcd_world = transform_point_cloud(pcd, extrinsics)

    pcd_filtered = apply_filtering_strategy(pcd_world, cfg)
    geometries = estimate_and_visualize_boxes(pcd_filtered, cfg)

    o3d.visualization.draw_geometries(geometries)

