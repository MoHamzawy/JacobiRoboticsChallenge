import yaml, os, numpy as np, open3d as o3d
import random

def load_config(path="box_pose_estimation/config.yaml"):
    with open(path) as f: 
        return yaml.safe_load(f)

def load_data(cfg):
    d = os.path.abspath(cfg["data_dir"])
    return (
        np.load(os.path.join(d, cfg["depth_file"])),
        np.load(os.path.join(d, cfg["color_file"])),
        np.load(os.path.join(d, cfg["intrinsics_file"])),
        np.load(os.path.join(d, cfg["extrinsics_file"]))
    )

def create_point_cloud(depth, color, k):
    if depth.dtype != np.uint16:
        depth = (depth * 1000).astype(np.uint16)
    if color.dtype != np.uint8:
        color = (color * 255).astype(np.uint8) if color.max() <= 1 else color.astype(np.uint8)
    if color.ndim == 2:
        color = np.stack([color] * 3, axis=-1)
    
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        convert_rgb_to_intensity=False
    )
    
    h, w = depth.shape
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, k[0,0], k[1,1], k[0,2], k[1,2])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intr)

    return pcd

def cluster_and_colorize(pcd, eps=0.02, min_points=50):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")

    colors = np.random.rand(max_label + 1, 3)
    colors = np.vstack([colors, np.zeros((1, 3))])  # Add black for noise label (-1)
    pcd.colors = o3d.utility.Vector3dVector(colors[labels])
    return pcd, labels

def analyze_clusters(pcd, labels):
    max_label = labels.max()
    print(f"\nCluster analysis (total: {max_label + 1}):\n")

    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        cluster = pcd.select_by_index(indices)
        obb = cluster.get_oriented_bounding_box()
        extent = obb.extent  # [width, height, depth] along principal axes

        print(f"Cluster {i}:")
        print(f"  Num Points: {len(indices)}")
        print(f"  Size (x, y, z): {extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}")
        print(f"  Center: {obb.center}\n")

if __name__ == "__main__":
    cfg = load_config()
    depth, color, K, extrinsics = load_data(cfg)
    cloud = create_point_cloud(depth, color, K)
    down_sampled = cloud.voxel_down_sample(voxel_size=0.005)
    clustered, labels = cluster_and_colorize(down_sampled)
    analyze_clusters(clustered, labels)
    o3d.visualization.draw_geometries([clustered])
