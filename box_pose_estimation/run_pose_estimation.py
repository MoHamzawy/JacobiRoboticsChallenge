import yaml, os, numpy as np, open3d as o3d
import random
import copy

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

def cluster_and_colorize(pcd, eps=0.02, min_points=20):
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

def create_clipping_plane(z_height, size=1.0, color=[0.8, 0.1, 0.1], alpha=0.3):
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.001)
    mesh.paint_uniform_color(color)
    mesh.translate([-size/2, -size/2, z_height])
    mesh.compute_vertex_normals()
    return mesh

def filter_by_world_height(pcd, extrinsics, z_min=0.1):
    pcd_in_world = copy.deepcopy(pcd)
    pcd_in_world.transform(extrinsics)
    pts = np.asarray(pcd_in_world.points)
    mask = pts[:, 2] > z_min
    filtered = pcd_in_world.select_by_index(np.where(mask)[0])

    # Coordinate frame at world origin
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Coordinate frame at camera position
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    camera_frame.transform(extrinsics)

    # Add clipping plane at z_min
    clipping_plane = create_clipping_plane(z_min, size=1.0)
    return filtered

if __name__ == "__main__":
    cfg = load_config()
    depth, color, K, extrinsics = load_data(cfg)
    extrinsics[:3, 3] /= 1000.0
    cloud = create_point_cloud(depth, color, K)
    filtered = filter_by_world_height(cloud, extrinsics, z_min=-0.85)
    down_sampled = filtered.voxel_down_sample(voxel_size=0.005)
    clustered, labels = cluster_and_colorize(down_sampled)
    o3d.visualization.draw_geometries([clustered])
