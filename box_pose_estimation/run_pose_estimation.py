import yaml, os, numpy as np, open3d as o3d

def load_config(path="box_pose_estimation/config.yaml"):
    with open(path) as f: 
        return yaml.safe_load(f)

def load_data(cfg):
    d = os.path.abspath(cfg["data_dir"])
    return (
        np.load(os.path.join(d, cfg["depth_file"])),
        np.load(os.path.join(d, cfg["color_file"])),
        np.load(os.path.join(d, cfg["intrinsics_file"]))
    )

def rgbd(depth, color):
    if depth.dtype != np.uint16: depth = (depth * 1000).astype(np.uint16)
    if color.dtype != np.uint8:
        color = (color * 255).astype(np.uint8) if color.max() <= 1 else color.astype(np.uint8)
    if color.ndim == 2: color = np.stack([color] * 3, axis=-1)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        convert_rgb_to_intensity=False
    )

def cam_intrinsics(k, w, h):
    return o3d.camera.PinholeCameraIntrinsic(w, h, k[0,0], k[1,1], k[0,2], k[1,2])

def estimate_pose(depth, color, k, thresh=0.005):
    rgbd_img = rgbd(depth, color)
    h, w = depth.shape
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, cam_intrinsics(k, w, h))
    plane, idx = pcd.segment_plane(distance_threshold=thresh, ransac_n=3, num_iterations=1000)
    top = pcd.select_by_index(idx)
    obb = top.get_oriented_bounding_box()
    normal = plane[:3] / np.linalg.norm(plane[:3])
    R = obb.R.copy()
    if np.dot(R[:,2], normal) < 0: R[:,0] *= -1; R[:,1] *= -1; R[:,2] *= -1
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = obb.center
    return T, obb, pcd

def visualise(pcd, obb):
    o3d.visualization.draw_geometries([pcd, obb])

if __name__ == "__main__":
    cfg = load_config()
    depth, color, K = load_data(cfg)
    pose, box, cloud = estimate_pose(depth, color, K)
    print("Camera → object transformation:\n", pose)
    visualise(cloud, box)
