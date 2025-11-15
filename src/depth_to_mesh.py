"""
Depth estimation → Point Cloud → Mesh using MiDaS + Open3D.

Flow:
1) Load RGB image
2) Run MiDaS depth model (DPT_Large or MiDaS_small)
3) Create Open3D RGBD image
4) Generate point cloud
5) Estimate normals
6) Poisson surface reconstruction → mesh
7) Export to .obj/.ply/.glb
"""
import os
import numpy as np
from PIL import Image

# Constants for depth estimation and mesh processing
DEPTH_SCALE = 5.0                # Scale factor for MiDaS depth to visible range
DEPTH_TRUNCATE = 10.0            # Max depth truncation for RGBD image
NORMAL_ESTIMATION_RADIUS = 0.2   # Radius for normal estimation
NORMAL_MAX_NN = 50               # Max nearest neighbors for normal estimation
CAMERA_LOCATION = np.array([0.0, 0.0, -5.0])  # Camera location for normal orientation
DENSITY_QUANTILE = 0.2           # Quantile for removing low-density vertices
MESH_SMOOTH_ITERATIONS = 2       # Number of smoothing iterations
OUTLIER_NEIGHBORS = 20           # Statistical outlier removal neighbors
OUTLIER_STD_RATIO = 2.0          # Outlier removal std ratio


def estimate_depth_midas(image_pil: Image.Image, model_type: str = "DPT_Large") -> np.ndarray:
	"""Run MiDaS depth estimation on GPU if available. Returns depth map (H,W)."""
	import torch
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
	midas.to(device).eval()
	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
	if model_type in ["DPT_Large", "DPT_Hybrid"]:
		transform = midas_transforms.dpt_transform
	else:
		transform = midas_transforms.small_transform
	img_np = np.array(image_pil.convert("RGB"))
	input_batch = transform(img_np).to(device)
	with torch.no_grad():
		prediction = midas(input_batch)
		prediction = torch.nn.functional.interpolate(
			prediction.unsqueeze(1),
			size=img_np.shape[:2],
			mode="bicubic",
			align_corners=False,
		).squeeze()
	depth = prediction.cpu().numpy()
	# Normalize depth to [0,1] for stable reconstruction
	depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
	return depth


def depth_to_point_cloud(rgb: np.ndarray, depth: np.ndarray, fx: float = 800.0, fy: float = 800.0):
	"""Convert RGB + depth to Open3D point cloud. fx/fy tuned for better scale."""
	import open3d as o3d
	h, w = depth.shape
	cx, cy = w / 2.0, h / 2.0
	# Scale depth for better geometry (MiDaS outputs relative depth)
	depth_scaled = depth * DEPTH_SCALE
	color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
	depth_o3d = o3d.geometry.Image(depth_scaled.astype(np.float32))
	rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
		color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=DEPTH_TRUNCATE, convert_rgb_to_intensity=False
	)
	intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
	# Remove statistical outliers for cleaner mesh
	pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
	return pcd


def reconstruct_mesh_poisson(pcd, depth: int = 9):
	"""Poisson surface reconstruction with better normals and cleanup."""
	import open3d as o3d
	# Estimate normals with larger radius for smoother surface
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_ESTIMATION_RADIUS, max_nn=NORMAL_MAX_NN))
	pcd.orient_normals_towards_camera_location(camera_location=CAMERA_LOCATION)
	mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=True)
	# Remove low-density artifacts (more aggressive)
	vertices_to_remove = densities < np.quantile(densities, DENSITY_QUANTILE)
	mesh.remove_vertices_by_mask(vertices_to_remove)
	# Smooth and simplify
	mesh = mesh.filter_smooth_simple(number_of_iterations=MESH_SMOOTH_ITERATIONS)
	mesh.compute_vertex_normals()
	# Flip Y and Z axes to correct image-to-mesh orientation for display
	flip_matrix = np.eye(4, dtype=np.float64)
	flip_matrix[1, 1] = -1.0
	flip_matrix[2, 2] = -1.0
	mesh.transform(flip_matrix)
	return mesh


def image_to_mesh(image_pil: Image.Image, out_path: str, model_type: str = "DPT_Large", poisson_depth: int = 9) -> str:
	"""Full pipeline: image → depth → point cloud → mesh → export."""
	# 1) Depth estimation
	depth = estimate_depth_midas(image_pil, model_type=model_type)
	# 2) Point cloud
	rgb = np.array(image_pil.convert("RGB"))
	pcd = depth_to_point_cloud(rgb, depth)
	# 3) Mesh reconstruction
	mesh = reconstruct_mesh_poisson(pcd, depth=poisson_depth)
	# 4) Export
	import open3d as o3d
	o3d.io.write_triangle_mesh(out_path, mesh)
	return out_path

