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


def estimate_depth_midas(image_pil: Image.Image, model_type: str = "DPT_Large"):
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
	depth_scaled = depth * 5.0  # scale factor for visible range
	color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
	depth_o3d = o3d.geometry.Image(depth_scaled.astype(np.float32))
	rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
		color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
	)
	intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
	# Remove statistical outliers for cleaner mesh
	pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
	return pcd


def reconstruct_mesh_poisson(pcd, depth: int = 9):
	"""Poisson surface reconstruction with better normals and cleanup."""
	import open3d as o3d
	# Estimate normals with larger radius for smoother surface
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
	pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, -5.0]))
	mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=True)
	# Remove low-density artifacts (more aggressive)
	vertices_to_remove = densities < np.quantile(densities, 0.2)
	mesh.remove_vertices_by_mask(vertices_to_remove)
	# Smooth and simplify
	mesh = mesh.filter_smooth_simple(number_of_iterations=2)
	mesh.compute_vertex_normals()
	return mesh


def image_to_mesh(image_pil: Image.Image, out_path: str, model_type: str = "DPT_Large", poisson_depth: int = 9):
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

