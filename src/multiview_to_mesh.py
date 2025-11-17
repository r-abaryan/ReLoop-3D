"""
Multi-view → 3D mesh using robust point cloud fusion.

Simplified approach:
1) Depth estimation per image (MiDaS)
2) Convert each depth to point cloud
3) Register point clouds using ICP alignment
4) Merge all point clouds
5) Mesh reconstruction via Poisson

More robust than pose-dependent TSDF.
"""
import os
import tempfile
from typing import List
import numpy as np
from PIL import Image
import open3d as o3d

# Constants for point cloud and mesh processing
VOXEL_SIZE = 0.02                # Point cloud voxel downsampling size
OUTLIER_NEIGHBORS = 20           # Statistical outlier removal neighbors
OUTLIER_STD_RATIO = 2.0          # Outlier removal std ratio
NORMAL_RADIUS = 0.1              # KDTree search radius for normals
NORMAL_MAX_NN = 30               # Max nearest neighbors for normal estimation
NORMAL_TANGENT_SAMPLES = 30      # Tangent plane samples for orientation
DENSITY_QUANTILE = 0.15          # Quantile for removing low-density vertices
MESH_SMOOTH_ITERATIONS = 2       # Number of smoothing iterations
DEPTH_SCALE = 4.0                # Depth scale for visible range


def estimate_depth_midas(image_np: np.ndarray, device) -> np.ndarray:
	"""Estimate depth using MiDaS model."""
	import torch
	midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
	midas.to(device).eval()
	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
	transform = midas_transforms.dpt_transform
	
	h, w = image_np.shape[:2]
	input_batch = transform(image_np).to(device)
	with torch.no_grad():
		prediction = midas(input_batch)
		prediction = torch.nn.functional.interpolate(
			prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
		).squeeze()
	depth = prediction.cpu().numpy()
	# Normalize to [0, 1]
	depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
	return depth


def depth_to_pcd(rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
	"""Convert RGB + depth to point cloud."""
	h, w = depth.shape
	
	# Create mesh grid
	x = np.linspace(-1, 1, w)
	y = np.linspace(-1, 1, h)
	xx, yy = np.meshgrid(x, y)
	
	# Scale depth
	zz = depth * DEPTH_SCALE
	
	# Stack into 3D coordinates
	points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
	colors = rgb.reshape(-1, 3) / 255.0
	
	# Create point cloud
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors)
	
	# Clean outliers
	pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
	
	return pcd


def register_pcds_icp(source: o3d.geometry.PointCloud, 
                       target: o3d.geometry.PointCloud) -> np.ndarray:
	"""Register source to target using ICP. Returns transformation matrix."""
	# Downsampling for faster ICP
	source_ds = source.voxel_down_sample(voxel_size=0.05)
	target_ds = target.voxel_down_sample(voxel_size=0.05)
	
	# Estimate normals
	source_ds.estimate_normals()
	target_ds.estimate_normals()
	
	# ICP registration
	try:
		result = o3d.pipelines.registration.registration_icp(
			source_ds, target_ds,
			max_correspondence_distance=0.3,
			init=np.eye(4),
			estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
			criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
				max_iteration=50,
				relative_fitness=1e-6,
				relative_rmse=1e-6
			)
		)
		return result.transformation
	except Exception:
		return np.eye(4)  # Fallback to identity


def images_to_point_cloud_robust(images: List[Image.Image]) -> o3d.geometry.PointCloud:
	"""Robust multi-view point cloud fusion without pose assumptions."""
	import torch
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Standardize image size
	target_size = (720, 540)
	images_resized = [img.convert("RGB").resize(target_size) for img in images]
	images_np = [np.array(img) for img in images_resized]
	h, w = images_np[0].shape[:2]
	
	# Load MiDaS once
	midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
	midas.to(device).eval()
	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
	transform = midas_transforms.dpt_transform
	
	# Estimate all depths and create point clouds
	pcds = []
	for idx, img_np in enumerate(images_np):
		print(f"Processing image {idx + 1}/{len(images_np)}...")
		
		# Depth estimation
		input_batch = transform(img_np).to(device)
		with torch.no_grad():
			prediction = midas(input_batch)
			prediction = torch.nn.functional.interpolate(
				prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
			).squeeze()
		depth = prediction.cpu().numpy()
		depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
		
		# Ensure dimensions match
		if depth.shape != (h, w):
			from PIL import Image as PILImage
			depth_img = PILImage.fromarray((depth * 255).astype(np.uint8))
			depth_img = depth_img.resize((w, h))
			depth = np.array(depth_img).astype(np.float32) / 255.0
		
		# Convert to point cloud
		pcd = depth_to_pcd(img_np, depth)
		pcds.append(pcd)
	
	# Register all point clouds to first one
	print("Registering point clouds...")
	for idx in range(1, len(pcds)):
		transformation = register_pcds_icp(pcds[idx], pcds[0])
		pcds[idx].transform(transformation)
	
	# Combine all point clouds
	print("Combining point clouds...")
	combined = o3d.geometry.PointCloud()
	for pcd in pcds:
		combined += pcd
	
	# Aggressive cleaning
	print("Cleaning merged point cloud...")
	combined = combined.voxel_down_sample(voxel_size=VOXEL_SIZE)
	combined, _ = combined.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
	combined, _ = combined.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.5)
	combined = combined.voxel_down_sample(voxel_size=VOXEL_SIZE * 1.5)
	
	return combined


def point_cloud_to_mesh(pcd: o3d.geometry.PointCloud, poisson_depth: int = 9) -> o3d.geometry.TriangleMesh:
	"""Convert point cloud to mesh using Poisson reconstruction."""
	pcd.estimate_normals(
		search_param=o3d.geometry.KDTreeSearchParamHybrid(
			radius=NORMAL_RADIUS,
			max_nn=NORMAL_MAX_NN
		)
	)
	pcd.orient_normals_consistent_tangent_plane(NORMAL_TANGENT_SAMPLES)
	
	mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
		pcd, depth=poisson_depth, linear_fit=True
	)
	
	# Remove low-density artifacts
	vertices_to_remove = densities < np.quantile(densities, DENSITY_QUANTILE)
	mesh.remove_vertices_by_mask(vertices_to_remove)
	mesh = mesh.filter_smooth_simple(number_of_iterations=MESH_SMOOTH_ITERATIONS)
	mesh.compute_vertex_normals()
	
	# Flip Y and Z axes to correct orientation
	flip_matrix = np.eye(4, dtype=np.float64)
	flip_matrix[1, 1] = -1.0
	flip_matrix[2, 2] = -1.0
	mesh.transform(flip_matrix)
	
	return mesh


def multiview_to_mesh(images: List[Image.Image], out_path: str, poisson_depth: int = 9) -> str:
	"""Full pipeline: multiple images → robust point cloud fusion → unified mesh."""
	if len(images) < 3:
		raise ValueError("Need at least 3 images for multi-view reconstruction")
	
	# Robust reconstruction with ICP registration
	pcd = images_to_point_cloud_robust(images)
	
	if pcd.is_empty():
		raise ValueError("Failed to create point cloud from images")
	
	print(f"Point cloud has {len(pcd.points)} points")
	
	# Convert to mesh
	mesh = point_cloud_to_mesh(pcd, poisson_depth=poisson_depth)
	o3d.io.write_triangle_mesh(out_path, mesh)
	
	return out_path
