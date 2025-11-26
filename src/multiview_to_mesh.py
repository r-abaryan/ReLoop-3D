"""
Multi-view → 3D mesh using proper point cloud alignment and merging.

Improved approach:
1) Estimate depth per image (MiDaS)
2) Convert each depth to point cloud
3) Use feature matching to estimate relative poses (or intelligent placement)
4) Align and merge point clouds using ICP
5) Create high-quality mesh via Poisson reconstruction

This creates a single unified mesh instead of disconnected pieces.
"""
from typing import List, Tuple
import numpy as np
from PIL import Image
import open3d as o3d
import cv2
import torch

# Constants
VOXEL_SIZE = 0.015  # Finer voxel size for better quality
OUTLIER_NEIGHBORS = 30
OUTLIER_STD_RATIO = 2.5
NORMAL_RADIUS = 0.08
NORMAL_MAX_NN = 50
NORMAL_TANGENT_SAMPLES = 50
DENSITY_QUANTILE = 0.1  # Keep more points
MESH_SMOOTH_ITERATIONS = 3
ICP_THRESHOLD = 0.05  # For point cloud alignment


def estimate_depth_midas(image_np: np.ndarray, midas, transform, device) -> np.ndarray:
	"""MiDaS depth estimation (reuses loaded model)."""
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


def depth_to_pcd(rgb: np.ndarray, depth: np.ndarray, depth_scale: float = 3.0) -> o3d.geometry.PointCloud:
	"""Convert RGB + depth to point cloud with improved parameters."""
	h, w = depth.shape
	
	# Scale depth for better 3D structure
	depth_scaled = depth * depth_scale
	
	# Create Open3D images
	color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
	depth_o3d = o3d.geometry.Image(depth_scaled.astype(np.float32))
	
	# Camera intrinsics (realistic focal length)
	fx = fy = max(w, h) * 1.0
	cx, cy = w / 2.0, h / 2.0
	intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
	
	# Create RGBD image
	rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
		color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=15.0, convert_rgb_to_intensity=False
	)
	
	# Convert to point cloud
	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
	
	# Remove outliers
	if len(pcd.points) > OUTLIER_NEIGHBORS:
		pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
	
	return pcd


def estimate_transform_from_features(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, bool]:
	"""
	Estimate relative transformation between two images using feature matching.
	Returns (4x4 transformation matrix, success_flag).
	"""
	try:
		# Convert to grayscale
		gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
		gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
		
		# Detect SIFT features
		sift = cv2.SIFT_create(nfeatures=2000)
		kp1, des1 = sift.detectAndCompute(gray1, None)
		kp2, des2 = sift.detectAndCompute(gray2, None)
		
		if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
			return np.eye(4), False
		
		# Match features
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
		matches = bf.knnMatch(des1, des2, k=2)
		
		# Apply ratio test
		good_matches = []
		for match_pair in matches:
			if len(match_pair) == 2:
				m, n = match_pair
				if m.distance < 0.7 * n.distance:
					good_matches.append(m)
		
		if len(good_matches) < 20:
			return np.eye(4), False
		
		# Extract matched keypoints
		pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
		pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
		
		# Estimate essential matrix (assuming calibrated cameras)
		h, w = gray1.shape
		focal = max(h, w) * 1.0
		pp = (w / 2.0, h / 2.0)
		E, mask = cv2.findEssentialMat(pts1, pts2, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		
		if E is None:
			return np.eye(4), False
		
		# Recover pose
		_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, focal=focal, pp=pp)
		
		# Build 4x4 transformation matrix
		T = np.eye(4)
		T[:3, :3] = R
		T[:3, 3] = t.flatten()
		
		return T, True
		
	except Exception as e:
		print(f"Feature matching failed: {e}")
		return np.eye(4), False


def align_point_clouds_icp(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, 
                           init_transform: np.ndarray = None) -> o3d.geometry.PointCloud:
	"""Align source point cloud to target using ICP."""
	if init_transform is None:
		init_transform = np.eye(4)
	
	# Downsample for faster ICP
	source_down = source.voxel_down_sample(voxel_size=VOXEL_SIZE * 2)
	target_down = target.voxel_down_sample(voxel_size=VOXEL_SIZE * 2)
	
	# Estimate normals for point-to-plane ICP
	source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS * 2, max_nn=30))
	target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS * 2, max_nn=30))
	
	# Point-to-plane ICP
	try:
		reg = o3d.pipelines.registration.registration_icp(
			source_down, target_down, ICP_THRESHOLD, init_transform,
			o3d.pipelines.registration.TransformationEstimationPointToPlane(),
			o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
		)
		
		# Apply transformation to original source
		source_aligned = source.transform(reg.transformation)
		return source_aligned
	except Exception as e:
		print(f"ICP alignment failed: {e}, using initial transform")
		return source.transform(init_transform)


def images_to_point_cloud(images: List[Image.Image]) -> o3d.geometry.PointCloud:
	"""
	Multi-view reconstruction with proper alignment.
	Uses feature matching + ICP to align point clouds from different views.
	"""
	import torch
	
	# Standardize size
	target_size = (800, 600)  # Higher resolution for better quality
	images_resized = [img.convert("RGB").resize(target_size) for img in images]
	images_np = [np.array(img) for img in images_resized]
	
	# Load MiDaS once
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Loading MiDaS depth model...")
	midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
	midas.to(device).eval()
	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
	transform = midas_transforms.dpt_transform
	
	# Generate point clouds for all images
	point_clouds = []
	print(f"Generating point clouds from {len(images_np)} images...")
	
	for idx, img_np in enumerate(images_np):
		print(f"  Processing image {idx + 1}/{len(images_np)}...")
		depth = estimate_depth_midas(img_np, midas, transform, device)
		pcd = depth_to_pcd(img_np, depth)
		
		if not pcd.is_empty():
			point_clouds.append((pcd, img_np))
	
	if len(point_clouds) == 0:
		raise ValueError("Failed to create any point clouds")
	
	# Start with the first point cloud as reference
	print("Aligning and merging point clouds...")
	combined = point_clouds[0][0]
	
	# Align and merge subsequent point clouds
	for idx in range(1, len(point_clouds)):
		pcd, img_np = point_clouds[idx]
		prev_img = images_np[idx - 1]
		
		print(f"  Aligning point cloud {idx + 1}/{len(point_clouds)}...")
		
		# Try to estimate transformation from features
		T, success = estimate_transform_from_features(prev_img, img_np)
		
		if not success:
			# Fallback: use orbital positioning
			angle = 2.0 * np.pi * (idx / len(point_clouds))
			R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle, 0])
			T = np.eye(4)
			T[:3, :3] = R
			print(f"    Feature matching failed, using orbital positioning")
		
		# Align using ICP with initial transform
		pcd_aligned = align_point_clouds_icp(pcd, combined, T)
		
		# Merge
		combined += pcd_aligned
	
	# Final cleanup and optimization
	print("Final point cloud cleanup...")
	combined = combined.voxel_down_sample(voxel_size=VOXEL_SIZE)
	
	if len(combined.points) > OUTLIER_NEIGHBORS:
		combined, _ = combined.remove_statistical_outlier(
			nb_neighbors=OUTLIER_NEIGHBORS, 
			std_ratio=OUTLIER_STD_RATIO
		)
	
	# Remove duplicate points
	combined = combined.remove_duplicated_points()
	combined = combined.remove_non_finite_points()
	
	print(f"Final point cloud has {len(combined.points)} points")
	
	return combined


def point_cloud_to_mesh(pcd: o3d.geometry.PointCloud, poisson_depth: int = 9) -> o3d.geometry.TriangleMesh:
	"""
	High-quality Poisson surface reconstruction.
	Creates a smooth, watertight mesh from the point cloud.
	"""
	print("Estimating normals for mesh reconstruction...")
	
	# Estimate normals with better parameters
	pcd.estimate_normals(
		search_param=o3d.geometry.KDTreeSearchParamHybrid(
			radius=NORMAL_RADIUS,
			max_nn=NORMAL_MAX_NN
		)
	)
	
	# Orient normals consistently
	pcd.orient_normals_consistent_tangent_plane(NORMAL_TANGENT_SAMPLES)
	
	print(f"Running Poisson reconstruction (depth={poisson_depth})...")
	
	# Poisson reconstruction with higher quality settings
	mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
		pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=True
	)
	
	# Remove low-density vertices (artifacts)
	print("Cleaning mesh...")
	densities = np.asarray(densities)
	density_threshold = np.quantile(densities, DENSITY_QUANTILE)
	vertices_to_remove = densities < density_threshold
	mesh.remove_vertices_by_mask(vertices_to_remove)
	
	# Smooth the mesh
	mesh = mesh.filter_smooth_simple(number_of_iterations=MESH_SMOOTH_ITERATIONS)
	
	# Additional cleanup
	mesh.remove_degenerate_triangles()
	mesh.remove_duplicated_triangles()
	mesh.remove_duplicated_vertices()
	mesh.remove_non_manifold_edges()
	
	# Recompute normals after cleanup
	mesh.compute_vertex_normals()
	
	# Flip for proper display orientation
	flip_matrix = np.eye(4, dtype=np.float64)
	flip_matrix[1, 1] = -1.0
	flip_matrix[2, 2] = -1.0
	mesh.transform(flip_matrix)
	
	print(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
	
	return mesh


def multiview_to_mesh(images: List[Image.Image], out_path: str, poisson_depth: int = 9) -> str:
	"""
	Multi-view 3D reconstruction: depth estimation + alignment + mesh generation.
	
	Args:
		images: List of PIL images from different viewpoints
		out_path: Output path for the mesh file
		poisson_depth: Poisson reconstruction depth (higher = more detail, 6-12 recommended)
	
	Returns:
		Path to the saved mesh file
	"""
	if len(images) < 2:
		raise ValueError("Need at least 2 images for multi-view reconstruction")
	
	print(f"\n=== Multi-view 3D Reconstruction ===")
	print(f"Input: {len(images)} images")
	print(f"Poisson depth: {poisson_depth}")
	
	# Generate aligned point cloud
	pcd = images_to_point_cloud(images)
	
	if pcd.is_empty():
		raise ValueError("Failed to create point cloud from images")
	
	# Generate mesh
	mesh = point_cloud_to_mesh(pcd, poisson_depth=poisson_depth)
	
	# Save mesh
	print(f"Saving mesh to {out_path}...")
	o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False, compressed=True)
	
	print(f"✓ Multi-view reconstruction complete!")
	print(f"  Output: {out_path}")
	
	return out_path
