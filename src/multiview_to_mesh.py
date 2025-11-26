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

# Constants - can be adjusted based on image count
DEFAULT_VOXEL_SIZE = 0.015
DEFAULT_OUTLIER_NEIGHBORS = 30
DEFAULT_OUTLIER_STD_RATIO = 2.5
DEFAULT_NORMAL_RADIUS = 0.08
DEFAULT_NORMAL_MAX_NN = 50
DEFAULT_NORMAL_TANGENT_SAMPLES = 50
DEFAULT_DENSITY_QUANTILE = 0.1
DEFAULT_MESH_SMOOTH_ITERATIONS = 3
DEFAULT_ICP_THRESHOLD = 0.05

# Global registration parameters
RANSAC_DISTANCE_THRESHOLD = 0.05
RANSAC_N_ITERATIONS = 100000
RANSAC_CONFIDENCE = 0.999


def get_adaptive_parameters(num_images: int) -> dict:
	"""Adjust reconstruction parameters based on number of input images."""
	params = {
		'voxel_size': DEFAULT_VOXEL_SIZE,
		'outlier_neighbors': DEFAULT_OUTLIER_NEIGHBORS,
		'outlier_std_ratio': DEFAULT_OUTLIER_STD_RATIO,
		'normal_radius': DEFAULT_NORMAL_RADIUS,
		'normal_max_nn': DEFAULT_NORMAL_MAX_NN,
		'normal_tangent_samples': DEFAULT_NORMAL_TANGENT_SAMPLES,
		'density_quantile': DEFAULT_DENSITY_QUANTILE,
		'mesh_smooth_iterations': DEFAULT_MESH_SMOOTH_ITERATIONS,
		'icp_threshold': DEFAULT_ICP_THRESHOLD,
	}
	
	# Adjust based on image count
	if num_images >= 8:
		# More images = can be more aggressive with filtering
		params['voxel_size'] = 0.012  # Finer detail
		params['density_quantile'] = 0.05  # Keep more points
		params['mesh_smooth_iterations'] = 2  # Less smoothing needed
	elif num_images <= 3:
		# Fewer images = need more conservative filtering
		params['voxel_size'] = 0.02  # Coarser to avoid gaps
		params['density_quantile'] = 0.15  # More aggressive filtering
		params['mesh_smooth_iterations'] = 4  # More smoothing
	
	return params



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


def global_registration(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, 
                        voxel_size: float) -> np.ndarray:
	"""Global registration using FPFH features and RANSAC."""
	try:
		# Downsample
		source_down = source.voxel_down_sample(voxel_size)
		target_down = target.voxel_down_sample(voxel_size)
		
		# Estimate normals
		radius_normal = voxel_size * 2
		source_down.estimate_normals(
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
		target_down.estimate_normals(
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
		
		# Compute FPFH features
		radius_feature = voxel_size * 5
		source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
			source_down,
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
		target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
			target_down,
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
		
		# RANSAC registration
		result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
			source_down, target_down, source_fpfh, target_fpfh,
			True, RANSAC_DISTANCE_THRESHOLD,
			o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
			3, [
				o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
				o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_DISTANCE_THRESHOLD)
			],
			o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_N_ITERATIONS, RANSAC_CONFIDENCE)
		)
		return result.transformation
	except Exception as e:
		print(f"Global registration failed: {e}, using identity")
		return np.eye(4)


def align_point_clouds_icp(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, 
                           init_transform: np.ndarray = None, voxel_size: float = DEFAULT_VOXEL_SIZE,
                           icp_threshold: float = DEFAULT_ICP_THRESHOLD) -> o3d.geometry.PointCloud:
	"""Align source point cloud to target using ICP with optional global registration."""
	if init_transform is None:
		init_transform = np.eye(4)
	
	# Try global registration first for better initialization
	if len(source.points) > 1000 and len(target.points) > 1000:
		global_transform = global_registration(source, target, voxel_size * 3)
		# Combine with initial transform
		init_transform = global_transform @ init_transform
	
	# Downsample for faster ICP
	source_down = source.voxel_down_sample(voxel_size=voxel_size * 2)
	target_down = target.voxel_down_sample(voxel_size=voxel_size * 2)
	
	# Estimate normals for point-to-plane ICP
	normal_radius = voxel_size * 5
	source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
	target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
	
	# Point-to-plane ICP
	try:
		reg = o3d.pipelines.registration.registration_icp(
			source_down, target_down, icp_threshold, init_transform,
			o3d.pipelines.registration.TransformationEstimationPointToPlane(),
			o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
		)
		
		# Apply transformation to original source
		source_aligned = source.transform(reg.transformation)
		print(f"    ICP fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")
		return source_aligned
	except Exception as e:
		print(f"ICP alignment failed: {e}, using initial transform")
		return source.transform(init_transform)


def images_to_point_cloud(images: List[Image.Image], progress_callback=None) -> o3d.geometry.PointCloud:
	"""
	Multi-view reconstruction with proper alignment.
	Uses feature matching + global registration + ICP to align point clouds.
	
	Args:
		images: List of PIL images from different viewpoints
		progress_callback: Optional callback function(step, total, message)
	"""
	import torch
	
	# Get adaptive parameters
	params = get_adaptive_parameters(len(images))
	
	def report_progress(step, total, msg):
		if progress_callback:
			progress_callback(step, total, msg)
		else:
			print(msg)
	
	# Standardize size - adaptive based on image count
	if len(images) >= 6:
		target_size = (640, 480)  # Lower res for many images to save memory
	else:
		target_size = (800, 600)  # Higher resolution for better quality
	
	images_resized = [img.convert("RGB").resize(target_size) for img in images]
	images_np = [np.array(img) for img in images_resized]
	
	# Load MiDaS once
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	report_progress(0, len(images) + 3, "Loading MiDaS depth model...")
	midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
	midas.to(device).eval()
	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
	transform = midas_transforms.dpt_transform
	
	# Generate point clouds for all images
	point_clouds = []
	report_progress(1, len(images) + 3, f"Generating point clouds from {len(images_np)} images...")
	
	for idx, img_np in enumerate(images_np):
		report_progress(2 + idx, len(images) + 3, f"Processing image {idx + 1}/{len(images_np)}...")
		depth = estimate_depth_midas(img_np, midas, transform, device)
		pcd = depth_to_pcd(img_np, depth)
		
		if not pcd.is_empty():
			point_clouds.append((pcd, img_np))
	
	if len(point_clouds) == 0:
		raise ValueError("Failed to create any point clouds")
	
	# Start with the first point cloud as reference
	report_progress(len(images) + 2, len(images) + 3, "Aligning and merging point clouds...")
	combined = point_clouds[0][0]
	
	# Align and merge subsequent point clouds
	for idx in range(1, len(point_clouds)):
		pcd, img_np = point_clouds[idx]
		prev_img = images_np[idx - 1]
		
		# Progress is already reported above
		
		# Try to estimate transformation from features
		T, success = estimate_transform_from_features(prev_img, img_np)
		
		if not success:
			# Fallback: use orbital positioning
			angle = 2.0 * np.pi * (idx / len(point_clouds))
			R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle, 0])
			T = np.eye(4)
			T[:3, :3] = R
			print(f"    Feature matching failed, using orbital positioning")
		
		# Align using ICP with initial transform and adaptive parameters
		pcd_aligned = align_point_clouds_icp(pcd, combined, T, 
		                                     voxel_size=params['voxel_size'],
		                                     icp_threshold=params['icp_threshold'])
		
		# Merge
		combined += pcd_aligned
	
	# Final cleanup and optimization
	report_progress(len(images) + 3, len(images) + 3, "Final point cloud cleanup...")
	combined = combined.voxel_down_sample(voxel_size=params['voxel_size'])
	
	if len(combined.points) > params['outlier_neighbors']:
		combined, _ = combined.remove_statistical_outlier(
			nb_neighbors=params['outlier_neighbors'], 
			std_ratio=params['outlier_std_ratio']
		)
	
	# Remove duplicate points
	combined = combined.remove_duplicated_points()
	combined = combined.remove_non_finite_points()
	
	print(f"Final point cloud has {len(combined.points)} points")
	
	return combined


def point_cloud_to_mesh(pcd: o3d.geometry.PointCloud, poisson_depth: int = 9, 
                        fill_holes: bool = True, progress_callback=None) -> o3d.geometry.TriangleMesh:
	"""
	High-quality Poisson surface reconstruction with post-processing.
	Creates a smooth, watertight mesh from the point cloud.
	
	Args:
		pcd: Input point cloud
		poisson_depth: Poisson reconstruction depth (6-12)
		fill_holes: Whether to attempt hole filling
		progress_callback: Optional callback function
	"""
	def report_progress(msg):
		if progress_callback:
			progress_callback(0, 1, msg)
		else:
			print(msg)
	
	# Get adaptive parameters
	num_points = len(pcd.points)
	params = get_adaptive_parameters(3)  # Use default params
	
	report_progress("Estimating normals for mesh reconstruction...")
	
	# Estimate normals with better parameters
	pcd.estimate_normals(
		search_param=o3d.geometry.KDTreeSearchParamHybrid(
			radius=params['normal_radius'],
			max_nn=params['normal_max_nn']
		)
	)
	
	# Orient normals consistently
	pcd.orient_normals_consistent_tangent_plane(params['normal_tangent_samples'])
	
	report_progress(f"Running Poisson reconstruction (depth={poisson_depth})...")
	
	# Poisson reconstruction with higher quality settings
	mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
		pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=True
	)
	
	# Remove low-density vertices (artifacts)
	report_progress("Cleaning mesh...")
	densities = np.asarray(densities)
	density_threshold = np.quantile(densities, params['density_quantile'])
	vertices_to_remove = densities < density_threshold
	mesh.remove_vertices_by_mask(vertices_to_remove)
	
	# Smooth the mesh
	mesh = mesh.filter_smooth_simple(number_of_iterations=params['mesh_smooth_iterations'])
	
	# Advanced smoothing with Laplacian
	if num_points > 10000:
		report_progress("Applying Laplacian smoothing...")
		mesh = mesh.filter_smooth_laplacian(number_of_iterations=2)
	
	# Additional cleanup
	mesh.remove_degenerate_triangles()
	mesh.remove_duplicated_triangles()
	mesh.remove_duplicated_vertices()
	mesh.remove_non_manifold_edges()
	
	# Fill holes if requested
	if fill_holes and len(mesh.vertices) > 0:
		try:
			report_progress("Filling holes...")
			# Simple hole filling by identifying boundary loops
			mesh = mesh.fill_holes()
		except AttributeError:
			# fill_holes not available in all Open3D versions
			pass
	
	# Recompute normals after cleanup
	mesh.compute_vertex_normals()
	
	# Flip for proper display orientation
	flip_matrix = np.eye(4, dtype=np.float64)
	flip_matrix[1, 1] = -1.0
	flip_matrix[2, 2] = -1.0
	mesh.transform(flip_matrix)
	
	report_progress(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
	
	return mesh


def multiview_to_mesh(images: List[Image.Image], out_path: str, poisson_depth: int = 9, 
                      fill_holes: bool = True, progress_callback=None) -> str:
	"""
	Multi-view 3D reconstruction: depth estimation + alignment + mesh generation.
	
	Args:
		images: List of PIL images from different viewpoints
		out_path: Output path for the mesh file
		poisson_depth: Poisson reconstruction depth (higher = more detail, 6-12 recommended)
		fill_holes: Whether to attempt hole filling in the mesh
		progress_callback: Optional callback function(step, total, message) for progress updates
	
	Returns:
		Path to the saved mesh file
	"""
	if len(images) < 2:
		raise ValueError("Need at least 2 images for multi-view reconstruction")
	
	print(f"\n=== Multi-view 3D Reconstruction ===")
	print(f"Input: {len(images)} images")
	print(f"Poisson depth: {poisson_depth}")
	
	# Generate aligned point cloud
	pcd = images_to_point_cloud(images, progress_callback=progress_callback)
	
	if pcd.is_empty():
		raise ValueError("Failed to create point cloud from images")
	
	# Generate mesh
	mesh = point_cloud_to_mesh(pcd, poisson_depth=poisson_depth, 
	                           fill_holes=fill_holes, progress_callback=progress_callback)
	
	# Save mesh
	print(f"Saving mesh to {out_path}...")
	o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False, compressed=True)
	
	print(f"✓ Multi-view reconstruction complete!")
	print(f"  Output: {out_path}")
	
	return out_path
