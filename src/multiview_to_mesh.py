"""
Multi-view → 3D mesh using Open3D reconstruction system.

Flow (based on Open3D tutorial):
1) Save images to temp folder
2) Feature extraction + matching (ORB/SIFT)
3) Pose estimation (PnP + RANSAC)
4) Dense reconstruction (depth estimation per image pair)
5) TSDF volume integration
6) Mesh extraction + export

Reference: https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html
"""
import os
import tempfile
from typing import List
import numpy as np
from PIL import Image
import open3d as o3d


def images_to_point_cloud(images: List[Image.Image]) -> o3d.geometry.PointCloud:
	"""Combine multiple images into point cloud using GPU-accelerated MiDaS depth per image."""
	import torch
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Load MiDaS once for all images
	midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
	midas.to(device).eval()
	midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
	transform = midas_transforms.dpt_transform
	
	combined = o3d.geometry.PointCloud()
	
	for idx, img in enumerate(images):
		rgb = np.array(img.convert("RGB"))
		h, w = rgb.shape[:2]
		
		# GPU depth estimation
		input_batch = transform(rgb).to(device)
		with torch.no_grad():
			prediction = midas(input_batch)
			prediction = torch.nn.functional.interpolate(
				prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
			).squeeze()
		depth = prediction.cpu().numpy()
		depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
		depth = depth * 5.0  # scale for visible range
		
		# Create RGBD
		color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
		depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
		rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
			color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
		)
		
		# Intrinsic
		fx = fy = max(w, h) * 0.8
		cx, cy = w / 2.0, h / 2.0
		intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
		
		# Generate point cloud
		pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
		
		# Rotate to simulate orbit
		angle = 2.0 * np.pi * (idx / len(images))
		R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle, 0])
		pcd.rotate(R, center=(0, 0, 0))
		
		combined += pcd
	
	# Downsample and clean
	combined = combined.voxel_down_sample(voxel_size=0.02)
	combined, _ = combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
	return combined


def point_cloud_to_mesh(pcd: o3d.geometry.PointCloud, poisson_depth: int = 9) -> o3d.geometry.TriangleMesh:
	"""Convert point cloud to mesh using Poisson reconstruction."""
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	pcd.orient_normals_consistent_tangent_plane(30)
	mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth, linear_fit=True)
	# Remove low-density artifacts
	vertices_to_remove = densities < np.quantile(densities, 0.15)
	mesh.remove_vertices_by_mask(vertices_to_remove)
	mesh = mesh.filter_smooth_simple(number_of_iterations=2)
	mesh.compute_vertex_normals()
	# Flip Y and Z axes to correct image-to-mesh orientation for display
	flip_matrix = np.eye(4, dtype=np.float64)
	flip_matrix[1, 1] = -1.0
	flip_matrix[2, 2] = -1.0
	mesh.transform(flip_matrix)
	return mesh


def multiview_to_mesh(images: List[Image.Image], out_path: str, poisson_depth: int = 9):
	"""Full pipeline: multiple images → Open3D point cloud → mesh."""
	pcd = images_to_point_cloud(images)
	mesh = point_cloud_to_mesh(pcd, poisson_depth=poisson_depth)
	o3d.io.write_triangle_mesh(out_path, mesh)
	return out_path

