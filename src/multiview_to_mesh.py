"""
Multi-view → 3D mesh using COLMAP + Open3D.

Flow:
1) Save uploaded images to temp folder
2) Run COLMAP feature extraction + matching + sparse reconstruction
3) Run COLMAP dense reconstruction (MVS)
4) Convert COLMAP dense point cloud to Open3D
5) Poisson surface reconstruction → mesh
6) Export to .obj/.ply/.glb

Requires COLMAP installed: https://colmap.github.io/install.html
"""
import os
import subprocess
import tempfile
from typing import List
import numpy as np
from PIL import Image


def run_colmap_reconstruction(image_folder: str, workspace: str):
	"""Run COLMAP sparse + dense reconstruction. Returns dense point cloud path."""
	# Check COLMAP availability
	try:
		subprocess.run(["colmap", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
	except Exception:
		raise RuntimeError("COLMAP not found. Install from https://colmap.github.io/install.html")
	
	db_path = os.path.join(workspace, "database.db")
	sparse_dir = os.path.join(workspace, "sparse")
	dense_dir = os.path.join(workspace, "dense")
	os.makedirs(sparse_dir, exist_ok=True)
	os.makedirs(dense_dir, exist_ok=True)
	
	# Feature extraction
	subprocess.run([
		"colmap", "feature_extractor",
		"--database_path", db_path,
		"--image_path", image_folder,
		"--ImageReader.single_camera", "1"
	], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	
	# Feature matching
	subprocess.run([
		"colmap", "exhaustive_matcher",
		"--database_path", db_path
	], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	
	# Sparse reconstruction
	subprocess.run([
		"colmap", "mapper",
		"--database_path", db_path,
		"--image_path", image_folder,
		"--output_path", sparse_dir
	], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	
	# Dense reconstruction
	subprocess.run([
		"colmap", "image_undistorter",
		"--image_path", image_folder,
		"--input_path", os.path.join(sparse_dir, "0"),
		"--output_path", dense_dir,
		"--output_type", "COLMAP"
	], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	
	subprocess.run([
		"colmap", "patch_match_stereo",
		"--workspace_path", dense_dir,
		"--workspace_format", "COLMAP"
	], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	
	subprocess.run([
		"colmap", "stereo_fusion",
		"--workspace_path", dense_dir,
		"--workspace_format", "COLMAP",
		"--input_type", "geometric",
		"--output_path", os.path.join(dense_dir, "fused.ply")
	], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	
	return os.path.join(dense_dir, "fused.ply")


def colmap_to_open3d_mesh(ply_path: str, out_path: str, poisson_depth: int = 9):
	"""Load COLMAP point cloud, clean, reconstruct mesh, export."""
	import open3d as o3d
	pcd = o3d.io.read_point_cloud(ply_path)
	# Remove outliers
	pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
	# Estimate normals
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	pcd.orient_normals_consistent_tangent_plane(30)
	# Poisson reconstruction
	mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth, linear_fit=True)
	vertices_to_remove = densities < np.quantile(densities, 0.1)
	mesh.remove_vertices_by_mask(vertices_to_remove)
	mesh = mesh.filter_smooth_simple(number_of_iterations=2)
	mesh.compute_vertex_normals()
	o3d.io.write_triangle_mesh(out_path, mesh)
	return out_path


def multiview_to_mesh(images: List[Image.Image], out_path: str, poisson_depth: int = 9):
	"""Full pipeline: multiple images → COLMAP → mesh."""
	with tempfile.TemporaryDirectory() as tmpdir:
		img_dir = os.path.join(tmpdir, "images")
		os.makedirs(img_dir, exist_ok=True)
		# Save images
		for i, img in enumerate(images):
			img.save(os.path.join(img_dir, f"img_{i:04d}.jpg"))
		# Run COLMAP
		ply_path = run_colmap_reconstruction(img_dir, tmpdir)
		# Convert to mesh
		colmap_to_open3d_mesh(ply_path, out_path, poisson_depth=poisson_depth)
	return out_path

