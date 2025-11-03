import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def _try_import_open3d():
	try:
		import open3d as o3d  # type: ignore
		return o3d
	except Exception as exc:
		raise RuntimeError("open3d is required for 3D rendering. Install with pip install open3d") from exc


def render_views(
	model_path: str,
	image_size: int = 224,
	num_views: int = 4,
	background_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> List[Image.Image]:
	"""Render multiple views of a triangle mesh using Open3D offscreen renderer.

	Supports common mesh formats like .obj/.ply/.stl. Returns a list of PIL Images.
	"""
	o3d = _try_import_open3d()
	if not os.path.isfile(model_path):
		raise FileNotFoundError(f"3D model not found: {model_path}")

	mesh = o3d.io.read_triangle_mesh(model_path)
	if mesh is None or mesh.is_empty():
		raise ValueError("Failed to load a valid triangle mesh from the provided file.")
	mesh.compute_vertex_normals()

	# Create renderer and scene
	renderer = o3d.visualization.rendering.OffscreenRenderer(image_size, image_size)
	scene = renderer.scene
	scene.set_background(background_rgba)

	# Default material
	material = o3d.visualization.rendering.MaterialRecord()
	material.shader = "defaultLit"

	# Add mesh
	mesh_name = "mesh"
	scene.add_geometry(mesh_name, mesh, material)

	# Fit camera to mesh bounds
	bbox = mesh.get_axis_aligned_bounding_box()
	center = bbox.get_center()
	extent = bbox.get_extent()
	diagonal = float(np.linalg.norm(extent))
	radius = max(diagonal, 1e-3)

	# Lighting
	scene.scene.remove_all_lights()
	scene.scene.set_sun_light([1.0, 1.0, 1.0], 6500, 1.5)
	scene.scene.enable_sun_light(True)

	# Generate evenly spaced azimuths
	views: List[Image.Image] = []
	for i in range(max(1, num_views)):
		az = 2.0 * np.pi * (i / max(1, num_views))
		elev = np.deg2rad(20.0)
		dist = 2.2 * radius
		x = center[0] + dist * np.cos(elev) * np.cos(az)
		y = center[1] + dist * np.sin(elev)
		z = center[2] + dist * np.cos(elev) * np.sin(az)
		cam_pos = np.array([x, y, z], dtype=np.float32)
		up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

		camera = o3d.visualization.rendering.Camera()
		# Set perspective camera
		fov = 60.0
		camera.set_projection(fov, image_size, image_size, 0.01, 1000.0)
		scene.camera.set_projection(fov, image_size, image_size, 0.01, 1000.0)
		scene.camera.look_at(center, cam_pos, up)

		img_o3d = renderer.render_to_image()
		img_np = np.asarray(img_o3d)
		# Ensure RGB
		if img_np.shape[-1] == 4:
			img_np = img_np[:, :, :3]
		img = Image.fromarray(img_np)
		views.append(img)

	# Cleanup
	renderer.scene.clear_geometry()
	renderer.release()
	return views



