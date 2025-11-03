import os
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


def _try_import_open3d():
	try:
		import open3d as o3d  # type: ignore
		return o3d
	except Exception as exc:
		raise RuntimeError("open3d is required for 3D rendering. Install with pip install open3d") from exc


def _convert_to_glb_for_render(model_path: str) -> str:
	"""Convert arbitrary mesh to GLB for consistent rendering; fallback to original path."""
	try:
		import trimesh  # type: ignore
		from pathlib import Path
		if not model_path:
			return model_path
		src = Path(model_path)
		if not src.exists():
			return model_path
		cache_dir = Path(".render_cache")
		cache_dir.mkdir(parents=True, exist_ok=True)
		out_path = cache_dir / (src.stem + ".glb")
		loaded = trimesh.load(str(src), force='scene')
		if isinstance(loaded, trimesh.Scene):
			scene = loaded
		else:
			scene = trimesh.Scene(loaded)
		# Normalize to origin and unit diagonal to keep camera behavior stable across backends
		try:
			bounds = scene.bounds
			minb = np.array(bounds[0]); maxb = np.array(bounds[1])
			center = (minb + maxb) / 2.0
			diag = float(np.linalg.norm(maxb - minb))
			scale = (1.0 / max(diag, 1e-6))
			T = np.eye(4, dtype=np.float32)
			T[:3, 3] = -center
			S = np.eye(4, dtype=np.float32)
			S[0,0] = S[1,1] = S[2,2] = scale
			scene.apply_transform(T @ S)
		except Exception:
			pass
		scene.export(str(out_path))
		if out_path.exists() and out_path.stat().st_size > 0:
			return str(out_path)
		return model_path
	except Exception:
		return model_path


def render_views(
	model_path: str,
	image_size: int = 224,
	num_views: int = 4,
	background_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
	blender_path: Optional[str] = None,
) -> List[Image.Image]:
	"""Render multiple views of a 3D model and return PIL Images.

	Backend strategy:
	1) If blender_path is provided, try Blender first (best results for complex materials)
	2) Try Open3D Visualizer offscreen
	3) Fallback to pyrender + trimesh
	4) Last resort: matplotlib (CPU-only)
	"""
	last_error: Optional[Exception] = None

	# Pre-normalize/convert to GLB to match web preview and stabilize transforms
	model_for_render = _convert_to_glb_for_render(model_path)

	# 1) Blender (optional, explicit)
	if blender_path:
		try:
			return _render_views_blender(
				blender_path=blender_path,
				model_path=model_for_render,
				image_size=image_size,
				num_views=num_views,
				background_rgba=background_rgba,
			)
		except Exception as exc:
			last_error = exc

	# 2) Open3D Visualizer offscreen
	try:
		import open3d as o3d  # type: ignore
		if not os.path.isfile(model_for_render):
			raise FileNotFoundError(f"3D model not found: {model_for_render}")
		mesh = o3d.io.read_triangle_mesh(model_for_render)
		if mesh is None or mesh.is_empty():
			raise ValueError("Failed to load mesh for Open3D Visualizer")
		if not mesh.has_vertex_normals():
			mesh.compute_vertex_normals()

		vis = o3d.visualization.Visualizer()
		vis.create_window(visible=False, width=image_size, height=image_size)
		vis.add_geometry(mesh)
		vis.update_geometry(mesh)
		opt = vis.get_render_option()
		if opt is not None:
			opt.background_color = np.array(background_rgba[:3], dtype=np.float32)
		vis.poll_events(); vis.update_renderer()

		bbox = mesh.get_axis_aligned_bounding_box()
		center = bbox.get_center()
		extent = bbox.get_extent()
		diagonal = float(np.linalg.norm(extent))
		radius = max(diagonal, 1e-3)

		views: List[Image.Image] = []
		vc = vis.get_view_control()
		for i in range(max(1, num_views)):
			if vc is not None:
				az = 2.0 * np.pi * (i / max(1, num_views))
				elev = np.deg2rad(20.0)
				dist = 2.2 * radius
				x = center[0] + dist * np.cos(elev) * np.cos(az)
				y = center[1] + dist * np.sin(elev)
				z = center[2] + dist * np.cos(elev) * np.sin(az)
				cam_pos = np.array([x, y, z], dtype=np.float32)
				front = (center - cam_pos)
				front = front / (np.linalg.norm(front) + 1e-8)
				up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
				vc.set_lookat(center)
				vc.set_front(front)
				vc.set_up(up)
				vc.set_zoom(0.7)
			vis.poll_events(); vis.update_renderer()
			img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
			img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
			views.append(Image.fromarray(img))

		vis.destroy_window()
		if views:
			return views
		raise RuntimeError("Open3D produced no views")
	except Exception as exc:
		last_error = exc

	# 3) pyrender fallback
	try:
		return _render_views_pyrender(
			model_path=model_for_render,
			image_size=image_size,
			num_views=num_views,
			background_rgba=background_rgba,
		)
	except Exception as exc:
		last_error = exc

	# 4) matplotlib fallback (CPU-only)
	try:
		return _render_views_matplotlib(
			model_path=model_for_render,
			image_size=image_size,
			num_views=num_views,
		)
	except Exception as exc:
		last_error = exc

	raise RuntimeError(f"All render backends failed. Last error: {last_error}")


def _render_views_pyrender(
	model_path: str,
	image_size: int = 224,
	num_views: int = 4,
	background_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> List[Image.Image]:
	"""Fallback renderer using trimesh + pyrender (generally works on Windows).

	Requires: pip install trimesh pyrender
	"""
	try:
		import trimesh  # type: ignore
		import pyrender  # type: ignore
	except Exception as exc:
		raise RuntimeError("pyrender + trimesh are required for fallback rendering. Install with pip install pyrender trimesh") from exc

	if not os.path.isfile(model_path):
		raise FileNotFoundError(f"3D model not found: {model_path}")

	# Load with trimesh; preserve materials/textures
	loaded = trimesh.load(model_path, force='scene')
	if isinstance(loaded, trimesh.Scene):
		if len(loaded.geometry) == 0:
			raise ValueError("Empty scene in fallback renderer.")
		# Convert scene to a single mesh with visuals baked when possible
		mesh_tm = loaded.dump().sum()
	else:
		mesh_tm = loaded
	if mesh_tm is None or mesh_tm.is_empty:
		raise ValueError("Failed to load a valid mesh for fallback renderer.")
	# Ensure triangulated
	if not mesh_tm.is_watertight:
		try:
			mesh_tm = mesh_tm.subdivide()
		except Exception:
			pass

	mesh_pr = pyrender.Mesh.from_trimesh(mesh_tm, smooth=True)
	scene = pyrender.Scene(bg_color=np.array(background_rgba, dtype=np.float32), ambient_light=(0.3, 0.3, 0.3))
	scene.add(mesh_pr)

	# Compute rough radius from bounds
	bbox = mesh_tm.bounds if hasattr(mesh_tm, 'bounds') else None
	if bbox is not None:
		minb = np.array(bbox[0]); maxb = np.array(bbox[1])
		center = (minb + maxb) / 2.0
		extent = (maxb - minb)
		radius = float(np.linalg.norm(extent))
	else:
		center = np.zeros(3, dtype=np.float32)
		radius = 1.0
	radius = max(radius, 1e-3)

	# Lights
	key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
	fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
	back = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
	scene.add(key, pose=np.eye(4))
	scene.add(fill, pose=_look_at(center + np.array([2.0, 1.0, 2.0]), center, np.array([0.0, 1.0, 0.0])))
	scene.add(back, pose=_look_at(center + np.array([-2.0, 1.0, -2.0]), center, np.array([0.0, 1.0, 0.0])))

	r = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
	views: List[Image.Image] = []
	for i in range(max(1, num_views)):
		az = 2.0 * np.pi * (i / max(1, num_views))
		elev = np.deg2rad(20.0)
		dist = 2.2 * radius
		x = center[0] + dist * np.cos(elev) * np.cos(az)
		y = center[1] + dist * np.sin(elev)
		z = center[2] + dist * np.cos(elev) * np.sin(az)
		cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0))
		cam_tf = _look_at(np.array([x, y, z]), center, np.array([0.0, 1.0, 0.0]))
		cam_node = scene.add(cam, pose=cam_tf)
		color, _ = r.render(scene)
		scene.remove_node(cam_node)
		img = Image.fromarray(color)
		views.append(img)

	r.delete()
	return views


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
	"""Build a right-handed look-at matrix for pyrender."""
	forward = (target - eye)
	forward = forward / (np.linalg.norm(forward) + 1e-8)
	right = np.cross(forward, up)
	right = right / (np.linalg.norm(right) + 1e-8)
	true_up = np.cross(right, forward)
	mat = np.eye(4, dtype=np.float32)
	mat[0, :3] = right
	mat[1, :3] = true_up
	mat[2, :3] = -forward
	mat[:3, 3] = eye
	return mat


def _render_views_matplotlib(
	model_path: str,
	image_size: int = 224,
	num_views: int = 4,
) -> List[Image.Image]:
	"""CPU-only fallback using matplotlib Poly3DCollection. No GL context required."""
	import trimesh  # type: ignore
	from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
	import matplotlib.pyplot as plt
	from io import BytesIO

	loaded = trimesh.load(model_path, force='scene')
	if isinstance(loaded, trimesh.Scene):
		if len(loaded.geometry) == 0:
			raise ValueError("Empty scene for matplotlib fallback.")
		mesh_tm = loaded.dump().sum()
	else:
		mesh_tm = loaded
	if mesh_tm is None or mesh_tm.is_empty:
		raise ValueError("Failed to load a valid mesh for matplotlib fallback.")

	verts = np.asarray(mesh_tm.vertices)
	faces = np.asarray(mesh_tm.faces)
	center = verts.mean(axis=0)
	extent = verts.max(axis=0) - verts.min(axis=0)
	radius = float(np.linalg.norm(extent))
	radius = max(radius, 1e-3)

	views: List[Image.Image] = []
	for i in range(max(1, num_views)):
		fig = plt.figure(figsize=(image_size/100.0, image_size/100.0), dpi=100)
		ax = fig.add_subplot(111, projection='3d')
		ax.axis('off')
		mesh_verts = verts[faces]
		poly = Poly3DCollection(mesh_verts, linewidths=0.05, alpha=1.0)
		poly.set_facecolor((0.8, 0.8, 0.8, 1.0))
		poly.set_edgecolor((0.2, 0.2, 0.2, 0.3))
		ax.add_collection3d(poly)
		max_range = radius
		x, y, z = center
		ax.set_xlim(x - max_range, x + max_range)
		ax.set_ylim(y - max_range, y + max_range)
		ax.set_zlim(z - max_range, z + max_range)
		az = 360.0 * (i / max(1, num_views))
		ax.view_init(elev=20.0, azim=az)
		buf = BytesIO()
		plt.tight_layout(pad=0)
		plt.savefig(buf, format='png', dpi=100, transparent=True)
		plt.close(fig)
		buf.seek(0)
		views.append(Image.open(buf).convert('RGB'))
		buf.close()
	return views


def _render_views_blender(
	blender_path: str,
	model_path: str,
	image_size: int = 224,
	num_views: int = 4,
	background_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> List[Image.Image]:
	import subprocess
	import tempfile
	import textwrap

	if not os.path.isfile(blender_path):
		raise FileNotFoundError(f"Blender not found at: {blender_path}")
	if not os.path.isfile(model_path):
		raise FileNotFoundError(f"3D model not found: {model_path}")

	# Temporary output directory for rendered PNGs
	with tempfile.TemporaryDirectory() as tmpdir:
		# Python script executed inside Blender
		script = textwrap.dedent(f"""
		import bpy, math, mathutils
		import os
		bpy.ops.wm.read_factory_settings(use_empty=True)
		# Scene setup
		scene = bpy.context.scene
		scene.render.engine = 'CYCLES'
		try:
			bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
			bpy.context.scene.cycles.device = 'GPU'
		except Exception:
			pass
		scene.render.resolution_x = {image_size}
		scene.render.resolution_y = {image_size}
		scene.render.film_transparent = True
		# Remove default objects
		for obj in list(bpy.data.objects):
			bpy.data.objects.remove(obj, do_unlink=True)
		# Import model
		path = r"{model_path}"
		lower = path.lower()
		if lower.endswith('.obj'):
			bpy.ops.import_scene.obj(filepath=path)
		elif lower.endswith('.ply'):
			bpy.ops.import_mesh.ply(filepath=path)
		elif lower.endswith('.stl'):
			bpy.ops.import_mesh.stl(filepath=path)
		elif lower.endswith('.glb') or lower.endswith('.gltf'):
			bpy.ops.import_scene.gltf(filepath=path)
		else:
			raise RuntimeError('Unsupported format for Blender import')
		mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
		if not mesh_objs:
			raise RuntimeError('No mesh objects after import')
		# Join all meshes for simplicity
		bpy.context.view_layer.objects.active = mesh_objs[0]
		for o in mesh_objs:
			o.select_set(True)
		bpy.ops.object.join()
		obj = bpy.context.active_object
		# Center and scale to unit size
		bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
		bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
		minx,miny,minz = [min(v.co[i] for v in obj.data.vertices) for i in range(3)]
		maxx,maxy,maxz = [max(v.co[i] for v in obj.data.vertices) for i in range(3)]
		cx,cy,cz = (minx+maxx)/2.0,(miny+maxy)/2.0,(minz+maxz)/2.0
		obj.location = obj.location - mathutils.Vector((cx,cy,cz))
		diag = math.sqrt((maxx-minx)**2 + (maxy-miny)**2 + (maxz-minz)**2)
		if diag > 0:
			s = 1.0/diag
			obj.scale = (s,s,s)
		# Camera
		cam_data = bpy.data.cameras.new('cam')
		cam = bpy.data.objects.new('cam', cam_data)
		bpy.context.collection.objects.link(cam)
		scene.camera = cam
		# Light
		light_data = bpy.data.lights.new(name="light", type='SUN')
		light_data.energy = 3.0
		light = bpy.data.objects.new(name="light", object_data=light_data)
		bpy.context.collection.objects.link(light)
		# Background
		b = bpy.data.worlds.new('world')
		scene.world = b
		scene.world.color = ({background_rgba[0]}, {background_rgba[1]}, {background_rgba[2]},)
		# Render multiple views
		import math
		for i in range(max(1, {num_views})):
			az = 2.0*math.pi*(i/max(1,{num_views}))
			elev = math.radians(20.0)
			dist = 2.2
			x = dist*math.cos(elev)*math.cos(az)
			y = dist*math.sin(elev)
			z = dist*math.cos(elev)*math.sin(az)
			cam.location = (x,y,z)
			cam.rotation_euler = (0,0,0)
			# Look at origin
			dir = mathutils.Vector((0,0,0)) - cam.location
			cam.rotation_euler = dir.to_track_quat('-Z','Y').to_euler()
			# Light follows camera
			light.location = cam.location
			# Output
			scene.render.filepath = os.path.join(r"{tmpdir}", f"view_{'{'}i{'}'}.png")
			bpy.ops.render.render(write_still=True)
		""")
		# Run Blender
		cmd = [blender_path, "-b", "--python-expr", script]
		proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		if proc.returncode != 0:
			raise RuntimeError(f"Blender render failed: {proc.stderr[:500]}\n{proc.stdout[:500]}")
		# Load images back
		views: List[Image.Image] = []
		for i in range(max(1, num_views)):
			img_path = os.path.join(tmpdir, f"view_{i}.png")
			if os.path.isfile(img_path):
				views.append(Image.open(img_path).convert('RGB'))
		return views



