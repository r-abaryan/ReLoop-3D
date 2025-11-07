import os
import sys
import textwrap
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gradio as gr

from models.simple_cnn import SimpleShapeCNN
from models.transformer import create_vit_tiny_classifier
from three_d import render_views
from depth_to_mesh import image_to_mesh
from multiview_to_mesh import multiview_to_mesh

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_classes: List[str] = []
_image_size: int = 128


def _load_latest_checkpoint(out_dir: str):
	global _model, _classes, _image_size
	latest = os.path.join(out_dir, "model_latest.pt")
	if not os.path.isfile(latest):
		raise FileNotFoundError("No latest checkpoint found. Run training first.")
	ckpt = torch.load(latest, map_location=_device)
	_classes = ckpt["classes"]
	model_name = ckpt.get("model_name", "cnn")
	if model_name == "cnn":
		_model = SimpleShapeCNN(num_classes=len(_classes))
		_image_size = 128
	else:
		_model = create_vit_tiny_classifier(num_classes=len(_classes))
		_image_size = 224
	_model.load_state_dict(ckpt["model_state"], strict=False)
	_model.to(_device)
	_model.eval()


def _build_tf():
	return transforms.Compose([
		transforms.Resize((_image_size, _image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


def predict_image(img: Image.Image, out_dir: str) -> Tuple[str, list]:
	if img is None:
		return "No image", []
	if _model is None:
		_load_latest_checkpoint(out_dir)
	transform = _build_tf()
	ten = transform(img.convert("RGB")).unsqueeze(0).to(_device)
	with torch.no_grad():
		logits = _model(ten)
		probs = F.softmax(logits, dim=1)[0]
	top_prob, top_idx = torch.topk(probs, k=min(5, len(_classes)))
	labels = [_classes[i] for i in top_idx.cpu().numpy().tolist()]
	scores = [float(x) for x in top_prob.cpu().numpy().tolist()]
	best = f"{labels[0]} ({scores[0]:.3f})" if labels else "N/A"
	rows = [[l, f"{s:.3f}"] for l, s in zip(labels, scores)]
	return best, rows


def _aggregate_probs(probs_list: List[torch.Tensor]) -> torch.Tensor:
	# Mean of probabilities across views
	stack = torch.stack(probs_list, dim=0)
	return stack.mean(dim=0)


def _prepare_preview_path(model_path: str) -> str:
	"""Copy model to a local cache path with a safe filename for Model3D preview."""
	try:
		import shutil
		from pathlib import Path
		if not model_path:
			return model_path
		src = Path(model_path)
		if not src.exists():
			return model_path
		cache_dir = Path(".preview_cache")
		cache_dir.mkdir(parents=True, exist_ok=True)
		dst = cache_dir / src.name
		# Overwrite if different file or missing
		try:
			shutil.copyfile(str(src), str(dst))
		except Exception:
			return model_path
		return str(dst)
	except Exception:
		return model_path


def _convert_to_glb_for_preview(model_path: str) -> str:
	"""Convert arbitrary mesh to GLB for reliable web preview; fallback to cached original."""
	try:
		import trimesh  # type: ignore
		from pathlib import Path
		if not model_path:
			return model_path
		src = Path(model_path)
		if not src.exists():
			return model_path
		cache_dir = Path(".preview_cache")
		cache_dir.mkdir(parents=True, exist_ok=True)
		out_path = cache_dir / (src.stem + ".glb")
		loaded = trimesh.load(str(src), force='scene')
		if isinstance(loaded, trimesh.Scene):
			scene = loaded
		else:
			scene = trimesh.Scene(loaded)  # wrap single mesh
		# Try export to GLB; if dependency missing, this may raise
		scene.export(str(out_path))
		if out_path.exists() and out_path.stat().st_size > 0:
			return str(out_path)
		return _prepare_preview_path(model_path)
	except Exception:
		# Any failure: use cached original
		return _prepare_preview_path(model_path)


def _clear_caches():
	"""Remove preview/render caches to free space and force regeneration."""
	import shutil
	removed = []
	for d in [".preview_cache", ".render_cache"]:
		if os.path.isdir(d):
			try:
				shutil.rmtree(d)
				removed.append(d)
			except Exception:
				pass
	msg = "Cleared: " + (", ".join(removed) if removed else "(nothing to clear)")
	return gr.update(value=msg)


def _gallery_update(images: List[Image.Image], num_views: int) -> gr.update:
	# Fit exactly one row: height ~= image_size + padding; cap min height
	height = max(200, _image_size + 40)
	cols = max(1, min(int(num_views), 6))
	return gr.update(value=images, height=height, columns=cols)


def predict_model_3d(model_path: str, out_dir: str, num_views: int = 4, blender_path: str = ""):
	if not model_path:
		return "No model", [], None, gr.update(value=None)
	if _model is None:
		_load_latest_checkpoint(out_dir)
	# Render views
	try:
		views = render_views(model_path, image_size=_image_size, num_views=num_views, blender_path=(blender_path or None))
	except Exception as exc:
		return f"Render error: {exc}", [], _gallery_update([], num_views), gr.update(value=_convert_to_glb_for_preview(model_path))
	# Predict per view
	transform = _build_tf()
	probs_list: List[torch.Tensor] = []
	thumbs: List[Image.Image] = []
	with torch.no_grad():
		for img in views:
			thumbs.append(img)
			ten = transform(img.convert("RGB")).unsqueeze(0).to(_device)
			logits = _model(ten)
			probs = F.softmax(logits, dim=1)[0]
			probs_list.append(probs)
	if not probs_list:
		return "No views rendered", [], _gallery_update([], num_views), gr.update(value=_convert_to_glb_for_preview(model_path))
	avg = _aggregate_probs(probs_list)
	top_prob, top_idx = torch.topk(avg, k=min(5, len(_classes)))
	labels = [_classes[i] for i in top_idx.cpu().numpy().tolist()]
	scores = [float(x) for x in top_prob.cpu().numpy().tolist()]
	best = f"{labels[0]} ({scores[0]:.3f})" if labels else "N/A"
	rows = [[l, f"{s:.3f}"] for l, s in zip(labels, scores)]
	return best, rows, _gallery_update(thumbs, num_views), gr.update(value=_convert_to_glb_for_preview(model_path))


def _model_file_path(model_path: str):
	return gr.update(value=_convert_to_glb_for_preview(model_path) if model_path else None)

# -------- Active Learning (UI) --------

def _acq_scores_from_probs(probs: torch.Tensor, method: str = "entropy", k: int = 2) -> float:
	# probs: (C,) on CPU
	method = (method or "entropy").lower()
	p = probs.clamp_min(1e-8)
	if method == "entropy":
		return float(-(p * p.log()).sum().item())
	# top-k margin: difference between top1 and top2 (smaller => more uncertain)
	values, _ = torch.topk(p, k=min(max(k, 2), p.numel()))
	margin = float(values[0].item() - values[1].item())
	# Convert to uncertainty score where higher = more uncertain
	return float(1.0 - margin)


def _list_meshes(mesh_dir: str) -> List[str]:
	if not mesh_dir or not os.path.isdir(mesh_dir):
		return []
	names = []
	for f in os.listdir(mesh_dir):
		fl = f.lower()
		if fl.endswith((".obj", ".ply", ".stl", ".glb", ".gltf")):
			names.append(os.path.join(mesh_dir, f))
	names.sort()
	return names


def al_score_unlabeled(mesh_dir: str, out_dir: str, num_views: int, acq: str, k: int):
	if _model is None:
		_load_latest_checkpoint(out_dir)
	paths = _list_meshes(mesh_dir)
	if not paths:
		return []
	transform = _build_tf()
	rows = []
	with torch.no_grad():
		for mp in paths:
			try:
				views = render_views(mp, image_size=_image_size, num_views=num_views)
				probs_list: List[torch.Tensor] = []
				for img in views:
					ten = transform(img.convert("RGB")).unsqueeze(0).to(_device)
					logits = _model(ten)
					probs = F.softmax(logits, dim=1)[0].cpu()
					probs_list.append(probs)
				if not probs_list:
					continue
				avg = torch.stack(probs_list, dim=0).mean(dim=0)
				score = _acq_scores_from_probs(avg, method=acq, k=k)
				best_p, best_idx = torch.topk(avg, k=1)
				pred = _classes[int(best_idx.item())] if _classes else ""
				rows.append([mp, pred, float(score), ""])  # label to be filled
			except Exception:
				rows.append([mp, "", float("nan"), ""])  # keep placeholder
	# Sort by uncertainty descending
	rows.sort(key=lambda r: (-(r[2] if isinstance(r[2], float) else 0.0)))
	return rows


def al_apply_labels(table_rows, num_views: int, out_images: str):
	# Accept pandas.DataFrame or list
	rows_list: List[List[str]] = []
	try:
		import pandas as pd  # type: ignore
		if hasattr(table_rows, "values"):
			rows_list = table_rows.values.tolist()  # type: ignore[attr-defined]
		elif isinstance(table_rows, list):
			rows_list = table_rows
		else:
			rows_list = []
	except Exception:
		rows_list = table_rows if isinstance(table_rows, list) else []

	os.makedirs(out_images, exist_ok=True)
	os.makedirs("annotations", exist_ok=True)
	csv_path = os.path.join("annotations", "labeled.csv")
	# Load existing labeled pairs to avoid duplicates
	existing = set()
	if os.path.isfile(csv_path):
		try:
			import csv as _csv
			with open(csv_path, "r", encoding="utf-8") as f:
				reader = _csv.reader(f)
				existing = set(tuple(r) for r in reader if len(r) >= 2)
		except Exception:
			existing = set()

	applied = 0
	new_rows: List[Tuple[str, str]] = []
	for r in rows_list:
		if not r or len(r) < 4:
			continue
		mp = str(r[0]).strip()
		label = str(r[3]).strip()
		if not mp or not label:
			continue
		label_sanitized = "".join(c for c in label if c.isalnum() or c in ("_", "-")) or "unknown"
		label_dir = os.path.join(out_images, label_sanitized)
		os.makedirs(label_dir, exist_ok=True)
		try:
			views = render_views(mp, image_size=_image_size, num_views=num_views)
			stem = os.path.splitext(os.path.basename(mp))[0]
			for i, img in enumerate(views):
				img.convert("RGB").save(os.path.join(label_dir, f"{stem}_v{i}.png"))
			applied += 1
			pair = (mp, label_sanitized)
			if pair not in existing:
				new_rows.append(pair)
		except Exception:
			pass
	# Append new rows
	if new_rows:
		try:
			import csv as _csv
			write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
			with open(csv_path, "a", newline="", encoding="utf-8") as f:
				writer = _csv.writer(f)
				if write_header:
					writer.writerow(["model_path", "label"])
				for mp, lab in new_rows:
					writer.writerow([mp, lab])
		except Exception:
			pass
	return f"Applied labels to {applied} meshes into {out_images}. Logged {len(new_rows)} entries at {csv_path}"


def img_to_3d_handler(mode: str, single_img, multi_imgs, model_type: str, poisson_depth: int):
	"""Image(s) → Mesh. Mode: 'Single' or 'Multi-view'."""
	try:
		os.makedirs("generated_meshes", exist_ok=True)
		out_path = os.path.join("generated_meshes", "mesh.obj")
		if mode == "Single":
			if single_img is None:
				return None, "No image provided."
			image_to_mesh(single_img, out_path, model_type=model_type, poisson_depth=poisson_depth)
			return out_path, f"Single-image mesh saved: {out_path}"
		else:  # Multi-view
			if not multi_imgs or len(multi_imgs) < 3:
				return None, "Upload at least 3 images for multi-view reconstruction."
			images = [Image.open(f.name) if hasattr(f, 'name') else f for f in multi_imgs]
			multiview_to_mesh(images, out_path, poisson_depth=poisson_depth)
			return out_path, f"Multi-view mesh saved: {out_path} (COLMAP)"
	except Exception as exc:
		return None, f"Error: {exc}"


def build_interface():
	with gr.Blocks(title="3D Active Learning Playground") as demo:
		gr.Markdown("## 3D Active Learning Playground\nUpload an image or a simple 3D mesh; we'll render multiple views and classify.")
		out_dir = gr.Textbox(value="outputs", label="Checkpoint Directory")
		with gr.Tabs():
			with gr.TabItem("Image"):
				img = gr.Image(type="pil", label="Input Image")
				btn_img = gr.Button("Predict Image")
				pred_i = gr.Textbox(label="Top Prediction")
				table_i = gr.Dataframe(headers=["label", "probability"], label="Top-K", interactive=False)
				btn_img.click(fn=predict_image, inputs=[img, out_dir], outputs=[pred_i, table_i])
			with gr.TabItem("Image → 3D"):
				gr.Markdown("Convert image(s) to 3D mesh. Single: depth-based (fast). Multi-view: COLMAP reconstruction (better quality, requires COLMAP installed).")
				mode = gr.Radio(choices=["Single", "Multi-view"], value="Single", label="Mode")
				with gr.Row():
					single_img = gr.Image(type="pil", label="Single Image", visible=True)
					multi_imgs = gr.File(label="Multiple Images (3+)", file_count="multiple", visible=False)
				model_type = gr.Dropdown(choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"], value="DPT_Large", label="Depth Model (Single mode only)")
				poisson_depth = gr.Slider(6, 12, value=9, step=1, label="Poisson Depth")
				btn_convert = gr.Button("Generate Mesh")
				mesh_out = gr.File(label="Download Mesh")
				status_3d = gr.Textbox(label="Status")
				# Toggle visibility based on mode
				def toggle_inputs(m):
					return gr.update(visible=(m=="Single")), gr.update(visible=(m=="Multi-view"))
				mode.change(fn=toggle_inputs, inputs=[mode], outputs=[single_img, multi_imgs])
				btn_convert.click(fn=img_to_3d_handler, inputs=[mode, single_img, multi_imgs, model_type, poisson_depth], outputs=[mesh_out, status_3d])
			with gr.TabItem("3D Model"):
				with gr.Row():
					with gr.Column(scale=1):
						model_file = gr.File(label="Mesh (.obj/.ply/.stl/.glb/.gltf)", file_types=[".obj", ".ply", ".stl", ".glb", ".gltf"], type="filepath")
						with gr.Accordion("Advanced", open=False):
							blender = gr.Textbox(value="", label="Blender Path (optional)")
							num_views = gr.Slider(1, 12, value=4, step=1, label="Num Views")
						btn_3d = gr.Button("Render + Predict")
						cache_status = gr.Markdown(visible=True)
						btn_clear = gr.Button("Clear Cache")
					with gr.Column(scale=1):
						model_view = gr.Model3D(label="3D Preview", height=420)
				# Outputs under controls
				pred_3d = gr.Textbox(label="Top Prediction")
				table_3d = gr.Dataframe(headers=["label", "probability"], label="Top-K", interactive=False)
				gallery = gr.Gallery(label="Rendered Views", columns=4, height=300)
				# Wiring
				model_file.change(fn=_model_file_path, inputs=[model_file], outputs=[model_view])
				btn_3d.click(fn=predict_model_3d, inputs=[model_file, out_dir, num_views, blender], outputs=[pred_3d, table_3d, gallery, model_view])
				btn_clear.click(fn=_clear_caches, inputs=[], outputs=[cache_status])
			with gr.TabItem("Active Learning"):
				with gr.Row():
					with gr.Column(scale=1):
						mesh_dir = gr.Textbox(value="unlabeled_meshes", label="Unlabeled Mesh Directory")
						acq = gr.Dropdown(choices=["entropy", "margin"], value="entropy", label="Acquisition Function")
						k = gr.Number(value=2, precision=0, label="Top-k (for margin)")
						num_views2 = gr.Slider(1, 12, value=4, step=1, label="Num Views")
						btn_score = gr.Button("Score Unlabeled")
						images_out = gr.Textbox(value="data/images", label="Images Out (apply labels)")
						btn_apply = gr.Button("Apply Labels")
					with gr.Column(scale=1):
						queue = gr.Dataframe(headers=["model_path", "pred_label", "uncertainty", "label"], label="Labeling Queue", wrap=True)
						status = gr.Markdown(visible=True)
				# Wire scoring and apply
				btn_score.click(fn=al_score_unlabeled, inputs=[mesh_dir, out_dir, num_views2, acq, k], outputs=[queue])
				btn_apply.click(fn=al_apply_labels, inputs=[queue, num_views2, images_out], outputs=[status])
	return demo


if __name__ == "__main__":
	demo = build_interface()
	demo.launch()


