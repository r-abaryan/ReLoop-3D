import os
import sys
import textwrap
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gradio as gr

# Allow running as a module (python -m src.app) or as a script (python src/app.py)
try:
	from .models.simple_cnn import SimpleShapeCNN  # type: ignore
	from .models.transformer import create_vit_tiny_classifier  # type: ignore
	from .three_d import render_views  # type: ignore
except Exception:
	from models.simple_cnn import SimpleShapeCNN
	from models.transformer import create_vit_tiny_classifier
	from three_d import render_views

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
				gallery = gr.Gallery(label="Rendered Views", columns=4, height=256)
				# Wiring
				model_file.change(fn=_model_file_path, inputs=[model_file], outputs=[model_view])
				btn_3d.click(fn=predict_model_3d, inputs=[model_file, out_dir, num_views, blender], outputs=[pred_3d, table_3d, gallery, model_view])
				btn_clear.click(fn=_clear_caches, inputs=[], outputs=[cache_status])
	return demo


if __name__ == "__main__":
	demo = build_interface()
	demo.launch()


