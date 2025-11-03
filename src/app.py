import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gradio as gr

from .models.simple_cnn import SimpleShapeCNN
from .models.transformer import create_vit_tiny_classifier
from .three_d import render_views


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


def predict_model_3d(file_obj, out_dir: str, num_views: int = 4):
	if file_obj is None:
		return "No model", [], None
	if _model is None:
		_load_latest_checkpoint(out_dir)
	# Render views
	try:
		views = render_views(file_obj.name, image_size=_image_size, num_views=num_views)
	except Exception as exc:
		return f"Render error: {exc}", [], None
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
		return "No views rendered", [], None
	avg = _aggregate_probs(probs_list)
	top_prob, top_idx = torch.topk(avg, k=min(5, len(_classes)))
	labels = [_classes[i] for i in top_idx.cpu().numpy().tolist()]
	scores = [float(x) for x in top_prob.cpu().numpy().tolist()]
	best = f"{labels[0]} ({scores[0]:.3f})" if labels else "N/A"
	rows = [[l, f"{s:.3f}"] for l, s in zip(labels, scores)]
	return best, rows, thumbs


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
				model_file = gr.File(label="Mesh (.obj/.ply/.stl)")
				num_views = gr.Slider(1, 12, value=4, step=1, label="Num Views")
				btn_3d = gr.Button("Render + Predict")
				pred_3d = gr.Textbox(label="Top Prediction")
				table_3d = gr.Dataframe(headers=["label", "probability"], label="Top-K", interactive=False)
				gallery = gr.Gallery(label="Rendered Views", columns=4, height=200)
				btn_3d.click(fn=predict_model_3d, inputs=[model_file, out_dir, num_views], outputs=[pred_3d, table_3d, gallery])
	return demo


if __name__ == "__main__":
	demo = build_interface()
	demo.launch()


