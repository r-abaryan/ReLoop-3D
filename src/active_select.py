import os
import csv
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

# Local imports work both as module and script
try:
	from .three_d import render_views  # type: ignore
	from .models.simple_cnn import SimpleShapeCNN  # type: ignore
	from .models.transformer import create_vit_tiny_classifier  # type: ignore
except Exception:
	from three_d import render_views
	from models.simple_cnn import SimpleShapeCNN
	from models.transformer import create_vit_tiny_classifier


def load_model(checkpoint_dir: str, device: torch.device):
	ckpt_path = os.path.join(checkpoint_dir, "model_latest.pt")
	if not os.path.isfile(ckpt_path):
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
	ckpt = torch.load(ckpt_path, map_location=device)
	classes: List[str] = ckpt["classes"]
	model_name = ckpt.get("model_name", "cnn")
	if model_name == "cnn":
		model = SimpleShapeCNN(num_classes=len(classes))
		image_size = 128
	else:
		model = create_vit_tiny_classifier(num_classes=len(classes))
		image_size = 224
	model.load_state_dict(ckpt["model_state"], strict=False)
	model.to(device).eval()
	return model, classes, image_size


def build_tf(image_size: int):
	from torchvision import transforms  # local import
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


def score_mesh(model_path: str, model: torch.nn.Module, device: torch.device, image_size: int, num_views: int) -> Tuple[float, int, List[Image.Image]]:
	views = render_views(model_path, image_size=image_size, num_views=num_views)
	transform = build_tf(image_size)
	probs_list: List[torch.Tensor] = []
	with torch.no_grad():
		for img in views:
			ten = transform(img.convert("RGB")).unsqueeze(0).to(device)
			logits = model(ten)
			probs = F.softmax(logits, dim=1)[0]
			probs_list.append(probs)
	if not probs_list:
		return 1.0, -1, views
	avg = torch.stack(probs_list, dim=0).mean(dim=0)
	top_prob, top_idx = torch.topk(avg, k=1)
	uncertainty = float(1.0 - top_prob.item())
	pred_idx = int(top_idx.item())
	return uncertainty, pred_idx, views


def main():
	parser = argparse.ArgumentParser(description="Select uncertain unlabeled meshes for labeling queue")
	parser.add_argument("--mesh_dir", required=True, help="Directory with unlabeled meshes")
	parser.add_argument("--out_csv", default="annotations/active_queue.csv", help="Output CSV path")
	parser.add_argument("--checkpoint", default="outputs", help="Checkpoint directory")
	parser.add_argument("--num_views", type=int, default=4)
	parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of meshes")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, classes, image_size = load_model(args.checkpoint, device)

	os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
	mesh_paths = [os.path.join(args.mesh_dir, f) for f in os.listdir(args.mesh_dir)
				  if f.lower().endswith((".obj", ".ply", ".stl", ".glb", ".gltf"))]
	mesh_paths.sort()
	if args.limit > 0:
		mesh_paths = mesh_paths[:args.limit]

	rows = [("model_path", "pred_label", "uncertainty", "label")]  # label is to be filled by user
	for mp in mesh_paths:
		try:
			unc, pred_idx, _ = score_mesh(mp, model, device, image_size, args.num_views)
			pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else ""
			rows.append((mp, pred_label, f"{unc:.6f}", ""))
		except Exception as exc:
			rows.append((mp, "", "nan", ""))

	with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerows(rows)
	print(f"Wrote {len(rows)-1} entries to {args.out_csv}")


if __name__ == "__main__":
	main()
