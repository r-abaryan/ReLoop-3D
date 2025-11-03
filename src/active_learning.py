import os
import argparse
import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from .utils.paths import ensure_dirs
from .models.simple_cnn import SimpleShapeCNN
from .models.transformer import create_vit_tiny_classifier


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
	probs = torch.softmax(logits, dim=1)
	log_probs = torch.log(probs.clamp(min=1e-9))
	return -(probs * log_probs).sum(dim=1)


def load_latest_checkpoint(out_dir: str, device: torch.device):
	latest = os.path.join(out_dir, "model_latest.pt")
	if not os.path.isfile(latest):
		raise FileNotFoundError("No latest checkpoint found. Train first.")
	ckpt = torch.load(latest, map_location=device)
	return ckpt


def build_val_transform(image_size: int = 128):
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--out_dir", type=str, default="outputs")
	parser.add_argument("--num_query", type=int, default=50)
	parser.add_argument("--image_size", type=int, default=128)
	args = parser.parse_args()

	_, unlabeled_dir, annotations_dir, out_dir = ensure_dirs(args.data_dir, args.out_dir)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ckpt = load_latest_checkpoint(out_dir, device)
	classes: List[str] = ckpt["classes"]
	model_name = ckpt.get("model_name", "cnn")
	if model_name == "cnn":
		model = SimpleShapeCNN(num_classes=len(classes))
	else:
		model = create_vit_tiny_classifier(num_classes=len(classes))
	model.load_state_dict(ckpt["model_state"], strict=False) 
	model.to(device)
	model.eval()

	val_tf = build_val_transform(args.image_size)
	image_paths = []
	for p in glob.glob(os.path.join(unlabeled_dir, "**", "*.*"), recursive=True):
		if os.path.splitext(p)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
			image_paths.append(p)

	if len(image_paths) == 0:
		print("No unlabeled images found under data/unlabeled/.")
		return

	records: List[Tuple[str, float, str]] = []
	with torch.no_grad():
		for path in tqdm(image_paths, desc="Scoring unlabeled"):
			img = Image.open(path).convert("RGB")
			t = val_tf(img).unsqueeze(0).to(device)
			logits = model(t)
			H = entropy_from_logits(logits)[0].item()
			pred_idx = int(torch.argmax(logits, dim=1).item())
			pred_label = classes[pred_idx]
			rel_path = os.path.relpath(path, args.data_dir).replace("\\", "/")
			records.append((rel_path, H, pred_label))

	records.sort(key=lambda x: x[1], reverse=True)
	selected = records[: args.num_query]

	queue_path = os.path.join(os.path.dirname(args.data_dir), "annotations", "annotation_queue.csv")
	os.makedirs(os.path.dirname(queue_path), exist_ok=True)
	df = pd.DataFrame(selected, columns=["image_path", "uncertainty", "predicted_label"])
	df["label"] = ""  # fill this manually
	df.to_csv(queue_path, index=False)
	print(f"Wrote annotation queue with {len(selected)} rows to: {queue_path}")


if __name__ == "__main__":
	main()


