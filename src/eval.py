import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .utils.data import load_or_build_labels_csv
from .datasets.shape_dataset import ShapesImageDataset
from .models.simple_cnn import SimpleShapeCNN
from .models.transformer import create_vit_tiny_classifier


def build_val_transform(image_size: int = 128):
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


def load_latest_checkpoint(out_dir: str, device: torch.device):
	latest = os.path.join(out_dir, "model_latest.pt")
	if not os.path.isfile(latest):
		raise FileNotFoundError("No latest checkpoint found. Train first.")
	return torch.load(latest, map_location=device)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--out_dir", type=str, default="outputs")
	parser.add_argument("--val_split", type=float, default=0.2)
	parser.add_argument("--image_size", type=int, default=128)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ckpt = load_latest_checkpoint(args.out_dir, device)
	classes: List[str] = ckpt["classes"]
	model_name = ckpt.get("model_name", "cnn")

	# Adjust default image size for ViT
	if model_name != "cnn" and ("image_size" not in vars(args) or args.image_size == 128):
		args.image_size = 224

	val_tf = build_val_transform(args.image_size)
	labels_df, classes_from_data = load_or_build_labels_csv(args.data_dir)
	# Keep original classes order from checkpoint
	full_ds = ShapesImageDataset(args.data_dir, labels_df, classes, transform=val_tf)
	val_size = int(len(full_ds) * args.val_split)
	train_size = len(full_ds) - val_size
	_, val_ds = random_split(full_ds, [train_size, val_size])

	val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

	if model_name == "cnn":
		model = SimpleShapeCNN(num_classes=len(classes))
	else:
		model = create_vit_tiny_classifier(num_classes=len(classes))
	model.load_state_dict(ckpt["model_state"], strict=False)
	model.to(device)
	model.eval()

	criterion = nn.CrossEntropyLoss()
	all_preds = []
	all_labels = []
	losses = []
	with torch.no_grad():
		for imgs, labels in val_loader:
			imgs = imgs.to(device)
			labels = labels.to(device)
			logits = model(imgs)
			loss = criterion(logits, labels)
			losses.append(loss.item())
			preds = logits.argmax(dim=1)
			all_preds.extend(preds.cpu().numpy().tolist())
			all_labels.extend(labels.cpu().numpy().tolist())

	all_preds = np.array(all_preds)
	all_labels = np.array(all_labels)
	acc = float((all_preds == all_labels).mean()) if len(all_labels) else 0.0
	cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))

	# Save metrics
	os.makedirs(args.out_dir, exist_ok=True)
	metrics_csv = os.path.join(args.out_dir, "eval_metrics.csv")
	pd.DataFrame([{ "val_acc": acc, "val_loss": float(np.mean(losses) if losses else 0.0) }]).to_csv(metrics_csv, index=False)

	# Save confusion matrix plot
	fig, ax = plt.subplots(figsize=(6, 6))
	disp = ConfusionMatrixDisplay(cm, display_labels=classes)
	disp.plot(cmap="Blues", ax=ax, colorbar=False)
	plt.title("Confusion Matrix (Val)")
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	fig_path = os.path.join(args.out_dir, "confusion_matrix.png")
	plt.savefig(fig_path, dpi=150)
	plt.close(fig)

	print(f"val_acc={acc:.4f}\nmetrics: {metrics_csv}\nfigure: {fig_path}")


if __name__ == "__main__":
	main()


