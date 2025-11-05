import os
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from .utils.paths import ensure_dirs
from .utils.data import load_or_build_labels_csv
from .datasets.shape_dataset import ShapesImageDataset
from .models.simple_cnn import SimpleShapeCNN
from .models.transformer import create_vit_tiny_classifier


def build_transforms(image_size: int = 128):
	train_tf = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	val_tf = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	return train_tf, val_tf


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
	model.eval()
	correct = 0
	total = 0
	losses: List[float] = []
	crit = nn.CrossEntropyLoss()
	with torch.no_grad():
		for imgs, labels in loader:
			imgs = imgs.to(device)
			labels = labels.to(device)
			logits = model(imgs)
			loss = crit(logits, labels)
			losses.append(loss.item())
			preds = logits.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
	acc = correct / max(1, total)
	return float(np.mean(losses) if losses else 0.0), acc


def save_checkpoint(model: nn.Module, classes: List[str], out_dir: str, epoch: int, model_name: str):
	os.makedirs(out_dir, exist_ok=True)
	ckpt_path = os.path.join(out_dir, f"model_epoch_{epoch}.pt")
	torch.save({"model_state": model.state_dict(), "classes": classes, "epoch": epoch, "model_name": model_name}, ckpt_path)
	latest = os.path.join(out_dir, "model_latest.pt")
	try:
		if os.path.exists(latest):
			os.remove(latest)
		os.link(ckpt_path, latest)
	except Exception:
		# Fallback on platforms that don't support hard links
		torch.save({"model_state": model.state_dict(), "classes": classes, "epoch": epoch, "model_name": model_name}, latest)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--out_dir", type=str, default="outputs")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--val_split", type=float, default=0.2)
	parser.add_argument("--image_size", type=int, default=128)
	parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "vit_tiny"]) 
	parser.add_argument("--allow_unlabeled", action="store_true", help="If no labels are found, create a dummy single-class dataset from any images under data_dir for smoke-testing.")
	parser.add_argument("--unlabeled_class_name", type=str, default="object", help="Class name to assign when --allow_unlabeled is used.")
	args = parser.parse_args()

	_, _, _, out_dir = ensure_dirs(args.data_dir, args.out_dir)
	try:
		labels_df, classes = load_or_build_labels_csv(args.data_dir)
	except FileNotFoundError:
		if not args.allow_unlabeled:
			raise
		# Build a dummy single-class labels.csv by scanning for images under data_dir and data_dir/unlabeled
		import glob
		from pathlib import Path
		exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
		candidates = []
		for root in [args.data_dir, os.path.join(args.data_dir, "unlabeled")]:
			for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
				if os.path.splitext(p)[1].lower() in exts and os.path.isfile(p):
					candidates.append(p)
		if not candidates:
			raise FileNotFoundError("No images found to create a dummy dataset. Place some images under data_dir or data_dir/unlabeled or provide labels.")
		rows = []
		for p in candidates:
			rel = os.path.relpath(p, args.data_dir).replace("\\", "/")
			rows.append({"image_path": rel, "label": args.unlabeled_class_name})
		labels_df = pd.DataFrame(rows)
		labels_df.to_csv(os.path.join(args.data_dir, "labels.csv"), index=False)
		classes = [args.unlabeled_class_name]
	# If using ViT, default to 224 unless overridden
	if args.model == "vit_tiny" and ("image_size" not in vars(args) or args.image_size == 128):
		args.image_size = 224
	train_tf, val_tf = build_transforms(args.image_size)

	full_ds = ShapesImageDataset(args.data_dir, labels_df, classes, transform=train_tf)
	val_size = int(len(full_ds) * args.val_split)
	train_size = len(full_ds) - val_size
	train_ds, val_ds = random_split(full_ds, [train_size, val_size])
	# Ensure val set uses val transforms
	val_ds.dataset.transform = val_tf

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if args.model == "cnn":
		model = SimpleShapeCNN(num_classes=len(classes)).to(device)
	else:
		model = create_vit_tiny_classifier(num_classes=len(classes)).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(1, args.epochs + 1):
		model.train()
		pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
		for imgs, labels in pbar:
			imgs = imgs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			logits = model(imgs)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()
			pbar.set_postfix({"loss": f"{loss.item():.4f}"})

		val_loss, val_acc = evaluate(model, val_loader, device)
		print(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
		save_checkpoint(model, classes, out_dir, epoch, model_name=args.model)

	# Save final metrics
	metrics_path = os.path.join(out_dir, "metrics.csv")
	row = pd.DataFrame([{"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}])
	if os.path.exists(metrics_path):
		row.to_csv(metrics_path, mode="a", header=False, index=False)
	else:
		row.to_csv(metrics_path, index=False)


if __name__ == "__main__":
	main()


