import os
import shutil
import argparse
from typing import List
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from .utils.paths import ensure_dirs, resolve_path
from .utils.data import merge_corrections_into_labels
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


def load_latest_checkpoint_if_any(out_dir: str, num_classes: int, device: torch.device):
	latest = os.path.join(out_dir, "model_latest.pt")
	model_name = "cnn"
	state_dict = None
	if os.path.isfile(latest):
		ckpt = torch.load(latest, map_location=device)
		model_name = ckpt.get("model_name", "cnn")
		state_dict = ckpt.get("model_state", {})

	if model_name == "cnn":
		model = SimpleShapeCNN(num_classes=num_classes)
	else:
		model = create_vit_tiny_classifier(num_classes=num_classes)

	if state_dict is not None:
		model.load_state_dict(state_dict, strict=False)
	return model


def move_corrected_images(data_dir: str, corrections_csv: str):
	corr_df = pd.read_csv(corrections_csv)
	for _, row in corr_df.iterrows():
		rel_path = str(row["image_path"]).replace("\\", "/")
		label = str(row["label"])  # e.g., cube/sphere/cylinder
		src = os.path.join(data_dir, rel_path)
		if not os.path.isfile(src):
			# Might be provided as absolute path; if so, just copy into images/<label>/
			src = rel_path if os.path.isabs(rel_path) else None
			if src is None or not os.path.isfile(src):
				continue
		dst_dir = os.path.join(data_dir, "images", label)
		os.makedirs(dst_dir, exist_ok=True)
		dst = os.path.join(dst_dir, os.path.basename(src))
		if os.path.abspath(src) != os.path.abspath(dst):
			shutil.move(src, dst)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
	model.eval()
	correct = 0
	total = 0
	losses = []
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
	return float(sum(losses) / max(1, len(losses))), acc


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--out_dir", type=str, default="outputs")
	parser.add_argument("--corrections", type=str, default="annotations/corrections.csv")
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=5e-4)
	parser.add_argument("--val_split", type=float, default=0.2)
	parser.add_argument("--image_size", type=int, default=128)
	args = parser.parse_args()

	images_dir, _, _, out_dir = ensure_dirs(args.data_dir, args.out_dir)
	corr_path = resolve_path(os.getcwd(), args.corrections)
	if not os.path.isfile(corr_path):
		raise FileNotFoundError(f"Corrections file not found: {corr_path}")

	# Move corrected images into data/images/<label>/
	move_corrected_images(args.data_dir, corr_path)

	# Merge corrections into labels.csv
	labels_df, classes = merge_corrections_into_labels(args.data_dir, corr_path)

	# Train again using latest checkpoint as init if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = load_latest_checkpoint_if_any(out_dir, num_classes=len(classes), device=device)
	model.to(device)

	train_tf, val_tf = build_transforms(args.image_size)
	full_ds = ShapesImageDataset(args.data_dir, labels_df, classes, transform=train_tf)
	val_size = int(len(full_ds) * args.val_split)
	train_size = len(full_ds) - val_size
	train_ds, val_ds = random_split(full_ds, [train_size, val_size])
	val_ds.dataset.transform = val_tf

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(1, args.epochs + 1):
		model.train()
		pbar = tqdm(train_loader, desc=f"Retrain {epoch}/{args.epochs}")
		for imgs, labels in pbar:
			imgs = imgs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			logits = model(imgs)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()
			pbar.set_postfix({"loss": f"{loss.item():.4f}"})

		# Simple evaluation
		model.eval()
		val_loss, val_acc = 0.0, 0.0
		with torch.no_grad():
			val_loss, val_acc = evaluate(model, val_loader, device)
		print(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

	# Save latest
	latest = os.path.join(out_dir, "model_latest.pt")
	torch.save({"model_state": model.state_dict(), "classes": classes}, latest)


if __name__ == "__main__":
	main()


