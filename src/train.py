import os
import argparse
from typing import List, Tuple, Optional
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


def build_transforms(image_size: int = 128) -> Tuple[transforms.Compose, transforms.Compose]:
	"""Build training and validation transforms."""
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
	"""Evaluate model on validation set. Returns (loss, accuracy)."""
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


def save_checkpoint(
	model: nn.Module, 
	classes: List[str], 
	out_dir: str, 
	epoch: int, 
	model_name: str,
	is_best: bool = False
) -> None:
	"""Save checkpoint with latest/best tracking."""
	os.makedirs(out_dir, exist_ok=True)
	
	checkpoint = {
		"model_state": model.state_dict(),
		"classes": classes,
		"epoch": epoch,
		"model_name": model_name
	}
	
	# Save epoch checkpoint
	ckpt_path = os.path.join(out_dir, f"model_epoch_{epoch}.pt")
	torch.save(checkpoint, ckpt_path)
	
	# Update latest
	latest = os.path.join(out_dir, "model_latest.pt")
	try:
		if os.path.exists(latest):
			os.remove(latest)
		os.link(ckpt_path, latest)
	except Exception:
		# Fallback on platforms that don't support hard links or if link fails
		torch.save(checkpoint, latest)
		
	# Save best
	if is_best:
		best_path = os.path.join(out_dir, "model_best.pt")
		torch.save(checkpoint, best_path)


def load_latest_checkpoint(out_dir: str, device: torch.device) -> Optional[dict]:
	"""Load latest checkpoint if available."""
	latest = os.path.join(out_dir, "model_latest.pt")
	if os.path.isfile(latest):
		try:
			return torch.load(latest, map_location=device)
		except Exception as e:
			print(f"Warning: Failed to load latest checkpoint: {e}")
	return None


def main():
	parser = argparse.ArgumentParser(description="Train 3D shape classifier")
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--out_dir", type=str, default="outputs")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--val_split", type=float, default=0.2)
	parser.add_argument("--image_size", type=int, default=128)
	parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "vit_tiny"]) 
	parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")
	parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
	parser.add_argument("--lr_scheduler", action="store_true", help="Use ReduceLROnPlateau scheduler")
	parser.add_argument("--allow_unlabeled", action="store_true", help="Create dummy dataset if no labels found.")
	parser.add_argument("--unlabeled_class_name", type=str, default="object")
	args = parser.parse_args()

	_, _, _, out_dir = ensure_dirs(args.data_dir, args.out_dir)
	
	try:
		labels_df, classes = load_or_build_labels_csv(args.data_dir)
	except FileNotFoundError:
		if not args.allow_unlabeled:
			raise
		import glob
		exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
		candidates = []
		for root in [args.data_dir, os.path.join(args.data_dir, "unlabeled")]:
			for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
				if os.path.splitext(p)[1].lower() in exts and os.path.isfile(p):
					candidates.append(p)
		if not candidates:
			raise FileNotFoundError("No images found for dummy dataset.")
		rows = [{"image_path": os.path.relpath(p, args.data_dir).replace("\\", "/"), "label": args.unlabeled_class_name} for p in candidates]
		labels_df = pd.DataFrame(rows)
		labels_df.to_csv(os.path.join(args.data_dir, "labels.csv"), index=False)
		classes = [args.unlabeled_class_name]

	# Auto-adjust for ViT
	if args.model == "vit_tiny" and args.image_size == 128:
		args.image_size = 224

	train_tf, val_tf = build_transforms(args.image_size)
	full_ds = ShapesImageDataset(args.data_dir, labels_df, classes, transform=train_tf)
	val_size = int(len(full_ds) * args.val_split)
	train_size = len(full_ds) - val_size
	train_ds, val_ds = random_split(full_ds, [train_size, val_size])
	val_ds.dataset.transform = val_tf

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Load or Init Model
	start_epoch = 1
	checkpoint = None
	if args.resume:
		checkpoint = load_latest_checkpoint(out_dir, device)
		
	if checkpoint:
		print(f"Resuming from epoch {checkpoint['epoch']}...")
		model_name = checkpoint.get("model_name", args.model)
		if model_name == "cnn":
			model = SimpleShapeCNN(num_classes=len(classes)).to(device)
		else:
			model = create_vit_tiny_classifier(num_classes=len(classes)).to(device)
		model.load_state_dict(checkpoint["model_state"])
		start_epoch = checkpoint["epoch"] + 1
	else:
		if args.model == "cnn":
			model = SimpleShapeCNN(num_classes=len(classes)).to(device)
		else:
			model = create_vit_tiny_classifier(num_classes=len(classes)).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True) if args.lr_scheduler else None
	criterion = nn.CrossEntropyLoss()

	best_val_loss = float('inf')
	patience_counter = 0
	history = []

	for epoch in range(start_epoch, args.epochs + 1):
		model.train()
		train_losses = []
		pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
		for imgs, labels in pbar:
			imgs = imgs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			logits = model(imgs)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())
			pbar.set_postfix({"loss": f"{loss.item():.4f}"})

		val_loss, val_acc = evaluate(model, val_loader, device)
		avg_train_loss = np.mean(train_losses)
		print(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
		
		if scheduler:
			scheduler.step(val_loss)

		# Check for improvement
		is_best = val_loss < best_val_loss
		if is_best:
			best_val_loss = val_loss
			patience_counter = 0
		else:
			patience_counter += 1

		save_checkpoint(model, classes, out_dir, epoch, model_name=args.model, is_best=is_best)
		
		history.append({
			"epoch": epoch, 
			"train_loss": avg_train_loss, 
			"val_loss": val_loss, 
			"val_acc": val_acc
		})
		
		# Early stopping
		if patience_counter >= args.patience:
			print(f"Early stopping at epoch {epoch}")
			break

	# Save history
	history_path = os.path.join(out_dir, "training_history.csv")
	pd.DataFrame(history).to_csv(history_path, index=False)
	print(f"Training history saved to {history_path}")


if __name__ == "__main__":
	main()
