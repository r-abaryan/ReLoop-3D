import os
import shutil
import argparse
from typing import List, Tuple, Optional
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


def load_latest_checkpoint_if_any(out_dir: str, num_classes: int, device: torch.device) -> Tuple[nn.Module, str, int]:
	"""Load latest checkpoint if available. Returns (model, model_name, start_epoch)."""
	latest = os.path.join(out_dir, "model_latest.pt")
	model_name = "cnn"
	state_dict = None
	start_epoch = 0
	
	if os.path.isfile(latest):
		try:
			ckpt = torch.load(latest, map_location=device)
			model_name = ckpt.get("model_name", "cnn")
			state_dict = ckpt.get("model_state", {})
			start_epoch = ckpt.get("epoch", 0)
			print(f"Loaded checkpoint from epoch {start_epoch}, model: {model_name}")
		except Exception as e:
			print(f"Warning: Failed to load checkpoint: {e}. Starting fresh.")

	if model_name == "cnn":
		model = SimpleShapeCNN(num_classes=num_classes)
	else:
		model = create_vit_tiny_classifier(num_classes=num_classes)

	if state_dict is not None:
		try:
			model.load_state_dict(state_dict, strict=False)
			print("Loaded model weights from checkpoint.")
		except Exception as e:
			print(f"Warning: Failed to load state dict: {e}. Using random initialization.")
	
	return model, model_name, start_epoch


def move_corrected_images(data_dir: str, corrections_csv: str) -> int:
	"""Move corrected images to proper directories. Returns count of moved files."""
	if not os.path.isfile(corrections_csv):
		print(f"Warning: Corrections file not found: {corrections_csv}")
		return 0
	
	try:
		corr_df = pd.read_csv(corrections_csv)
	except Exception as e:
		print(f"Warning: Failed to read corrections CSV: {e}")
		return 0
	
	moved_count = 0
	for _, row in corr_df.iterrows():
		try:
			rel_path = str(row["image_path"]).replace("\\", "/")
			label = str(row["label"]).strip()
			if not label:
				continue
			
			# Try relative path first
			src = os.path.join(data_dir, rel_path)
			if not os.path.isfile(src):
				# Try absolute path
				src = rel_path if os.path.isabs(rel_path) and os.path.isfile(rel_path) else None
				if src is None:
					continue
			
			dst_dir = os.path.join(data_dir, "images", label)
			os.makedirs(dst_dir, exist_ok=True)
			dst = os.path.join(dst_dir, os.path.basename(src))
			
			# Only move if source and destination differ
			if os.path.abspath(src) != os.path.abspath(dst):
				if os.path.exists(dst):
					# Avoid overwriting - add suffix
					base, ext = os.path.splitext(dst)
					dst = f"{base}_corrected{ext}"
				shutil.move(src, dst)
				moved_count += 1
		except Exception as e:
			print(f"Warning: Failed to move {row.get('image_path', 'unknown')}: {e}")
			continue
	
	print(f"Moved {moved_count} corrected images.")
	return moved_count


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
	avg_loss = sum(losses) / max(1, len(losses))
	return float(avg_loss), acc


def save_checkpoint(
	model: nn.Module, 
	classes: List[str], 
	out_dir: str, 
	epoch: int, 
	model_name: str,
	is_best: bool = False
) -> None:
	"""Save checkpoint with epoch tracking."""
	os.makedirs(out_dir, exist_ok=True)
	
	# Save epoch checkpoint
	ckpt_path = os.path.join(out_dir, f"model_epoch_{epoch}.pt")
	checkpoint = {
		"model_state": model.state_dict(),
		"classes": classes,
		"epoch": epoch,
		"model_name": model_name
	}
	torch.save(checkpoint, ckpt_path)
	
	# Update latest
	latest = os.path.join(out_dir, "model_latest.pt")
	try:
		if os.path.exists(latest):
			os.remove(latest)
		os.link(ckpt_path, latest)
	except Exception:
		# Fallback on platforms that don't support hard links
		torch.save(checkpoint, latest)
	
	# Save best model
	if is_best:
		best_path = os.path.join(out_dir, "model_best.pt")
		torch.save(checkpoint, best_path)
		print(f"Saved best model to {best_path}")


def main():
	parser = argparse.ArgumentParser(description="Retrain model with corrections")
	parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
	parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for checkpoints")
	parser.add_argument("--corrections", type=str, default="annotations/corrections.csv", help="Corrections CSV file")
	parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
	parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
	parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
	parser.add_argument("--image_size", type=int, default=128, help="Image size")
	parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
	parser.add_argument("--lr_scheduler", action="store_true", help="Use learning rate scheduler")
	parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
	args = parser.parse_args()

	# Setup directories
	images_dir, _, _, out_dir = ensure_dirs(args.data_dir, args.out_dir)
	corr_path = resolve_path(os.getcwd(), args.corrections)
	
	if not os.path.isfile(corr_path):
		raise FileNotFoundError(f"Corrections file not found: {corr_path}")

	# Move corrected images
	move_corrected_images(args.data_dir, corr_path)

	# Merge corrections into labels
	try:
		labels_df, classes = merge_corrections_into_labels(args.data_dir, corr_path)
		print(f"Found {len(labels_df)} labeled images across {len(classes)} classes: {classes}")
	except Exception as e:
		raise RuntimeError(f"Failed to merge corrections: {e}")

	if len(classes) == 0:
		raise ValueError("No classes found. Check your labels.csv and corrections.csv.")

	# Load model from checkpoint if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, model_name, start_epoch = load_latest_checkpoint_if_any(out_dir, num_classes=len(classes), device=device)
	model.to(device)

	# Auto-adjust image size for ViT
	if model_name == "vit_tiny" and args.image_size == 128:
		args.image_size = 224
		print(f"Auto-adjusted image size to {args.image_size} for ViT")

	# Build datasets
	train_tf, val_tf = build_transforms(args.image_size)
	full_ds = ShapesImageDataset(args.data_dir, labels_df, classes, transform=train_tf)
	
	if len(full_ds) == 0:
		raise ValueError("Dataset is empty. Check your data directory and labels.")
	
	val_size = int(len(full_ds) * args.val_split)
	train_size = len(full_ds) - val_size
	train_ds, val_ds = random_split(full_ds, [train_size, val_size])
	val_ds.dataset.transform = val_tf

	print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

	# Setup optimizer and scheduler
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = None
	if args.lr_scheduler:
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode='min', factor=0.5, patience=2, min_lr=args.min_lr, verbose=True
		)
	
	criterion = nn.CrossEntropyLoss()

	# Training loop with early stopping
	best_val_acc = 0.0
	best_val_loss = float('inf')
	patience_counter = 0
	metrics_history = []

	for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
		# Training phase
		model.train()
		train_losses = []
		pbar = tqdm(train_loader, desc=f"Retrain Epoch {epoch}/{start_epoch + args.epochs}")
		
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
		
		avg_train_loss = sum(train_losses) / len(train_losses)

		# Validation phase
		val_loss, val_acc = evaluate(model, val_loader, device)
		
		# Learning rate scheduling
		if scheduler:
			scheduler.step(val_loss)
		
		# Check for improvement
		is_best = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
		if is_best:
			best_val_acc = val_acc
			best_val_loss = val_loss
			patience_counter = 0
		else:
			patience_counter += 1

		# Logging
		print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f} ({'BEST' if is_best else ''})")
		
		# Save checkpoint
		save_checkpoint(model, classes, out_dir, epoch, model_name, is_best=is_best)
		
		# Track metrics
		metrics_history.append({
			"epoch": epoch,
			"train_loss": avg_train_loss,
			"val_loss": val_loss,
			"val_acc": val_acc
		})

		# Early stopping
		if patience_counter >= args.patience:
			print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
			break

	# Save metrics
	metrics_path = os.path.join(out_dir, "retrain_metrics.csv")
	metrics_df = pd.DataFrame(metrics_history)
	if os.path.exists(metrics_path):
		metrics_df.to_csv(metrics_path, mode="a", header=False, index=False)
	else:
		metrics_df.to_csv(metrics_path, index=False)
	
	print(f"\nTraining complete! Best val_acc: {best_val_acc:.4f}, Best val_loss: {best_val_loss:.4f}")
	print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
	main()
