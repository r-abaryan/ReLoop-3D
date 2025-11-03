import os
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from .models.simple_cnn import SimpleShapeCNN
from .models.transformer import create_vit_tiny_classifier


def load_latest_checkpoint(out_dir: str, device: torch.device):
	latest = os.path.join(out_dir, "model_latest.pt")
	if not os.path.isfile(latest):
		raise FileNotFoundError("No latest checkpoint found. Train first.")
	return torch.load(latest, map_location=device)


def build_val_transform(image_size: int = 128):
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


def annotate_image(img_path: str, text: str, save_path: str):
	img = Image.open(img_path).convert("RGB")
	draw = ImageDraw.Draw(img)
	# Try to use a default font
	try:
		font = ImageFont.truetype("arial.ttf", 18)
	except Exception:
		font = ImageFont.load_default()
	draw.rectangle([(0, 0), (img.width, 24)], fill=(0, 0, 0, 128))
	draw.text((5, 4), text, fill=(255, 255, 255), font=font)
	img.save(save_path)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--image", type=str, required=True)
	parser.add_argument("--out_dir", type=str, default="outputs")
	parser.add_argument("--image_size", type=int, default=128)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ckpt = load_latest_checkpoint(args.out_dir, device)
	classes: List[str] = ckpt["classes"]
	model_name = ckpt.get("model_name", "cnn")

	if model_name != "cnn" and args.image_size == 128:
		args.image_size = 224
	val_tf = build_val_transform(args.image_size)

	if model_name == "cnn":
		model = SimpleShapeCNN(num_classes=len(classes))
	else:
		model = create_vit_tiny_classifier(num_classes=len(classes))
	model.load_state_dict(ckpt["model_state"], strict=False)
	model.to(device)
	model.eval()

	img = Image.open(args.image).convert("RGB")
	ten = val_tf(img).unsqueeze(0).to(device)
	with torch.no_grad():
		logits = model(ten)
		probs = F.softmax(logits, dim=1)[0]
		top_prob, top_idx = torch.topk(probs, k=min(3, len(classes)))

	labels = [classes[i] for i in top_idx.cpu().numpy().tolist()]
	scores = top_prob.cpu().numpy().tolist()
	pairs = list(zip(labels, [float(s) for s in scores]))

	print("Top predictions:")
	for lbl, sc in pairs:
		print(f"- {lbl}: {sc:.4f}")

	os.makedirs(args.out_dir, exist_ok=True)
	annot_path = os.path.join(args.out_dir, "prediction_annotated.png")
	annot_text = ", ".join([f"{l}:{s:.2f}" for l, s in pairs])
	annotate_image(args.image, annot_text, annot_path)
	print(f"Saved annotated image: {annot_path}")


if __name__ == "__main__":
	main()


