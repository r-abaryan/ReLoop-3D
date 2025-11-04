import os
import csv
import argparse
from typing import List

from PIL import Image

try:
	from .three_d import render_views  # type: ignore
except Exception:
	from three_d import render_views


def read_queue(csv_path: str) -> List[dict]:
	rows: List[dict] = []
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			rows.append(r)
	return rows


def sanitize_label(label: str) -> str:
	return "".join(c for c in label.strip() if c.isalnum() or c in ("_", "-")) or "unknown"


def main():
	parser = argparse.ArgumentParser(description="Apply labels: render labeled meshes into dataset folders for training")
	parser.add_argument("--queue", default="annotations/active_queue.csv", help="CSV with columns: model_path,label")
	parser.add_argument("--out_images", default="data/images", help="Root folder for labeled images")
	parser.add_argument("--image_size", type=int, default=224)
	parser.add_argument("--num_views", type=int, default=4)
	args = parser.parse_args()

	rows = read_queue(args.queue)
	os.makedirs(args.out_images, exist_ok=True)
	count = 0
	for r in rows:
		label = (r.get("label") or "").strip()
		model_path = (r.get("model_path") or "").strip()
		if not label or not model_path:
			continue
		label_dir = os.path.join(args.out_images, sanitize_label(label))
		os.makedirs(label_dir, exist_ok=True)
		try:
			views: List[Image.Image] = render_views(model_path, image_size=args.image_size, num_views=args.num_views)
			stem = os.path.splitext(os.path.basename(model_path))[0]
			for i, img in enumerate(views):
				out_path = os.path.join(label_dir, f"{stem}_v{i}.png")
				img.convert("RGB").save(out_path)
			count += 1
		except Exception:
			pass
	print(f"Rendered {count} labeled meshes into {args.out_images}")


if __name__ == "__main__":
	main()
