import os
from typing import Tuple


def ensure_dirs(base_data_dir: str, out_dir: str) -> Tuple[str, str, str, str]:

	images_dir = os.path.join(base_data_dir, "images")
	unlabeled_dir = os.path.join(base_data_dir, "unlabeled")
	annotations_dir = os.path.join(base_data_dir, "..", "annotations")
	annotations_dir = os.path.normpath(annotations_dir)

	os.makedirs(images_dir, exist_ok=True)
	os.makedirs(unlabeled_dir, exist_ok=True)
	os.makedirs(annotations_dir, exist_ok=True)
	os.makedirs(out_dir, exist_ok=True)

	return images_dir, unlabeled_dir, annotations_dir, out_dir


def resolve_path(root: str, path: str) -> str:
	if os.path.isabs(path):
		return path
	return os.path.normpath(os.path.join(root, path))


