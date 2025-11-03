import os
import glob
from typing import Dict, List, Tuple
import pandas as pd


def load_or_build_labels_csv(data_dir: str) -> Tuple[pd.DataFrame, List[str]]:
	labels_csv_path = os.path.join(data_dir, "labels.csv")
	if os.path.isfile(labels_csv_path):
		df = pd.read_csv(labels_csv_path)
		assert {"image_path", "label"}.issubset(df.columns), "labels.csv must have columns: image_path,label"
		classes = sorted(df["label"].unique().tolist())
		return df, classes

	# Build from folder structure data/images/<class>/*
	images_root = os.path.join(data_dir, "images")
	rows: List[Dict[str, str]] = []
	classes_set = set()
	for class_dir in sorted(glob.glob(os.path.join(images_root, "*"))):
		if not os.path.isdir(class_dir):
			continue
		class_name = os.path.basename(class_dir)
		for img_path in glob.glob(os.path.join(class_dir, "**", "*.*"), recursive=True):
			if os.path.splitext(img_path)[1].lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
				continue
			rel_path = os.path.relpath(img_path, data_dir).replace("\\", "/")
			rows.append({"image_path": rel_path, "label": class_name})
			classes_set.add(class_name)

	df = pd.DataFrame(rows)
	if not df.empty:
		df.to_csv(labels_csv_path, index=False)
		classes = sorted(classes_set)
		return df, classes

	raise FileNotFoundError(
		"No labels.csv found and no classed folders under data/images/. Provide at least one."
	)


def merge_corrections_into_labels(
	data_dir: str, corrections_csv: str
) -> Tuple[pd.DataFrame, List[str]]:
	labels_path = os.path.join(data_dir, "labels.csv")
	labels_df, classes = load_or_build_labels_csv(data_dir)
	corr_df = pd.read_csv(corrections_csv)
	assert {"image_path", "label"}.issubset(corr_df.columns)

	# Normalize paths to be relative to data_dir
	corr_df = corr_df.copy()
	corr_df["image_path"] = corr_df["image_path"].astype(str).str.replace("\\", "/")

	# Upsert rows
	key_to_idx = {p: i for i, p in enumerate(labels_df["image_path"].astype(str))}
	for _, row in corr_df.iterrows():
		img_path = row["image_path"]
		label = row["label"]
		if img_path in key_to_idx:
			labels_df.loc[key_to_idx[img_path], "label"] = label
		else:
			labels_df.loc[len(labels_df)] = {"image_path": img_path, "label": label}

	labels_df.to_csv(labels_path, index=False)
	classes = sorted(labels_df["label"].unique().tolist())
	return labels_df, classes


