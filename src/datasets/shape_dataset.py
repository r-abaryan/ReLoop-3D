import os
from typing import Callable, List, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ShapesImageDataset(Dataset):
	def __init__(
		self,
		data_dir: str,
		labels_df: pd.DataFrame,
		classes: List[str],
		transform: Optional[Callable] = None,
	):
		self.data_dir = data_dir
		self.labels_df = labels_df.reset_index(drop=True)
		self.classes = classes
		self.class_to_idx = {c: i for i, c in enumerate(classes)}
		self.transform = transform

	def __len__(self) -> int:
		return len(self.labels_df)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		row = self.labels_df.iloc[idx]
		img_rel = str(row["image_path"]).replace("\\", "/")
		img_path = os.path.join(self.data_dir, img_rel)
		img = Image.open(img_path).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)
		label = self.class_to_idx[str(row["label"])]
		return img, label


