from typing import Optional
import torch.nn as nn


def create_vit_tiny_classifier(num_classes: int, model_name: str = "vit_tiny_patch16_224") -> nn.Module:
	try:
		import timm
	except ImportError as exc:
		raise RuntimeError("timm is required for transformer models. Install with pip install timm") from exc

	model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
	return model


