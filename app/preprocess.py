from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


def preprocess_image(image: Image.Image, image_size: int = 96) -> torch.Tensor:
    img = np.array(image.convert("L"), dtype=np.float32)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    return tensor


def postprocess_prediction(pred: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    return (torch.sigmoid(pred)[0, 0].cpu().numpy() > threshold).astype(np.uint8)
