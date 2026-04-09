from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


def preprocess_image(image: Image.Image, image_size: int = 96) -> torch.Tensor:
    """
    Convert an input image into the same tensor format used during training.

    Steps:
    - convert to grayscale
    - resize to 96x96
    - normalize to [0, 1]
    - add batch and channel dimensions

    Returns:
        Tensor of shape [1, 1, H, W]
    """
    img = np.array(image.convert("L"), dtype=np.float32)

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # The Colab deployment tests used already-normalized slice-like images.
    # This keeps inference simple and robust for demo purposes.
    img = img / 255.0

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    return tensor


def mask_from_prediction(prediction: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Convert model logits to a binary uint8 mask.
    """
    pred_mask = (torch.sigmoid(prediction)[0, 0].cpu().numpy() > threshold).astype(np.uint8)
    return pred_mask


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, image_size: int = 96) -> Image.Image:
    """
    Create a red overlay of the predicted ROI on top of the input image.
    """
    base = np.array(image.convert("L"), dtype=np.uint8)
    base = cv2.resize(base, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    base_rgb = np.stack([base, base, base], axis=-1)
    overlay = base_rgb.copy()

    overlay[mask == 1] = [255, 0, 0]

    blended = cv2.addWeighted(base_rgb, 0.7, overlay, 0.3, 0)
    return Image.fromarray(blended)
