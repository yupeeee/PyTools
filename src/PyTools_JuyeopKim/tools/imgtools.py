import cv2
import numpy as np
import torch


__all__ = [
    "grayscale_to_colormap",
]


def grayscale_to_colormap(
        grayscale_images: np.ndarray,
        colormap: str = "jet",
) -> np.ndarray:
    def apply_colormap(grayscale_image, colormap):
        converted = cv2.applyColorMap(
            grayscale_image,
            getattr(cv2, f"COLORMAP_{colormap.upper()}")
        )

        return cv2.cvtColor(converted, cv2.COLOR_BGR2RGB)

    if isinstance(grayscale_images, torch.Tensor):
        grayscale_images = np.array(grayscale_images)

    assert len(grayscale_images.shape) in [2, 3]

    if len(grayscale_images) == 2:
        converted = apply_colormap(grayscale_images, colormap)

    else:
        converted = [
            np.expand_dims(apply_colormap(img, colormap), axis=0)
            for img in grayscale_images
        ]

        converted = np.concatenate(converted, axis=0)

    return converted
