from typing import Sequence

import torch
import torch.nn as nn
import torchvision.transforms as tf
from torchvision.transforms import Normalize

from .config import datasets, means, stds


__all__ = [
    "normalize",
    "denormalize",
]


class DeNormalize(nn.Module):
    def __init__(
            self,
            mean: Sequence,
            std: Sequence,
    ) -> None:
        super(DeNormalize, self).__init__()

        self.mean = mean
        self.std = std

        dim = len(mean)

        self.denormalize = tf.Compose([
            tf.Normalize(mean=[0.] * dim, std=[1. / v for v in std]),
            tf.Normalize(mean=[-v for v in mean], std=[1.] * dim),
        ])

    def forward(
            self,
            tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.denormalize(tensor)

    def __repr__(
            self,
    ) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


normalize = dict()
denormalize = dict()

for dataset in datasets:
    normalize[dataset] = Normalize(
        mean=means[dataset],
        std=stds[dataset],
    )

    denormalize[dataset] = DeNormalize(
        mean=means[dataset],
        std=stds[dataset],
    )
