from typing import Any

import torch.nn as nn

from .config import *


__all__ = [
    "Preprocess",
]


class Preprocess(nn.Module):
    def __init__(
            self,
            model_name: str,
            specify_weights: str = default_weight_specification,
    ) -> None:
        from .base import load_pytorch_model_weights

        assert model_name in imagenet_models

        super(Preprocess, self).__init__()

        if model_name in list_of_pytorch_imagenet_models:
            weights = load_pytorch_model_weights(model_name, specify_weights)
            self.preprocess = weights.transforms()

        elif model_name in list_of_not_in_pytorch_imagenet_models:
            self.preprocess = default_imagenet_preprocess

    def forward(
            self,
            data: Any,
    ) -> Any:
        return self.preprocess(data)
