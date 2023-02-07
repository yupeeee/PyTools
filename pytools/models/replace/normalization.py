"""
Error occurs when replaced: TBD
"""
import math
import torch.nn as nn


__all__ = [
    "BatchNorm2d_to_LayerNorm",
    "LayerNorm_to_BatchNorm2d",
]


def BatchNorm2d_to_LayerNorm(
        module: nn.Module,
) -> nn.Module:
    from .common import get_params, exponential_string_to_float

    num_features, eps, _, affine, _ = get_params(module)

    normalized_shape = (int(num_features),)

    eps = exponential_string_to_float(eps.split("=")[-1])

    elementwise_affine = bool(affine.split("=")[-1])

    return nn.LayerNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
    )


def LayerNorm_to_BatchNorm2d(
        module: nn.Module
) -> nn.Module:
    from .common import get_params, exponential_string_to_float

    normalized_shape, eps, elementwise_affine = get_params(module)

    num_features = normalized_shape.split("(")[-1].split(")")[0].split(",")
    num_features = math.prod([int(v) for v in num_features if v != " "])

    eps = exponential_string_to_float(eps.split("=")[-1])

    affine = bool(elementwise_affine.split("=")[-1])

    return nn.BatchNorm2d(
        num_features=num_features,
        eps=eps,
        affine=affine,
    )
