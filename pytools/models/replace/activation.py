import torch.nn as nn


__all__ = [
    "GELU_to_ReLU",
    "ReLU_to_GELU",
]


def GELU_to_ReLU(
        module: nn.Module,
) -> nn.Module:
    return nn.ReLU(
        inplace=True,
    )


def ReLU_to_GELU(
        module: nn.Module,
) -> nn.Module:
    return nn.GELU(
        approximate='none',
    )
