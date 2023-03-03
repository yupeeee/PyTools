import torch.nn as nn


__all__ = [
    "build_criterion",
]

criterions = [
    "CrossEntropyLoss",
]


def build_criterion(config):
    """
    in config.yaml:
    ⋮
    CRITERION:
      TYPE: str
    ⋮
    """
    criterion_type = config.CRITERION.TYPE

    assert criterion_type in criterions

    if criterion_type == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()

    else:
        raise ValueError

    return criterion
