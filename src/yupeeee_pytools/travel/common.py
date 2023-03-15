from typing import Tuple

import torch
from torch.nn import Softmax


__all__ = [
    "model_output",
]


def model_output(
        model,
        data: torch.Tensor,
        use_cuda: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if use_cuda else "cpu")

    out = model(data.to(device)).detach().cpu()

    conf = Softmax(dim=-1)(out)
    pred = torch.argmax(conf, dim=-1)

    return conf, pred
