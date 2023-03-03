import torch
import torch.nn as nn


__all__ = [
    "normalize_direction",
    "fgsm_direction",
    "random_direction",
]


def normalize_direction(
        direction: torch.Tensor,
        method: str = "dim",
) -> torch.Tensor:
    assert method in ["dim", "unit"]

    shape = direction.shape

    normalized = direction.reshape(-1)

    if method == "dim":
        normalized = normalized / torch.norm(normalized) * len(normalized) ** 0.5

    elif method == "unit":
        normalized = normalized / torch.norm(normalized)

    else:
        raise ValueError

    return normalized.reshape(shape)


def fgsm_direction(
        data: torch.Tensor,
        targets: torch.Tensor,
        model,
        normalize: str = None,
        seed: int = None,
        use_cuda: bool = False,
) -> torch.Tensor:
    from ..tools import set_random_seed

    assert model is not None

    if seed is not None:
        set_random_seed(seed)

    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    data, targets = data.detach().to(device), targets.detach().to(device)
    data.requires_grad = True

    out = model(data)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(out, targets)
    loss.backward()

    with torch.no_grad():
        data_grad = data.grad.data

    direction = data_grad.sign().detach().cpu()

    if normalize is not None:
        direction = normalize_direction(direction, normalize)

    return direction


def random_direction(
        data: torch.Tensor,
        normalize: str = None,
        seed: int = None,
) -> torch.Tensor:
    from ..tools import set_random_seed

    if seed is not None:
        set_random_seed(seed)

    direction = torch.randn_like(data)

    if normalize is not None:
        direction = normalize_direction(direction, normalize)

    return direction
