import torch


__all__ = [
    "angle_of_three_points",
]


def angle_of_three_points(
        start: torch.Tensor,
        mid: torch.Tensor,
        end: torch.Tensor,
        eps: float = 1e-6,
):
    v1 = start - mid
    v2 = end - mid

    v1 = v1 / torch.norm(v1)
    v2 = v2 / torch.norm(v2)

    angle = torch.dot(v1, v2)

    angle = angle.clip(-1, 1)

    if 1 - eps < angle < 1:
        angle = torch.Tensor([0])
    elif -1 < angle < -1 + eps:
        angle = torch.Tensor([torch.pi])
    else:
        angle = torch.acos(angle)

    return angle
