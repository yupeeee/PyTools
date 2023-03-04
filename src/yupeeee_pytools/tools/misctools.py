from typing import Tuple

import matplotlib.pyplot as plt
import torch


__all__ = [
    "best_figsize_for_subplots",
    "plot_multiple_images",

    "repeat_tensor",
]


def best_figsize_for_subplots(
        num_figs: int,
        desired_hw_ratio: Tuple[int, int] = (1, 1),
) -> Tuple[int, int]:
    from .listools import argmin_list

    # initialize
    grow = 1

    while True:
        h, w = desired_hw_ratio
        h, w = h * grow, w * grow
        size = h * w

        if size < num_figs:
            grow += 1
            continue

        break

    # optimize
    best = (h, w)
    err = size - num_figs

    while err > 0:
        candidates = [best]
        errs = [err]

        h_reduced_err = (h - 1) * w - num_figs
        w_reduced_err = h * (w - 1) - num_figs

        if h_reduced_err >= 0:
            candidates.append((h - 1, w))
            errs.append(h_reduced_err)

        if w_reduced_err >= 0:
            candidates.append((h, w - 1))
            errs.append(w_reduced_err)

        min_idx = argmin_list(errs)

        best = candidates[min_idx]
        err = errs[min_idx]

        if best == best:
            break

    return best


def plot_multiple_images(
        images,
        desired_hw_ratio: Tuple[int, int] = (1, 1),
        dpi: int = 300,
        pad_inches: float = 0.,
        save_to: str = None,
) -> None:
    dim = len(images.shape)
    assert dim in [3, 4]

    if dim == 3:
        num_figs, img_h, img_w = images.shape
    else:
        num_figs, img_h, img_w, c = images.shape

    h, w = best_figsize_for_subplots(num_figs, desired_hw_ratio)

    fig, axes = plt.subplots(figsize=(w*img_w/dpi, h*img_h/dpi), nrows=h, ncols=w, dpi=dpi)
    axes = axes.ravel()

    for i in range(num_figs):
        axes[i].imshow(images[i])

    [ax.set_axis_off() for ax in axes]

    if save_to is not None:
        plt.savefig(save_to, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
        plt.draw()
        plt.close("all")
    else:
        plt.show()


def repeat_tensor(
        tensor: torch.Tensor,
        repeat: int,
        dim: int = 0,
) -> torch.Tensor:
    return torch.repeat_interleave(tensor[None, ...], repeat, dim=dim)
