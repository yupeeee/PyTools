from typing import Tuple

import matplotlib.pyplot as plt


__all__ = [
    "best_figsize_for_subplots",
    "plot_multiple_images",
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
) -> None:
    num_figs = len(images)
    h, w = best_figsize_for_subplots(num_figs, desired_hw_ratio)

    fig, axes = plt.subplots(nrows=h, ncols=w)
    axes = axes.ravel()

    for i in range(num_figs):
        axes[i].imshow(images[i])

    [ax.set_axis_off() for ax in axes]

    plt.tight_layout()
    plt.show()
