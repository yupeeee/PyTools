from typing import List

import torch.nn as nn


__all__ = [
    "get_params",
    "exponential_string_to_float",
]


def get_params(
        module: nn.Module,
) -> List[str]:
    attrs = ")".join(
        "(".join(
            str(module).replace(" ", "")
            .split("(")[1:])
        .split(")")[:-1]).split(",")

    params = []
    i = 0

    while i < len(attrs):
        if "(" in attrs[i] and ")" in attrs[i + 1]:
            params.append(f"{attrs[i]}, {attrs[i + 1]}")
            i += 1
        else:
            params.append(attrs[i])

        i += 1

    return params


def exponential_string_to_float(
        exponential: str,
) -> float:
    num, exponent = exponential.split("e")

    return float(f"{num}E{exponent}")
