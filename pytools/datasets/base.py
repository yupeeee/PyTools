from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch


__all__ = [
    "ClassificationDataset",
]


class ClassificationDataset:
    def __init__(
            self,
            name: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.transform = transform
        self.target_transform = target_transform

        self.data = None        # << must be initialized
        self.targets = None     # << must be initialized

    def __getitem__(
            self,
            index: int,
    ) -> Tuple[Any, Any]:
        if isinstance(self.data, list):     # ImageNet
            path, target = self.data[index]
            data = Image.open(path).convert('RGB')
        else:
            data = self.data[index]
            target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(
            self,
    ) -> int:
        return len(self.data)

    def initialize(
            self,
            dataset,
    ) -> None:
        self.data = dataset.data
        self.targets = dataset.targets

    def mean_and_std_of_data(
            self,
    ) -> Tuple[Any, Any]:
        assert self.data is not None
        assert not isinstance(self.data, list)

        data = self.data

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        data = data.type(torch.float32)

        mean = data.mean(axis=(0, 1, 2))
        std = data.std(axis=(0, 1, 2))

        return mean, std

    def data_and_targets_of_class_c(
            self,
            c: int,
    ) -> Tuple[Any, Any]:
        assert self.data is not None and self.targets is not None

        indices = self.targets == c

        data_c = self.data[indices]
        targets_c = self.targets[indices]

        if self.transform:
            data_c = self.transform(data_c)

        if self.target_transform:
            targets_c = self.target_transform(targets_c)

        return data_c, targets_c
