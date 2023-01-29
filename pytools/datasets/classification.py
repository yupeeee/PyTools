from typing import Callable, Optional

from .base import ClassificationDataset


__all__ = [
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "FashionMNISTDataset",
    "ImageNetDataset",
    "MNISTDataset",
]


class CIFAR10Dataset(ClassificationDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import CIFAR10

        super().__init__(
            name='CIFAR-10',
            transform=transform,
            target_transform=target_transform,
        )

        dataset = CIFAR10(root, train, None, None, download)

        self.initialize(dataset)


class CIFAR100Dataset(ClassificationDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import CIFAR100

        super().__init__(
            name='CIFAR-100',
            transform=transform,
            target_transform=target_transform,
        )

        dataset = CIFAR100(root, train, None, None, download)

        self.initialize(dataset)


class FashionMNISTDataset(ClassificationDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import FashionMNIST

        super().__init__(
            name='Fashion-MNIST',
            transform=transform,
            target_transform=target_transform,
        )

        dataset = FashionMNIST(root, train, None, None, download)

        self.initialize(dataset)


class ImageNetDataset(ClassificationDataset):
    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        from torchvision.datasets import ImageNet

        super().__init__(
            name='ImageNet',
            transform=transform,
            target_transform=target_transform,
        )

        dataset = ImageNet(root, split)
        setattr(dataset, 'data', dataset.imgs)

        self.initialize(dataset)


class MNISTDataset(ClassificationDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        from torchvision.datasets import MNIST

        super().__init__(
            name='MNIST',
            transform=transform,
            target_transform=target_transform,
        )

        dataset = MNIST(root, train, None, None, download)

        self.initialize(dataset)
