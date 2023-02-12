from typing import Callable, List, Optional, Tuple

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
            name="CIFAR-10",
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
            name="CIFAR-100",
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
            name="Fashion-MNIST",
            transform=transform,
            target_transform=target_transform,
        )

        dataset = FashionMNIST(root, train, None, None, download)

        self.initialize(dataset)


class ImageNetDataset(ClassificationDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        from torchvision.datasets import ImageNet
        from ..tools import get_file_list, make_attrdict

        super().__init__(
            name="ImageNet",
            transform=transform,
            target_transform=target_transform,
        )

        if split in get_file_list(root, fext=None):
            path_and_class = self.make_dataset(root, split)

            dataset = make_attrdict({
                "data": path_and_class,
                "targets": [v[-1] for v in path_and_class],
            })

        else:
            dataset = ImageNet(root, split)
            setattr(dataset, "data", dataset.imgs)

        self.initialize(dataset)

    @staticmethod
    def make_dataset(
            root: str,
            split: str = "train",
    ) -> List[Tuple[str, int]]:
        from torchvision.datasets import DatasetFolder

        extensions = [".JPEG",]

        datasetfolder = DatasetFolder(
            root=f"{root}/{split}",
            loader=None,
            extensions=extensions,
            transform=None,
            target_transform=None,
            is_valid_file=None,
        )

        _, class_to_idx = datasetfolder.find_classes(directory=f"{root}/{split}")

        path_and_class = datasetfolder.make_dataset(
            directory=f"{root}/{split}",
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=None,
        )

        return path_and_class


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
            name="MNIST",
            transform=transform,
            target_transform=target_transform,
        )

        dataset = MNIST(root, train, None, None, download)

        self.initialize(dataset)
