from ..replace import reshape_classifier_output
from .base import ClassificationModel


__all__ = [
    "CIFAR10ClassificationModel",
    "CIFAR100ClassificationModel",
    "ImageNetClassificationModel",
]


class CIFAR10ClassificationModel(ClassificationModel):
    def __init__(
            self,
            name: str,
            weights_path: str = None,
            mode: str = None,
            use_cuda: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            pretrained=False,
            specify_weights=None,
            weights_dir=None,
            mode=mode,
            use_cuda=use_cuda,
        )

        reshape_classifier_output(
            model=self.model,
            out_features=10,
            use_cuda=use_cuda,
        )

        if weights_path is not None:
            self.load_state_dict(weights_path)


class CIFAR100ClassificationModel(ClassificationModel):
    def __init__(
            self,
            name: str,
            weights_path: str = None,
            mode: str = None,
            use_cuda: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            pretrained=False,
            specify_weights=None,
            weights_dir=None,
            mode=mode,
            use_cuda=use_cuda,
        )

        reshape_classifier_output(
            model=self.model,
            out_features=100,
            use_cuda=use_cuda,
        )

        if weights_path is not None:
            self.load_state_dict(weights_path)


class ImageNetClassificationModel(ClassificationModel):
    def __init__(
            self,
            name: str,
            pretrained: bool = False,
            specify_weights: str = None,
            weights_dir: str = None,
            mode: str = None,
            use_cuda: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            pretrained=pretrained,
            specify_weights=specify_weights,
            weights_dir=weights_dir,
            mode=mode,
            use_cuda=use_cuda,
        )
