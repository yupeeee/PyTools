from .base import ClassificationModel


__all__ = [
    "ImageNetClassificationModel",
]


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
