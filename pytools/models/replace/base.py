from .config import *


__all__ = [
    "Replacer",
]


class Replacer:
    def __init__(
            self,
            target: str,
            to: str,
    ) -> None:
        self.target = target
        self.to = to

    def __call__(
            self,
            model,
    ):
        if (self.target, self.to) in activation_replacements:
            from . import activation as lib

        elif (self.target, self.to) in normalization_replacements:
            from . import normalization as lib

        else:
            raise ValueError

        replacer = getattr(lib, f"{self.target}_to_{self.to}")

        self.replace_layer(model, replacer)

        return model

    def replace_layer(
            self,
            model,
            replacer,
    ) -> None:
        for child_name, child in model.named_children():
            class_name = str(child.__class__).split(".")[-1].split("'")[0]

            if self.target == class_name:
                setattr(model, child_name, replacer(child))
            else:
                self.replace_layer(child, replacer)
