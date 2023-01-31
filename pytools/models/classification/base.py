from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .config import *


__all__ = [
    "load_pytorch_model_weights",
    "load_not_in_pytorch_model",
    "ClassificationModel",
]


def load_pytorch_model_weights(
        model_name: str,
        specify_weights: str = default_weight_specification,
) -> Any:
    assert model_name in list_of_pytorch_imagenet_models
    assert specify_weights is not None and specify_weights in weight_specifications

    weights = getattr(models, f"{imagenet_model_names[model_name]}_Weights")
    weights = getattr(weights, specify_weights)

    return weights


def load_not_in_pytorch_model(
        model_name: str,
        pretrained: bool = False,
        weights_dir: str = weights_dir,
) -> Any:
    assert model_name in list_of_not_in_pytorch_imagenet_models

    supported_models = not_in_pytorch_imagenet_models
    supported_model_names = not_in_pytorch_imagenet_model_names

    # DeiT
    if model_name in supported_models["DeiT"]:
        from .nets import deit

        model = getattr(deit, model_name)(pretrained=pretrained)

    # PoolFormer
    elif model_name in supported_models["PoolFormer"]:
        from .nets import poolformer

        model = getattr(poolformer, model_name)()

        if pretrained:
            state_dict = torch.load(f"{weights_dir}/{model_name}.pth.tar")
            model.load_state_dict(state_dict)

    # PVT
    elif model_name in supported_models["PVT"]:
        from .nets import pvt

        model = getattr(pvt, model_name)()

        if pretrained:
            state_dict = torch.load(f"{weights_dir}/{model_name}.pth")
            model.load_state_dict(state_dict)

    # MLP-Mixer
    elif model_name in supported_models["MLP-Mixer"]:
        from .nets.mixer import MlpMixer, CONFIGS

        model = MlpMixer(config=CONFIGS[supported_model_names[model_name]])

        if pretrained:
            model.load_from(np.load(f"{weights_dir}/imagenet1k_{supported_model_names[model_name]}.npz"))

    else:
        raise ValueError

    return model


class ClassificationModel:
    def __init__(
            self,
            name: str,
            pretrained: bool = False,
            specify_weights: str = None,
            weights_dir: str = None,
            mode: str = None,
            use_cuda: bool = False,
    ) -> None:
        assert mode in modes

        self.name = name
        self.pretrained = pretrained
        self.specify_weights = specify_weights
        self.weights_dir = weights_dir
        self.mode = mode
        self.machine = "cuda" if use_cuda else "cpu"

        self.model = self.load_model()

        self.softmax = nn.Softmax(dim=-1)

    def __call__(
            self,
            data: Any,
    ) -> Any:
        return self.model(data).detach()

    def load_model(
            self,
    ) -> Any:
        assert self.name in imagenet_models

        if self.name in list_of_pytorch_imagenet_models:
            model = getattr(models, self.name)

            if self.pretrained:
                weights = load_pytorch_model_weights(self.name, self.specify_weights)
                model = model(weights=weights)

        else:
            model = load_not_in_pytorch_model(self.name, self.pretrained, self.weights_dir)

        if self.mode is not None:
            model = getattr(model, self.mode)()

        model = getattr(model, self.machine)()

        return model

    def predict(
            self,
            data: Any,
    ) -> Tuple[Any, Any]:
        out = self.__call__(data)

        confs = self.softmax(out).to(torch.device("cpu"))

        preds = torch.argmax(confs, dim=-1)

        return preds, confs

    def dissect(
            self,
            dummy_data: Any,
    ) -> Dict[str, Any]:
        hooks = []
        layers = {}

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(layers)

                module_key = f"{class_name}-{module_idx + 1}"

                layers[module_key] = module

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
            ):
                hooks.append(module.register_forward_hook(hook))

        self.model.apply(register_hook)
        self.model(dummy_data)

        for hook in hooks:
            hook.remove()

        return layers

    def x_ray(
            self,
            data: Any,
    ) -> Tuple[Dict, Dict]:
        hooks = []
        inputs = {}
        outputs = {}

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(outputs)

                module_key = f"{class_name}-{module_idx + 1}"

                if isinstance(input, tuple):
                    input = [v for v in input if v is not None]
                    input = torch.cat(input, dim=0)

                if isinstance(output, tuple):
                    output = [v for v in output if v is not None]
                    output = torch.cat(output, dim=0)

                inputs[module_key] = input.detach().cpu()
                outputs[module_key] = output.detach().cpu()

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
            ):
                hooks.append(module.register_forward_hook(hook))

        self.model.apply(register_hook)
        self.model(data)

        for hook in hooks:
            hook.remove()

        return inputs, outputs
