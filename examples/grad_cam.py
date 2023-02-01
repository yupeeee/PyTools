import numpy as np

from pytools.datasets import ImageNetDataset, normalize, denormalize
from pytools.models import ImageNetClassificationModel, Preprocess, default_weight_specification, weights_dir
from pytools.tools import plot_multiple_images


ImageNet_dir = "D:/dataset/ImageNet"
model_name = "resnet50"
layer_kind = "Conv2d"
alpha = 0.5
class_index = 208
weight_specification = default_weight_specification
use_cuda = True
machine = "cuda" if use_cuda else "cpu"


dataset = ImageNetDataset(
    root=ImageNet_dir,
    split="val",
    transform=Preprocess(
        model_name=model_name,
        specify_weights=default_weight_specification,
    ),
    target_transform=None,
)

normalizer = normalize[dataset.name]
denormalizer = denormalize[dataset.name]

model = ImageNetClassificationModel(
    name=model_name,
    pretrained=True,
    specify_weights=default_weight_specification,
    weights_dir=weights_dir,
    mode="eval",
    use_cuda=use_cuda,
)

data, targets = dataset.data_and_targets_of_class_c(c=class_index)

layer_names = list(model.dissect(dummy_data=getattr(data, machine)()).keys())
indices = np.arange(len(layer_names)) if layer_kind == "all" else \
    [i for i in np.arange(len(layer_names)) if layer_kind in layer_names[i]]

cam = model.grad_cam(
    data=getattr(data, machine)(),
    targets=targets,
    indices=indices,
    colormap="jet",
    aug_smooth=False,
    eigen_smooth=False,
)

plot_multiple_images(
    images=denormalizer(data).permute(0, 2, 3, 1) * (1 - alpha) + cam * alpha,
    desired_hw_ratio=(1, 1),
)
