import torch

from pytools.models import ImageNetClassificationModel, default_weight_specification, weights_dir, Replacer


model_name = "resnet50"
target = "BatchNorm2d"
to = "LayerNorm"
use_cuda = True
machine = "cuda" if use_cuda else "cpu"


model = ImageNetClassificationModel(
    name=model_name,
    pretrained=True,
    specify_weights=default_weight_specification,
    weights_dir=weights_dir,
    mode="eval",
    use_cuda=use_cuda,
)

Replacer(
    target=target,
    to=to,
    use_cuda=use_cuda,
)(model.model)

print(model.model)


# To check if the replacement is done properly
dummy_data = getattr(torch.zeros(size=(1, 3, 224, 224)), machine)()
model(dummy_data)

print("Replacement successful.")
