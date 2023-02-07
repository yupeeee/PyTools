import tqdm

from pytools.datasets import ImageNetDataset
from pytools.models import \
    ImageNetClassificationModel, Preprocess, default_weight_specification, weights_dir, imagenet_model_names


ImageNet_dir = "D:/dataset/ImageNet"
model_name = "resnet50"
weight_specification = default_weight_specification
use_cuda = True
machine = "cuda" if use_cuda else "cpu"


dataset = ImageNetDataset(
    root=ImageNet_dir,
    split="val",
    transform=Preprocess(
        model_name=model_name,
        specify_weights=weight_specification,
    ),
    target_transform=None,
)

model = ImageNetClassificationModel(
    name=model_name,
    pretrained=True,
    specify_weights=weight_specification,
    weights_dir=weights_dir,
    mode="eval",
    use_cuda=True,
)

classes = list(set([int(v) for v in dataset.targets]))
classes.sort()

acc = 0

for c in tqdm.tqdm(
        classes,
        desc=f"Evaluating {imagenet_model_names[model_name]} on {dataset.name}...",
):
    data, targets = dataset.data_and_targets_of_class_c(c)
    preds, _ = model.predict(getattr(data, machine)())

    acc += sum(preds == targets)

acc = acc / len(dataset)

print(f"Acc@1 of {imagenet_model_names[model_name]} on {dataset.name}: {acc * 100:.2f}%")
