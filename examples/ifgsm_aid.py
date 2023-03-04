import torchvision.transforms as tf

from yupeeee_pytools.attacks import IFGSM
from yupeeee_pytools.datasets import ImageNetDataset, normalize
from yupeeee_pytools.models import ImageNetClassificationModel
from yupeeee_pytools.tools import plot_multiple_images, set_random_seed, repeat_tensor


imagenet_root = "D:/dataset/ImageNet"
imagenet_class = 208
model_name = "resnet50"
use_cuda = True
machine = "cuda" if use_cuda else "cpu"

epsilon = 10
iteration = 100
clip_per_iter = True


set_random_seed(seed=0)

dataset = ImageNetDataset(
    root=imagenet_root,
    split="val",
    transform=tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
    ]),
    target_transform=None,
)
images, targets = dataset.data_and_targets_of_class_c(imagenet_class)

normalizer = normalize["ImageNet"]

model = ImageNetClassificationModel(
    name=model_name,
    pretrained=True,
    specify_weights="IMAGENET1K_V1",
    weights_dir=None,
    mode="eval",
    use_cuda=use_cuda,
)

ifgsm = IFGSM(
    model=model.model,
    epsilon=epsilon,
    iteration=iteration,
    aid=True,
    normalizer=normalizer,
    clip_per_iter=clip_per_iter,
    use_cuda=use_cuda,
)


# Acc@1 (original)
preds, _ = model.predict(normalizer(images).cuda())
acc = sum(preds == targets) / len(targets)
print(
    f"Acc@1 of {model_name} "
    f"on class {imagenet_class} images "
    f"of ImageNet validation set: "
    f"{acc * 100:.2f}%"
)
print("Plotting original images...\n")
plot_multiple_images(images.permute(0, 2, 3, 1))


# Acc@1 (I-FGSM aid)
_images = ifgsm(
    data=images,
    targets=targets,
    verbose=True,
)

_preds, _ = model.predict(normalizer(_images).cuda())
_acc = sum(_preds == targets) / len(targets)
print(
    f"Acc@1 after aid: "
    f"{_acc * 100:.2f}%"
)
print("Plotting aided images...\n")
plot_multiple_images(_images.permute(0, 2, 3, 1))


# Acc@1 (Universal I-FGSM aid)
universal_perturbation = ifgsm.generate_universal_perturbation(
    data=images,
    targets=targets,
    batch_size=None,
    verbose=True,
)

_images = (images + repeat_tensor(
    tensor=universal_perturbation,
    repeat=len(images),
    dim=0,
)).clamp(0, 1)

_preds, _ = model.predict(normalizer(_images).cuda())
_acc = sum(_preds == targets) / len(targets)
print(
    f"Acc@1 after universal aid: "
    f"{_acc * 100:.2f}%"
)
print("Plotting universal aided images...\n")
plot_multiple_images(_images.permute(0, 2, 3, 1))
