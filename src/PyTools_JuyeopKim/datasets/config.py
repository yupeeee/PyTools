# supported datasets
datasets = [
    "CIFAR-10",
    "CIFAR-100",
    "Fashion-MNIST",
    "ImageNet",
    "MNIST",
]


# normalize & denormalize
standard = 'train'

means = {
    "CIFAR-10_train": [125.3069, 122.9501, 113.8660],
    "CIFAR-10_test": [126.0243, 123.7084, 114.8544],

    "CIFAR-100_train": [129.3039, 124.0699, 112.4336],
    "CIFAR-100_test": [129.7427, 124.2850, 112.6954],

    "Fashion-MNIST_train": [72.9404],
    "Fashion-MNIST_test": [73.1466],

    "ImageNet": [0.485, 0.456, 0.406],

    "MNIST_train": [33.3184],
    "MNIST_test": [33.7912],
}

stds = {
    "CIFAR-10_train": [62.9932, 62.0887, 66.7049],
    "CIFAR-10_test": [62.8964, 61.9375, 66.7061],

    "CIFAR-100_train": [68.1702, 65.3918, 70.4184],
    "CIFAR-100_test": [68.4042, 65.6278, 70.6594],

    "Fashion-MNIST_train": [90.0212],
    "Fashion-MNIST_test": [89.8733],

    "ImageNet": [0.229, 0.224, 0.225],

    "MNIST_train": [78.5675],
    "MNIST_test": [79.1725],
}

for dataset in datasets:
    if dataset == "ImageNet":
        continue

    means[f"{dataset}"] = [v / 255 for v in means[f"{dataset}_{standard}"]]
    stds[f"{dataset}"] = [v / 255 for v in stds[f"{dataset}_{standard}"]]
