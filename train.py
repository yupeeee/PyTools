"""
python train.py -dataset imagenet -model swin_t -config ./train_config.yaml -seed 0 -cuda
"""
from argparse import ArgumentParser

from pytools.datasets import *
from pytools.models import *
from pytools.tools import set_random_seed
from pytools.train import SupervisedLearner


dataset_root = "D:/dataset"
weights_save_root = "./weights"
log_save_root = "./logs"


def run():
    set_random_seed(seed)

    if dataset_name == "cifar10":
        train_dataset = CIFAR10Dataset(
            root=f"{dataset_root}/CIFAR-10",
            train=True,
            transform=default_cifar10_train_preprocess,
            target_transform=None,
        )

        val_dataset = CIFAR10Dataset(
            root=f"{dataset_root}/CIFAR-10",
            train=False,
            transform=default_cifar10_val_preprocess,
            target_transform=None,
        )

        model = CIFAR10ClassificationModel(
            name=model_name,
            weights_path=None,
            mode=None,
            use_cuda=use_cuda,
        )

    elif dataset_name == "cifar100":
        train_dataset = CIFAR100Dataset(
            root=f"{dataset_root}/CIFAR-100",
            train=True,
            transform=default_cifar100_train_preprocess,
            target_transform=None,
        )

        val_dataset = CIFAR100Dataset(
            root=f"{dataset_root}/CIFAR-100",
            train=False,
            transform=default_cifar100_val_preprocess,
            target_transform=None,
        )

        model = CIFAR100ClassificationModel(
            name=model_name,
            weights_path=None,
            mode=None,
            use_cuda=use_cuda,
        )

    elif dataset_name == "imagenet":
        train_dataset = ImageNetDataset(
            root=f"{dataset_root}/ImageNet",
            split="train",
            transform=default_imagenet_preprocess,
            target_transform=None,
        )

        val_dataset = ImageNetDataset(
            root=f"{dataset_root}/ImageNet",
            split="val",
            transform=default_imagenet_preprocess,
            target_transform=None,
        )

        model = ImageNetClassificationModel(
            name=model_name,
            pretrained=False,
            specify_weights=default_weight_specification,
            weights_dir=weights_dir,
            mode=None,
            use_cuda=use_cuda,
        )

    else:
        raise ValueError

    SupervisedLearner(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        config_path=config_path,
        use_cuda=use_cuda,
    ).run(
        weights_save_root=weights_save_root,
        log_save_root=log_save_root,
        weights_save_period=10,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-dataset", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-config", type=str)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-cuda", action="store_true", default=False)

    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    config_path = args.config
    seed = args.seed
    use_cuda = args.cuda
    machine = "cuda" if use_cuda else "cpu"

    run()
