"""
python train_cifar100.py -model swin_t -config ./train_config.yaml -seed 0 -cuda -replace LayerNorm-BatchNorm2d,GELU-ReLU
"""
from argparse import ArgumentParser

from pytools.datasets import CIFAR100Dataset
from pytools.models import CIFAR100ClassificationModel, Replacer, \
    default_cifar100_train_preprocess, default_cifar100_val_preprocess
from pytools.tools import set_random_seed
from pytools.train import SupervisedLearner


dataset_root = "D:/dataset/CIFAR-100"
# dataset_root = "/home/juyeopkim/dataset/CIFAR-100"
weights_save_root = "./weights"
log_save_root = "./logs"
weights_save_period = 10


def run():
    set_random_seed(seed)

    train_dataset = CIFAR100Dataset(
        root=dataset_root,
        train=True,
        transform=default_cifar100_train_preprocess,
        target_transform=None,
    )

    val_dataset = CIFAR100Dataset(
        root=dataset_root,
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

    if replacements is not None:
        for option in replacements:
            target, to = option.split("-")

            Replacer(
                target=target,
                to=to,
                use_cuda=use_cuda,
            )(model.model)

    trainer = SupervisedLearner(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        config_path=config_path,
        use_cuda=use_cuda,
    )

    if resume:
        trainer.resume(
            weights_save_root=weights_save_root,
            log_save_root=log_save_root,
            prev_datetime=prev_datetime,
            weights_save_period=weights_save_period,
        )
    else:
        trainer.run(
            weights_save_root=weights_save_root,
            log_save_root=log_save_root,
            weights_save_period=weights_save_period,
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-model", type=str)
    parser.add_argument("-config", type=str)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-cuda", action="store_true", default=False)

    parser.add_argument("-replace", type=str, default=None)

    parser.add_argument("-resume", type=str, default=None)

    args = parser.parse_args()

    model_name = args.model
    config_path = args.config
    seed = args.seed
    use_cuda = args.cuda
    machine = "cuda" if use_cuda else "cpu"

    replacements = args.replace

    if replacements is not None:
        replacements = replacements.split(",")

    prev_datetime = args.resume
    resume = False

    if prev_datetime is not None:
        resume = True

    run()
